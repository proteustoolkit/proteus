import proteus
from proteus import Profiling
import numpy as np
from proteus.Transport import OneLevelTransport
from proteus.Transport import pi
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
from proteus.Transport import OneLevelTransport, C0_AffineQuadraticOnSimplexWithNodalBasis
from proteus.Transport import logEvent
from proteus.TransportCoefficients import TC_base
from proteus.SubgridError import SGE_base
from proteus.ShockCapturing import ShockCapturing_base
from . import cRANS3PF
from . import cRANS3PF2D
from . import cArgumentsDict
import proteus.NumericalFlux
import proteus.Profiling
import proteus.ShockCapturing
import proteus.SubgridError
import proteus.Transport
import proteus.TransportCoefficients

class SubgridError(proteus.SubgridError.SGE_base):

    def __init__(
            self,
            coefficients,
            nd,
            lag=False,
            nStepsToDelay=0,
            hFactor=1.0,
            noPressureStabilization=False):
        self.noPressureStabilization = noPressureStabilization
        proteus.SubgridError.SGE_base.__init__(self, coefficients, nd, lag)
        coefficients.stencil[0].add(0)
        self.hFactor = hFactor
        self.nStepsToDelay = nStepsToDelay
        self.nSteps = 0
        if self.lag:
            log("RANS3PF.SubgridError: lagging requested but must lag the first step; switching lagging off and delaying")
            self.nStepsToDelay = 1
            self.lag = False

    def initializeElementQuadrature(self, mesh, t, cq):
        import copy
        self.cq = cq
        self.v_last = self.cq[('velocity', 0)]

    def updateSubgridErrorHistory(self, initializationPhase=False):
        self.nSteps += 1
        if self.lag:
            self.v_last[:] = self.cq[('velocity', 0)]
        if self.lag == False and self.nStepsToDelay is not None and self.nSteps > self.nStepsToDelay:
            log("RANS3PF.SubgridError: switched to lagged subgrid error")
            self.lag = True
            self.v_last = self.cq[('velocity', 0)].copy()

    def calculateSubgridError(self, q):
        pass


class NumericalFlux(
        proteus.NumericalFlux.NavierStokes_Advection_DiagonalUpwind_Diffusion_SIPG_exterior):
    hasInterior = False

    def __init__(self, vt, getPointwiseBoundaryConditions,
                 getAdvectiveFluxBoundaryConditions,
                 getDiffusiveFluxBoundaryConditions,
                 getPeriodicBoundaryConditions=None):
        proteus.NumericalFlux.NavierStokes_Advection_DiagonalUpwind_Diffusion_SIPG_exterior.__init__(
            self,
            vt,
            getPointwiseBoundaryConditions,
            getAdvectiveFluxBoundaryConditions,
            getDiffusiveFluxBoundaryConditions,
            getPeriodicBoundaryConditions)
        self.penalty_constant = 2.0
        self.includeBoundaryAdjoint = True
        self.boundaryAdjoint_sigma = 1.0
        self.hasInterior = False


class ShockCapturing(proteus.ShockCapturing.DelayedLagState, proteus.ShockCapturing.ShockCapturing_base):

    def __init__(
            self,
            coefficients,
            nd,
            shockCapturingFactor=0.25,
            lag=False,
            nStepsToDelay=3):
        proteus.ShockCapturing.ShockCapturing_base.__init__(
            self, coefficients, nd, shockCapturingFactor, lag)
        self.nStepsToDelay = nStepsToDelay
        self.nSteps = 0
        if self.lag:
            log("RANS3PF.ShockCapturing: lagging requested but must lag the first step; switching lagging off and delaying")
            self.nStepsToDelay = 1
            self.lag = False

    def initializeElementQuadrature(self, mesh, t, cq):
        self.mesh = mesh
        self.numDiff = {}
        self.numDiff_last = {}
        for ci in range(0, 3):
            self.numDiff[ci] = cq[('numDiff', ci, ci)]
            self.numDiff_last[ci] = cq[('numDiff', ci, ci)]

    def updateShockCapturingHistory(self):
        self.nSteps += 1
        if self.lag:
            for ci in range(0, 3):
                self.numDiff_last[ci][:] = self.numDiff[ci]
        if self.lag == False and self.nStepsToDelay is not None and self.nSteps > self.nStepsToDelay:
            log("RANS3PF.ShockCapturing: switched to lagged shock capturing")
            self.lag = True
            for ci in range(0, 3):
                self.numDiff_last[ci] = self.numDiff[ci].copy()
        log(
            "RANS3PF: max numDiff_1 %e numDiff_2 %e numDiff_3 %e" %
            (globalMax(
                self.numDiff_last[0].max()), globalMax(
                self.numDiff_last[1].max()), globalMax(
                self.numDiff_last[2].max())))


class Coefficients(proteus.TransportCoefficients.TC_base):
    """
    The coefficients for two incompresslble fluids governed by the Navier-Stokes equations and separated by a sharp interface represented by a level set function
    """
    from proteus.ctransportCoefficients import TwophaseNavierStokes_ST_LS_SO_2D_Evaluate
    from proteus.ctransportCoefficients import TwophaseNavierStokes_ST_LS_SO_3D_Evaluate
    from proteus.ctransportCoefficients import TwophaseNavierStokes_ST_LS_SO_2D_Evaluate_sd
    from proteus.ctransportCoefficients import TwophaseNavierStokes_ST_LS_SO_3D_Evaluate_sd

    def __init__(self,
                 outputQuantDOFs = True,
                 INT_BY_PARTS_PRESSURE=0,
                 MULTIPLY_EXTERNAL_FORCE_BY_DENSITY=0,
                 CORRECT_VELOCITY=True,
                 USE_SUPG=1,
                 ARTIFICIAL_VISCOSITY=1,
                 cMax=1.0,  # For entropy viscosity (mql)
                 cE=1.0,  # For entropy viscosity (mql)
                 epsFact=1.5,
                 sigma=72.8,
                 rho_0=998.2, nu_0=1.004e-6,
                 rho_1=1.205, nu_1=1.500e-5,
                 g=[0.0, 0.0, -9.8],
                 nd=3,
                 ME_model=5,
                 PRESSURE_model=7,
                 VOS_model=0,
                 SED_model=5,
                 CLSVOF_model=None,
                 LS_model=None,
                 VOF_model=None,
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
                 dragAlpha=0.0,
                 dragBeta=0.0,
                 setParamsFunc=None,  # uses setParamsFunc if given
                 dragAlphaTypes=None,  # otherwise can use element constant values
                 dragBetaTypes=None,  # otherwise can use element constant values
                 porosityTypes=None,
                 killNonlinearDrag=False,
                 epsFact_solid=None,
                 eb_adjoint_sigma=1.0,
                 eb_penalty_constant=10.0,
                 forceStrongDirichlet=False,
                 turbulenceClosureModel=0,
                 # 0=No Model, 1=Smagorinksy, 2=Dynamic Smagorinsky,
                 # 3=K-Epsilon, 4=K-Omega
                 smagorinskyConstant=0.1,
                 barycenters=None,
                 PSTAB=0.0,
                 set_vos=None,
                 set_sed_velocity=None,
                 aDarcy=150.0,
                 betaForch=0.0,
                 grain=0.0102,
                 packFraction=0.2,
                 packMargin=0.01,
                 maxFraction=0.635,
                 frFraction=0.57,
                 sigmaC=1.1,
                 C3e=1.2,
                 C4e=1.0,
                 eR=0.8,
                 fContact=0.02,
                 mContact=2.0,
                 nContact=5.0,
                 angFriction=pi/6.0,
                 nParticles=0,
                 particle_epsFact=3.0,
                 particle_alpha=1000.0,
                 particle_beta=1000.0,
                 particle_penalty_constant=1000.0,
                 particle_nitsche=1.0,
                 particle_sdfList=None,
                 vos_function=None,
                 particle_velocityList=[],
                 granular_sdf_Calc=None,
                 granular_vel_Calc=None,
                 vos_limiter = 0.05,
                 mu_fr_limiter = 1.00,
                 use_sbm=0,
                 use_ball_as_particle=0,
                 ball_center=None,
                 ball_radius=None,
                 ball_velocity=None,
                 ball_angular_velocity=None,
                 particles=None,
                 useExact=False,
                 initialize=True):
        self.useExact=useExact
        self.outputQuantDOFs=outputQuantDOFs
        self.INT_BY_PARTS_PRESSURE=INT_BY_PARTS_PRESSURE
        self.MULTIPLY_EXTERNAL_FORCE_BY_DENSITY=MULTIPLY_EXTERNAL_FORCE_BY_DENSITY
        self.CORRECT_VELOCITY = CORRECT_VELOCITY
        self.nParticles = nParticles
        self.particle_nitsche = particle_nitsche
        self.particle_epsFact = particle_epsFact
        self.particle_alpha = particle_alpha
        self.particle_beta = particle_beta
        self.particle_penalty_constant = particle_penalty_constant
        self.particle_sdfList = particle_sdfList
        self.particle_velocityList = particle_velocityList
        self.granular_sdf_Calc = granular_sdf_Calc
        self.granular_vel_Calc = granular_vel_Calc
        self.aDarcy = aDarcy
        self.betaForch = betaForch
        self.grain = grain
        self.packFraction = packFraction
        self.packMargin = packMargin
        self.maxFraction = maxFraction
        self.frFraction = frFraction
        self.sigmaC = sigmaC
        self.C3e = C3e
        self.C4e = C4e
        self.eR = eR
        self.fContact = fContact
        self.mContact = mContact
        self.nContact = nContact
        self.angFriction = angFriction
        self.set_vos = set_vos
        self.set_sed = set_sed_velocity
        self.PSTAB = PSTAB
        self.barycenters = barycenters
        self.smagorinskyConstant = smagorinskyConstant
        self.turbulenceClosureModel = turbulenceClosureModel
        self.forceStrongDirichlet = forceStrongDirichlet
        self.eb_adjoint_sigma = eb_adjoint_sigma
        self.eb_penalty_constant = eb_penalty_constant
        self.movingDomain = movingDomain
        self.epsFact_solid = epsFact_solid
        self.useConstant_he = useConstant_he
        self.useVF = useVF
        self.useRBLES = useRBLES
        self.useMetrics = useMetrics
        self.sd = sd
        self.vos_limiter = vos_limiter
        self.mu_fr_limiter = mu_fr_limiter
        self.epsFact_density = epsFact_density
        self.stokes = stokes
        self.ME_model = ME_model
        self.PRESSURE_model = PRESSURE_model
        self.VOS_model = VOS_model
        self.SED_model = SED_model
        self.CLSVOF_model = CLSVOF_model
        self.LS_model = LS_model
        self.VOF_model = VOF_model
        self.KN_model = KN_model
        self.Closure_0_model = Closure_0_model
        self.Closure_1_model = Closure_1_model
        self.epsFact = epsFact
        self.eps = None
        self.sigma = sigma
        self.rho_0 = rho_0
        self.nu_0 = nu_0
        # cek for debugging using single phase test problems
        self.rho = rho_0
        self.nu = nu_0
        self.rho_1 = rho_1
        self.nu_1 = nu_1
        self.g = np.array(g)
        self.nd = nd
        #
        # mql: for entropy viscosity
        assert (USE_SUPG==0 or USE_SUPG==1), "USE_SUPG must be 0, or 1"
        self.USE_SUPG=USE_SUPG
        assert (ARTIFICIAL_VISCOSITY>=0 and ARTIFICIAL_VISCOSITY<=4), "ARTIFICIAL_VISCOSITY must be 0,1, 2, 3 or 4"
        self.ARTIFICIAL_VISCOSITY=ARTIFICIAL_VISCOSITY
        #0: no art viscosity,
        #1: shock capturing,
        #2: cell based entropy viscosity
        #3: edge based with smoothness indicator
        #4: edge based with EV
        self.cMax = cMax
        self.cE = cE
        #
        self.dragAlpha = dragAlpha
        self.dragBeta = dragBeta
        self.setParamsFunc = setParamsFunc
        self.dragAlphaTypes = dragAlphaTypes
        self.dragBetaTypes = dragBetaTypes
        self.porosityTypes = porosityTypes
        self.killNonlinearDrag = int(killNonlinearDrag)
        self.linearDragFactor = 1.0
        self.nonlinearDragFactor = 1.0
        #
        self.use_sbm = use_sbm

        self.particles = particles
        self.use_ball_as_particle = use_ball_as_particle
        self.ball_center = ball_center
        self.ball_radius = ball_radius
        self.ball_velocity = ball_velocity
        self.ball_angular_velocity = ball_angular_velocity
        self.forceTerms = None
        if(nd == 2):
            self.variableNames=['u','v']
        else:
            self.variableNames= ['u', 'v','w']

        if initialize:
            self.initialize()

    def initialize(self):
        if self.epsFact_density is None:
            self.epsFact_density = self.epsFact
        if self.killNonlinearDrag:
            self.nonlinearDragFactor = 0.0
        if self.particles:
            self.nParticles = self.particles.size();
        if self.ball_center is None:
            self.ball_center = 1e10*np.ones((self.nParticles,3),'d')
        if self.ball_radius is None:
            self.ball_radius = 1e10*np.ones((self.nParticles,1),'d')
        if self.ball_velocity is None:
            self.ball_velocity = np.zeros((self.nParticles,3),'d')
        if self.ball_angular_velocity is None:
            self.ball_angular_velocity = np.zeros((self.nParticles,3),'d')
        #
        nd = self.nd
        sd = self.sd
        mass = {}
        advection = {}
        diffusion = {}
        potential = {}
        reaction = {}
        hamiltonian = {}
        if nd == 2:
            variableNames = ['u', 'v']
            mass = {0: {0: 'linear'},
                    1: {1: 'linear'}}
            advection = {0: {0: 'nonlinear',
                             1: 'nonlinear'},
                         1: {0: 'nonlinear',
                             1: 'nonlinear'}}
            diffusion = {0: {0: {0: 'constant'}, 1: {1: 'constant'}},
                         1: {1: {1: 'constant'}, 0: {0: 'constant'}}}
            sdInfo = {(0, 0): (np.array([0, 1, 2], dtype='i'),
                               np.array([0, 1], dtype='i')),
                      (0, 1): (np.array([0, 0, 1], dtype='i'),
                               np.array([0], dtype='i')),
                      (1, 1): (np.array([0, 1, 2], dtype='i'),
                               np.array([0, 1], dtype='i')),
                      (1, 0): (np.array([0, 1, 1], dtype='i'),
                               np.array([1], dtype='i'))}
            potential = {0: {0: 'u'},
                         1: {1: 'u'}}
            reaction = {0: {0: 'nonlinear', 1: 'nonlinear'},
                        1: {0: 'nonlinear', 1: 'nonlinear'}}
            hamiltonian = {0: {0: 'linear'},
                           1: {1: 'linear'}}
            TC_base.__init__(self,
                             2,
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames,
                             sparseDiffusionTensors=sdInfo,
                             useSparseDiffusion=sd,
                             movingDomain=self.movingDomain)
            self.vectorComponents = [0, 1]
            self.vectorName = "velocity"
        elif nd == 3:
            variableNames = ['u', 'v', 'w']
            mass = {0: {0: 'linear'},
                    1: {1: 'linear'},
                    2: {2: 'linear'}}
            advection = {0: {0: 'nonlinear',
                             1: 'nonlinear',
                             2: 'nonlinear'},
                         1: {0: 'nonlinear',
                             1: 'nonlinear',
                             2: 'nonlinear'},
                         2: {0: 'nonlinear',
                             1: 'nonlinear',
                             2: 'nonlinear'}}
            diffusion = {0: {0: {0: 'constant'},
                             1: {1: 'constant'},
                             2: {2: 'constant'}},
                         1: {0: {0: 'constant'},
                             1: {1: 'constant'},
                             2: {2: 'constant'}},
                         2: {0: {0: 'constant'},
                             1: {1: 'constant'},
                             2: {2: 'constant'}}}
            sdInfo = {}
            sdInfo = {(0, 0): (np.array([0, 1, 2, 3], dtype='i'), np.array([0, 1, 2], dtype='i')),
                      (0, 1): (np.array([0, 0, 1, 1], dtype='i'), np.array([0], dtype='i')),
                      (0, 2): (np.array([0, 0, 0, 1], dtype='i'), np.array([0], dtype='i')),
                      (1, 0): (np.array([0, 1, 1, 1], dtype='i'), np.array([1], dtype='i')),
                      (1, 1): (np.array([0, 1, 2, 3], dtype='i'), np.array([0, 1, 2], dtype='i')),
                      (1, 2): (np.array([0, 0, 0, 1], dtype='i'), np.array([1], dtype='i')),
                      (2, 0): (np.array([0, 1, 1, 1], dtype='i'), np.array([2], dtype='i')),
                      (2, 1): (np.array([0, 0, 1, 1], dtype='i'), np.array([2], dtype='i')),
                      (2, 2): (np.array([0, 1, 2, 3], dtype='i'), np.array([0, 1, 2], dtype='i'))}
            potential = {0: {0: 'u'},
                         1: {1: 'u'},
                         2: {2: 'u'}}
            reaction = {0: {0: 'nonlinear', 1: 'nonlinear', 2: 'nonlinear'},
                        1: {0: 'nonlinear', 1: 'nonlinear', 2: 'nonlinear'},
                        2: {0: 'nonlinear', 1: 'nonlinear', 2: 'nonlinear'}}
            hamiltonian = {0: {0: 'linear'},
                           1: {1: 'linear'},
                           2: {2: 'linear'}}
            TC_base.__init__(self,
                             3,
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames,
                             sparseDiffusionTensors=sdInfo,
                             useSparseDiffusion=sd,
                             movingDomain=self.movingDomain)
            self.vectorComponents = [0, 1, 2]
            self.vectorName = "velocity"
        # DEM particles
        self.particle_netForces = np.zeros((3*self.nParticles, 3), 'd')#####[total_force_1,total_force_2,...,stress_1,stress_2,...,pressure_1,pressure_2,...]
        self.particle_netMoments = np.zeros((self.nParticles, 3), 'd')
        self.particle_surfaceArea = np.zeros((self.nParticles,), 'd')
        self.particle_centroids = np.zeros((self.nParticles, 3), 'd')

    def attachModels(self, modelList):
        # level set
        self.model = modelList[self.ME_model]
        self.model.q['phi_solid'] = self.q_phi_solid
        self.q_rho = self.model.q[('u', 0)].copy()
        self.ebqe_rho = self.model.ebqe[('u', 0)].copy()
        self.q_nu = self.model.q[('u', 0)].copy()
        self.ebqe_nu = self.model.ebqe[('u', 0)].copy()

        self.phisField = np.ones(self.model.q[('u', 0)].shape, 'd') * 1e10
        self.ebq_global_phi_s = np.ones_like(self.model.ebq_global[('totalFlux',0)]) * 1.e10
        self.ebq_global_grad_phi_s = np.ones(self.model.ebq_global[('velocityAverage',0)].shape[:-1]+(3,),'d') * 1.e10
        self.ebq_particle_velocity_s = np.ones(self.model.ebq_global[('velocityAverage',0)].shape[:-1]+(3,),'d') * 1.e10

        # DEM particles
        self.particle_signed_distances = np.zeros((self.nParticles,) + self.model.q[('u', 0)].shape, 'd')
        self.particle_signed_distance_normals = np.zeros((self.nParticles,) + self.model.q[('velocity', 0)].shape[:-1]+(3,), 'd')
        self.particle_velocities = np.zeros((self.nParticles,) + self.model.q[('velocity', 0)].shape[:-1]+(3,), 'd')
        # This is making a special case for granular material simulations
        # if the user inputs a list of position/velocities then the sdf are calculated based on the "spherical" particles
        # otherwise the sdf are calculated based on the input sdf list for each body
        if self.use_ball_as_particle==1:
            pass
        elif self.particles is not None:
            self.particles.updateSDF(self.mesh.nodeArray,
                                     self.model.q['x'],
                                     self.model.ebq_global['x'],
                                     self.phi_s,
                                     self.particle_signed_distances,
                                     self.particle_signed_distance_normals,
                                     self.particle_velocities,
                                     self.particle_centroids,
                                     self.ebq_global_phi_s,
                                     self.ebq_global_grad_phi_s,
                                     self.ebq_particle_velocity_s)

            # This is a temporary hack... something weird is happening where the
            # particle centroids are not being updated by updateSDF. Will address later
            for i in range(self.particles.size()):
                self.particle_centroids[i,:] = self.particles[i].x()
        elif self.granular_sdf_Calc is not None:
            corresponding_point_on_boundary = np.zeros((3,),'d')
            temp_1 = np.zeros(self.model.q[('u', 0)].shape, 'd')
            temp_2 = np.zeros(self.model.q[('u', 0)].shape, 'd')
            temp_3 = np.zeros(self.model.q[('u', 0)].shape, 'd')
            for i in range(self.nParticles):
                logEvent("Attaching particle i={0}".format(i))
                for eN in range(self.model.q['x'].shape[0]):
                    for k in range(self.model.q['x'].shape[1]):
                        self.particle_signed_distances[i, eN, k], self.particle_signed_distance_normals[i,
                                                                                                        eN, k] = self.granular_sdf_Calc(self.model.q['x'][eN, k], i)
                        self.particle_velocities[i, eN, k] = self.granular_vel_Calc(self.model.q['x'][eN, k], i)
                # This is important to write the a field for the sdf in the domain which distinguishes solid particles
                temp_1 = np.minimum(abs(self.particle_signed_distances[i]), abs(self.phisField))
                temp_2 = np.minimum(self.particle_signed_distances[i], temp_1)
                temp_3 = np.minimum(temp_2, self.phisField)
                self.phisField = temp_3
                self.model.q[('phis')] = temp_3
                for ebN in range(self.model.ebq_global['x'].shape[0]):
                    for kb in range(self.model.ebq_global['x'].shape[1]):
                        sdf_ebN_kb,sdNormals = self.granular_sdf_Calc(self.model.ebq_global['x'][ebN,kb],i)
                        if ( abs(sdf_ebN_kb) < abs(self.ebq_global_phi_s[ebN,kb]) ):
                            self.ebq_global_phi_s[ebN,kb]=sdf_ebN_kb
                            self.ebq_global_grad_phi_s[ebN,kb,:]=sdNormals
                            for j in range(len(sdNormals)):
                                corresponding_point_on_boundary[j] = self.model.ebq_global['x'][ebN,kb][j] - sdf_ebN_kb*sdNormals[j]
                            self.ebq_particle_velocity_s[ebN,kb,:] = self.granular_vel_Calc(corresponding_point_on_boundary,i)
        elif self.nParticles > 0:
            corresponding_point_on_boundary = np.zeros((3,),'d')
            for i, sdf, vel in zip(list(range(self.nParticles)),
                                   self.particle_sdfList, self.particle_velocityList):
                for eN in range(self.model.q['x'].shape[0]):
                    for k in range(self.model.q['x'].shape[1]):
                        self.particle_signed_distances[i, eN, k], self.particle_signed_distance_normals[i, eN, k] = sdf(0.0, self.model.q['x'][eN, k])
                        self.particle_velocities[i, eN, k] = vel(0.0, self.model.q['x'][eN, k])
                self.model.q[('phis', i)] = self.particle_signed_distances[i]
                self.model.q[('phis_vel', i)] = self.particle_velocities[i]
                for ebN in range(self.model.ebq_global['x'].shape[0]):
                    for kb in range(self.model.ebq_global['x'].shape[1]):
                        sdf_ebN_kb,sdNormals = sdf(0.0, self.model.ebq_global['x'][ebN,kb],)
                        if ( abs(sdf_ebN_kb) < abs(self.ebq_global_phi_s[ebN,kb]) ):
                            self.ebq_global_phi_s[ebN,kb]=sdf_ebN_kb
                            self.ebq_global_grad_phi_s[ebN,kb,:]=sdNormals
                            for j in range(len(sdNormals)):
                                corresponding_point_on_boundary[j] = self.model.ebq_global['x'][ebN,kb][j] - sdf_ebN_kb*sdNormals[j]
                            self.ebq_particle_velocity_s[ebN,kb,:] = vel(0.0, corresponding_point_on_boundary)

        if self.PRESSURE_model is not None:
            self.model.pressureModel = modelList[self.PRESSURE_model]
            self.model.q_p_fluid = modelList[self.PRESSURE_model].q[('u', 0)]
            self.model.ebqe_p_fluid = modelList[self.PRESSURE_model].ebqe[('u', 0)]
            self.model.q_grad_p_fluid = modelList[self.PRESSURE_model].q[('grad(u)', 0)]
            self.model.ebqe_grad_p_fluid = modelList[self.PRESSURE_model].ebqe[('grad(u)', 0)]
        if self.VOS_model is not None:
            self.model.vos_dof = modelList[self.VOS_model].u[0].dof
            self.model.q_vos = modelList[self.VOS_model].q[('u',0)]
            self.model.q_dvos_dt = modelList[self.VOS_model].q[('mt',0)]
            self.model.ebqe_vos = modelList[self.VOS_model].ebqe[('u',0)]
            self.vos_dof = self.model.vos_dof
            self.q_vos = self.model.q_vos
            self.q_dvos_dt = self.model.q_dvos_dt
            self.ebqe_vos = self.model.ebqe_vos
            self.q_grad_vos = modelList[self.VOS_model].q[('grad(u)',0)]
        else:
            if self.VOF_model is None:
                self.vos_dof = modelList[self.ME_model].u[0].dof.copy()
                self.vos_dof[:] = 0.0
                self.q_vos = modelList[self.ME_model].q[('u', 0)].copy()
                self.q_grad_vos = modelList[self.ME_model].q[('grad(u)',0)].copy()
                self.q_vos[:] = 0.0
                self.q_dvos_dt = self.q_vos.copy()
                self.q_dvos_dt[:] = 0.0
                self.ebqe_vos = modelList[self.ME_model].ebqe[('u', 0)].copy()
                self.ebqe_vos[:] = 0.0
            else:
                self.vos_dof = modelList[self.VOF_model].u[0].dof.copy()
                self.vos_dof[:] = 0.0
                self.q_vos = modelList[self.VOF_model].coefficients.q_vos
                self.q_dvos_dt = self.q_vos.copy()
                self.q_dvos_dt[:] = 0.0
                self.q_grad_vos = modelList[self.VOF_model].q[('grad(u)',0)].copy()
                self.ebqe_vos = modelList[self.VOF_model].coefficients.ebqe_vos
        if self.SED_model is not None:
            self.sedModel = modelList[self.SED_model]
            self.rho_s = modelList[self.SED_model].coefficients.rho_s
            self.q_velocity_solid = modelList[self.SED_model].q[('velocity',0)]
            self.q_velocityStar_solid = modelList[self.SED_model].q[('velocityStar',0)]
            self.ebqe_velocity_solid = modelList[self.SED_model].ebqe[('velocity',0)]
        else:
            self.sedModel = None
            self.rho_s = self.rho_0
            self.q_velocity_solid = self.model.q[('velocity', 0)].copy()
            self.q_velocity_solid[:] = 0.0
            self.q_velocityStar_solid = self.model.q[('velocity', 0)].copy()
            self.q_velocityStar_solid[:] = 0.0
            self.ebqe_velocity_solid = self.model.ebqe[('velocity', 0)].copy()
            self.ebqe_velocity_solid[:] = 0.0
        if self.CLSVOF_model is not None: # use CLSVOF
            # LS part #
            self.q_phi = modelList[self.CLSVOF_model].q[('u', 0)]
            self.phi_dof = modelList[self.CLSVOF_model].u[0].dof
            self.ebq_phi = None # Not used. What is this for?
            self.ebqe_phi = modelList[self.CLSVOF_model].ebqe[('u', 0)]
            self.bc_ebqe_phi = modelList[self.CLSVOF_model].ebqe[('u', 0)] #Dirichlet BCs for level set. I don't have it since I impose 1 or -1. Therefore I attach the soln at boundary
            self.q_n = modelList[self.CLSVOF_model].q[('grad(u)', 0)]
            self.ebq_n = None # Not used. What is this for?
            self.ebqe_n = modelList[self.CLSVOF_model].ebqe[('grad(u)', 0)]
            # VOF part #
            self.q_vf = modelList[self.CLSVOF_model].q[('H(u)', 0)]
            self.ebq_vf = None# Not used. What is this for?
            self.ebqe_vf = modelList[self.CLSVOF_model].ebqe[('H(u)', 0)]
            self.bc_ebqe_vf = 0.5*(1.0+modelList[self.CLSVOF_model].numericalFlux.ebqe[('u',0)]) # Dirichlet BCs for VOF. What I have is BCs for Signed function
        else: # use NCLS-RDLS-VOF-MCORR instead
            if self.LS_model is not None: #LEVEL SET MODEL
                self.q_phi = modelList[self.LS_model].q[('u', 0)]
                self.phi_dof = modelList[self.LS_model].u[0].dof
                if ('u', 0) in modelList[self.LS_model].ebq:
                    self.ebq_phi = modelList[self.LS_model].ebq[('u', 0)]
                else:
                    self.ebq_phi = None
                self.ebqe_phi = modelList[self.LS_model].ebqe[('u', 0)]
                self.bc_ebqe_phi = modelList[
                    self.LS_model].numericalFlux.ebqe[
                        ('u', 0)]
                # normal
                self.q_n = modelList[self.LS_model].q[('grad(u)', 0)]
                if ('grad(u)', 0) in modelList[self.LS_model].ebq:
                    self.ebq_n = modelList[self.LS_model].ebq[('grad(u)', 0)]
                else:
                    self.ebq_n = None
                self.ebqe_n = modelList[self.LS_model].ebqe[('grad(u)', 0)]
            else:
                self.q_phi = 10.0 * np.ones(self.model.q[('u', 0)].shape, 'd')
                self.phi_dof = 10.0 * np.ones((self.model.mesh.nNodes_global,), 'd')
                self.ebqe_phi = 10.0 * \
                                np.ones(self.model.ebqe[('u', 0)].shape, 'd')
                self.bc_ebqe_phi = 10.0 * \
                                   np.ones(self.model.ebqe[('u', 0)].shape, 'd')
                self.q_n = np.ones(self.model.q[('velocity', 0)].shape, 'd')
                self.ebqe_n = np.ones(
                    self.model.ebqe[
                        ('velocity', 0)].shape, 'd')
            if self.VOF_model is not None: #VOF MODEL
                self.q_vf = modelList[self.VOF_model].q[('u', 0)]
                if ('u', 0) in modelList[self.VOF_model].ebq:
                    self.ebq_vf = modelList[self.VOF_model].ebq[('u', 0)]
                else:
                    self.ebq_vf = None
                self.ebqe_vf = modelList[self.VOF_model].ebqe[('u', 0)]
                self.bc_ebqe_vf = modelList[
                    self.VOF_model].numericalFlux.ebqe[
                        ('u', 0)]
            else:
                self.q_vf = np.zeros(self.model.q[('u', 0)].shape, 'd')
                self.ebqe_vf = np.zeros(self.model.ebqe[('u', 0)].shape, 'd')
                self.bc_ebqe_vf = np.zeros(self.model.ebqe[('u', 0)].shape, 'd')
        # curvature
        if self.KN_model is not None:
            self.q_kappa = modelList[self.KN_model].q[('u', 0)]
            self.ebqe_kappa = modelList[self.KN_model].ebqe[('u', 0)]
            if ('u', 0) in modelList[self.KN_model].ebq:
                self.ebq_kappa = modelList[self.KN_model].ebq[('u', 0)]
            else:
                self.ebq_kappa = None
        else:
            self.q_kappa = -np.ones(self.model.q[('u', 0)].shape, 'd')
            self.ebqe_kappa = -np.ones(self.model.ebqe[('u', 0)].shape, 'd')
        # Turbulence Closures
        # only option for now is k-epsilon
        self.q_turb_var = {}
        self.q_turb_var_grad = {}
        self.ebqe_turb_var = {}
        if self.Closure_0_model is not None:
            self.q_turb_var[0] = modelList[self.Closure_0_model].q[('u', 0)]
            self.q_turb_var_grad[0] = modelList[
                self.Closure_0_model].q[
                ('grad(u)', 0)]
            self.ebqe_turb_var[0] = modelList[
                self.Closure_0_model].ebqe[
                ('u', 0)]
        else:
            self.q_turb_var[0] = np.ones(self.model.q[('u', 0)].shape, 'd')
            self.q_turb_var_grad[0] = np.ones(
                self.model.q[('grad(u)', 0)].shape, 'd')
            self.ebqe_turb_var[0] = np.ones(
                self.model.ebqe[('u', 0)].shape, 'd')
        if self.Closure_1_model is not None:
            self.q_turb_var[1] = modelList[self.Closure_1_model].q[('u', 0)]
            self.ebqe_turb_var[1] = modelList[
                self.Closure_1_model].ebqe[
                ('u', 0)]
        else:
            self.q_turb_var[1] = np.ones(self.model.q[('u', 0)].shape, 'd')
            self.ebqe_turb_var[1] = np.ones(
                self.model.ebqe[('u', 0)].shape, 'd')
        if self.epsFact_solid is None:
            self.epsFact_solid = np.ones(
                self.model.mesh.elementMaterialTypes.max() + 1)
        assert len(self.epsFact_solid) > self.model.mesh.elementMaterialTypes.max(
        ), "epsFact_solid  array is not large  enough for the materials  in this mesh; length must be greater  than largest  material type ID"

    def initializeMesh(self, mesh):


        self.phi_s = np.ones(mesh.nodeArray.shape[0], 'd')*1e10

        logEvent("updating {0} particles...".format(self.nParticles))

        for i in range(self.nParticles):
            if self.use_ball_as_particle == 1:
                sdf = lambda x: (np.linalg.norm(x-self.ball_center[i]),0)
            else:
                if self.granular_sdf_Calc is not None:
                    sdf = lambda x: self.granular_sdf_Calc(x,i)
                else:
                    if self.particles is not None:
                        sdf = lambda x: self.particles[i].sdf(x)
                    else:
                        sdf = lambda x: self.particle_sdfList[i](0.0, x)

            for j in range(mesh.nodeArray.shape[0]):
                if self.use_ball_as_particle==1 or self.particle_sdfList is not None:
                    sdf_at_node, _ = sdf(mesh.nodeArray[j,:])
                elif self.particles is not None:
                    sdf_at_node = sdf(mesh.nodeArray[j,:])
                if (abs(sdf_at_node) < abs(self.phi_s[j])):
                    self.phi_s[j] = sdf_at_node

        # cek we eventually need to use the local element diameter
        self.eps_density = self.epsFact_density * mesh.h
        self.eps_viscosity = self.epsFact * mesh.h
        self.mesh = mesh
        self.elementMaterialTypes = mesh.elementMaterialTypes
        nBoundariesMax = int(
            globalMax(max(self.mesh.elementBoundaryUserFlags))) + 1
        self.wettedAreas = np.zeros((nBoundariesMax,), 'd')
        self.netForces_p = np.zeros((nBoundariesMax, 3), 'd')
        self.netForces_v = np.zeros((nBoundariesMax, 3), 'd')
        self.netMoments = np.zeros((nBoundariesMax, 3), 'd')
        if self.barycenters is None:
            self.barycenters = np.zeros((nBoundariesMax, 3), 'd')
        comm = Comm.get()
        if comm.isMaster():
            self.timeHistory = open(os.path.join(proteus.Profiling.logDir, "timeHistory.txt"), "w")
            self.wettedAreaHistory = open(os.path.join(proteus.Profiling.logDir, "wettedAreaHistory.txt"), "w")
            self.forceHistory_p = open(os.path.join(proteus.Profiling.logDir, "forceHistory_p.txt"), "w")
            self.forceHistory_v = open(os.path.join(proteus.Profiling.logDir, "forceHistory_v.txt"), "w")
            self.momentHistory = open(os.path.join(proteus.Profiling.logDir, "momentHistory.txt"), "w")
            if self.nParticles:
                self.particle_surfaceAreaHistory = open(os.path.join(proteus.Profiling.logDir, "particle_surfaceAreaHistory.txt"), "w")
                self.particle_forceHistory = open(os.path.join(proteus.Profiling.logDir, "particle_forceHistory.txt"), "w")
                self.particle_pforceHistory = open(os.path.join(proteus.Profiling.logDir, "particle_pforceHistory.txt"), "w")
                self.particle_vforceHistory = open(os.path.join(proteus.Profiling.logDir, "particle_vforceHistory.txt"), "w")
                self.particle_momentHistory = open(os.path.join(proteus.Profiling.logDir, "particle_momentHistory.txt"), "w")
    # initialize so it can run as single phase

    def initializeElementQuadrature(self, t, cq):
        # VRANS
        self.q_phi_solid = np.ones(cq[('u', 0)].shape, 'd')
        self.q_porosity = np.ones(cq[('u', 0)].shape, 'd')
        self.q_dragAlpha = np.ones(cq[('u', 0)].shape, 'd')
        self.q_dragAlpha.fill(self.dragAlpha)
        self.q_dragBeta = np.ones(cq[('u', 0)].shape, 'd')
        self.q_dragBeta.fill(self.dragBeta)
        if self.setParamsFunc is not None:
            self.setParamsFunc(
                cq['x'],
                self.q_vos,
                self.q_dragAlpha,
                self.q_dragBeta)
        else:
            #Leave porosity in when porosity types are used
            if self.porosityTypes is not None:
                for eN in range(self.q_porosity.shape[0]):
                    self.q_porosity[
                        eN, :] = self.porosityTypes[
                        self.elementMaterialTypes[eN]]
            if self.dragAlphaTypes is not None:
                for eN in range(self.q_dragAlpha.shape[0]):
                    self.q_dragAlpha[
                        eN, :] = self.dragAlphaTypes[
                        self.elementMaterialTypes[eN]]
            if self.dragBetaTypes is not None:
                for eN in range(self.q_dragBeta.shape[0]):
                    self.q_dragBeta[
                        eN, :] = self.dragBetaTypes[
                        self.elementMaterialTypes[eN]]
        cq['phisError'] = cq[('u',0)].copy()
        cq['velocityError'] = cq[('velocity',0)].copy()

    def initializeElementBoundaryQuadrature(self, t, cebq, cebq_global):
        # VRANS
        self.ebq_vos = np.zeros(cebq['det(J)'].shape, 'd')
        self.ebq_porosity = np.ones(cebq['det(J)'].shape, 'd')
        self.ebq_dragAlpha = np.ones(cebq['det(J)'].shape, 'd')
        self.ebq_dragAlpha.fill(self.dragAlpha)
        self.ebq_dragBeta = np.ones(cebq['det(J)'].shape, 'd')
        self.ebq_dragBeta.fill(self.dragBeta)
        if self.setParamsFunc is not None:
            self.setParamsFunc(
                cebq['x'],
                self.ebq_vos,
                self.ebq_dragAlpha,
                self.ebq_dragBeta)
        # TODO which mean to use or leave discontinuous
        # TODO make loops faster
        if self.porosityTypes is not None:
            for ebNI in range(self.mesh.nInteriorElementBoundaries_global):
                ebN = self.mesh.interiorElementBoundariesArray[ebNI]
                eN_left = self.mesh.elementBoundaryElementsArray[ebN, 0]
                eN_right = self.mesh.elementBoundaryElementsArray[ebN, 1]
                ebN_element_left = self.mesh.elementBoundaryLocalElementBoundariesArray[
                    ebN, 0]
                ebN_element_right = self.mesh.elementBoundaryLocalElementBoundariesArray[
                    ebN, 1]
                avg = 0.5 * (self.porosityTypes[self.elementMaterialTypes[eN_left]] +
                             self.porosityTypes[self.elementMaterialTypes[eN_right]])
                self.ebq_porosity[
                    eN_left, ebN_element_left, :] = self.porosityTypes[
                    self.elementMaterialTypes[eN_left]]
                self.ebq_porosity[
                    eN_right, ebN_element_right, :] = self.porosityTypes[
                    self.elementMaterialTypes[eN_right]]
            for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
                ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
                eN = self.mesh.elementBoundaryElementsArray[ebN, 0]
                ebN_element = self.mesh.elementBoundaryLocalElementBoundariesArray[
                    ebN, 0]
                self.ebq_porosity[
                    eN, ebN_element, :] = self.porosityTypes[
                    self.elementMaterialTypes[eN]]
        if self.dragAlphaTypes is not None:
            for ebNI in range(self.mesh.nInteriorElementBoundaries_global):
                ebN = self.mesh.interiorElementBoundariesArray[ebNI]
                eN_left = self.mesh.elementBoundaryElementsArray[ebN, 0]
                eN_right = self.mesh.elementBoundaryElementsArray[ebN, 1]
            ebN_element_left = self.mesh.elementBoundaryLocalElementBoundariesArray[
                ebN, 0]
            ebN_element_right = self.mesh.elementBoundaryLocalElementBoundariesArray[
                ebN, 1]
            avg = 0.5 * (self.dragAlphaTypes[self.elementMaterialTypes[eN_left]] +
                         self.dragAlphaTypes[self.elementMaterialTypes[eN_right]])
            self.ebq_dragAlpha[
                eN_left, ebN_element_left, :] = self.dragAlphaTypes[
                self.elementMaterialTypes[eN_left]]
            self.ebq_dragAlpha[
                eN_right, ebN_element_right, :] = self.dragAlphaTypes[
                self.elementMaterialTypes[eN_right]]
            for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
                ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
                eN = self.mesh.elementBoundaryElementsArray[ebN, 0]
                ebN_element = self.mesh.elementBoundaryLocalElementBoundariesArray[
                    ebN, 0]
                self.ebq_dragAlpha[
                    eN, ebN_element, :] = self.dragAlphaTypes[
                    self.elementMaterialTypes[eN]]
        if self.dragBetaTypes is not None:
            for ebNI in range(self.mesh.nInteriorElementBoundaries_global):
                ebN = self.mesh.interiorElementBoundariesArray[ebNI]
                eN_left = self.mesh.elementBoundaryElementsArray[ebN, 0]
                eN_right = self.mesh.elementBoundaryElementsArray[ebN, 1]
            ebN_element_left = self.mesh.elementBoundaryLocalElementBoundariesArray[
                ebN, 0]
            ebN_element_right = self.mesh.elementBoundaryLocalElementBoundariesArray[
                ebN, 1]
            avg = 0.5 * (self.dragBetaTypes[self.elementMaterialTypes[eN_left]] +
                         self.dragBetaTypes[self.elementMaterialTypes[eN_right]])
            self.ebq_dragBeta[
                eN_left, ebN_element_left, :] = self.dragBetaTypes[
                self.elementMaterialTypes[eN_left]]
            self.ebq_dragBeta[
                eN_right, ebN_element_right, :] = self.dragBetaTypes[
                self.elementMaterialTypes[eN_right]]
            for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
                ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
                eN = self.mesh.elementBoundaryElementsArray[ebN, 0]
                ebN_element = self.mesh.elementBoundaryLocalElementBoundariesArray[
                    ebN, 0]
                self.ebq_dragBeta[
                    eN, ebN_element, :] = self.dragBetaTypes[
                    self.elementMaterialTypes[eN]]

    def initializeGlobalExteriorElementBoundaryQuadrature(self, t, cebqe):
        # VRANS
        log("ebqe_global allocations in coefficients")
        self.ebqe_velocity_last = np.zeros(cebqe[('velocity',0)].shape)
        self.ebqe_dragAlpha = np.ones(cebqe[('u', 0)].shape, 'd')
        self.ebqe_dragAlpha.fill(self.dragAlpha)
        self.ebqe_dragBeta = np.ones(cebqe[('u', 0)].shape, 'd')
        self.ebqe_dragBeta.fill(self.dragBeta)
        log("porosity and drag")
        # TODO make loops faster
        if self.setParamsFunc is not None:
            self.setParamsFunc(
                cebqe['x'],
                self.ebqe_vos,
                self.ebqe_dragAlpha,
                self.ebqe_dragBeta)
        else:
            if self.porosityTypes is not None:
                for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
                    ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
                    eN = self.mesh.elementBoundaryElementsArray[ebN, 0]
                    self.ebqe_porosity[
                        ebNE, :] = self.porosityTypes[
                        self.elementMaterialTypes[eN]]
                #convert from porosity to volume of sediment - ASD: Leave porosity in for porosity Types for now!!!
#                self.ebqe_vos -= 1.0
#                self.ebqe_vos *= -1.0
            if self.dragAlphaTypes is not None:
                for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
                    ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
                    eN = self.mesh.elementBoundaryElementsArray[ebN, 0]
                    self.ebqe_dragAlpha[
                        ebNE, :] = self.dragAlphaTypes[
                        self.elementMaterialTypes[eN]]
            if self.dragBetaTypes is not None:
                for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
                    ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
                    eN = self.mesh.elementBoundaryElementsArray[ebN, 0]
                    self.ebqe_dragBeta[
                        ebNE, :] = self.dragBetaTypes[
                        self.elementMaterialTypes[eN]]

    def updateToMovingDomain(self, t, c):
        pass
      
    def evaluate(self, t, c):
        pass

    #: what preStep and postStep leave on the level model for the next step: the entropy residual of the
    #: last step, the two previous solutions (for the extrapolated star velocity), dt, the first-step flag,
    #: and the velocity at the quadrature points as it was after the last step and before it. A hot start
    #: rebuilds them from the restart state, which is not the state the archived step ended with.
    _state_model_arrays = ("entropyResidualPerNode", "u_dof_old", "v_dof_old", "w_dof_old",
                           "u_dof_old_old", "v_dof_old_old", "w_dof_old_old")
    _state_q_arrays = (('velocity', 0), ('velocityOld', 0))

    def state(self):
        m = getattr(self, "model", None)
        if m is None or not hasattr(m, "entropyResidualPerNode"):
            return None
        st = {"dt_last": float(getattr(m, "dt_last", 0.0)), "firstStep": bool(m.firstStep)}
        for name in self._state_model_arrays:
            if hasattr(m, name):
                st[name] = getattr(m, name).copy()
        for key in self._state_q_arrays:
            st["q_%s_%d" % key] = m.q[key].copy()
        return st

    def set_state(self, state):
        m = self.model
        m.dt_last = state["dt_last"]
        m.firstStep = bool(state["firstStep"])
        for name in self._state_model_arrays:
            if name in state and hasattr(m, name):
                getattr(m, name)[:] = state[name]
        for key in self._state_q_arrays:
            m.q[key][:] = state["q_%s_%d" % key]

    def preStep(self, t, firstStep=False):
        if firstStep:
            self.model.entropyResidualPerNode[:] = 1.E15 # first step use low order stab
        #
        self.model.laggedEntropyResidualPerNode[:] = self.model.entropyResidualPerNode[:]
        ###########################
        # COMPUTE STAR VELOCITIES #
        ###########################
        # Compute 2nd order extrapolation on velocity
        if (firstStep):
            self.model.uStar_dof[:] = self.model.u[0].dof
            self.model.vStar_dof[:] = self.model.u[1].dof
            if (self.model.nSpace_global == 3):
                self.model.wStar_dof[:] = self.model.u[2].dof
            #
            self.model.q[('velocityStar', 0)][:] = self.model.q[('velocity', 0)]
        else:
            if self.model.timeIntegration.timeOrder == 1:
                r = 1.
            else:
                r = self.model.timeIntegration.dt/self.model.timeIntegration.dt_history[0]
            #
            self.model.uStar_dof[:] = (1+r)*self.model.u[0].dof[:] - r*self.model.u_dof_old[:]
            self.model.vStar_dof[:] = (1+r)*self.model.u[1].dof[:] - r*self.model.v_dof_old[:]
            if (self.model.nSpace_global == 3):
                self.model.wStar_dof[:] = (1+r)*self.model.u[2].dof[:] - r*self.model.w_dof_old[:]
            self.model.q[('velocityStar', 0)][:] = (1 + r) * self.model.q[('velocity', 0)] - r * self.model.q[('velocityOld', 0)]
        self.model.q[('velocityOld', 0)][:] = self.model.q[('velocity', 0)]

        ######################
        # SAVE OLD SOLUTIONS #
        ######################
        # solution at tnm1
        self.model.u_dof_old_old[:] = self.model.u_dof_old[:]
        self.model.v_dof_old_old[:] = self.model.v_dof_old[:]
        if (self.model.nSpace_global == 3):
            self.model.w_dof_old_old[:] = self.model.w_dof_old[:]
        # solution at tn
        self.model.u_dof_old[:] = self.model.u[0].dof
        self.model.v_dof_old[:] = self.model.u[1].dof
        if (self.model.nSpace_global == 3):
            self.model.w_dof_old[:] = self.model.u[2].dof

        # COMPUTE MATERIAL PARAMETERS AND FORCE TERMS AS FUNCTIONS (if provided)
        if self.model.hasMaterialParametersAsFunctions:
            self.model.updateMaterialParameters()
        if self.model.forceTerms is not None:
            self.model.updateForceTerms()
        self.model.dt_last = self.model.timeIntegration.dt

    def postStep(self, t, firstStep=False):
        if firstStep == True:
            self.model.firstStep = False
        self.model.dt_last = self.model.timeIntegration.dt
        self.model.q['dV_last'][:] = self.model.q['dV']

        # Save uncorrected velocity
        self.model.q[('uncorrectedVelocity',0)][:] = self.model.q[('velocity',0)]
        self.model.ebqe[('uncorrectedVelocity',0)][:] = self.model.ebqe[('velocity',0)]

        # Correct drag
        if self.sedModel is not None:
            vos = self.model.vos_vel_nodes
            one_by_vos = (vos**2) / (vos**2 + np.maximum(1.0e-8,vos**2))
            for i in range(self.nd):
                self.sedModel.u[i].dof += -(one_by_vos*self.model.ncDrag[...,i] - self.sedModel.ncDrag[...,i])/self.sedModel.coefficients.rho_s
        logEvent("updating {0} particles...".format(self.nParticles))
        if self.use_ball_as_particle == 0:
            if self.particles is not None:
                self.particles.moveParticles(self.model.dt_last,
                                             t,
                                             self.particle_netForces,
                                             self.particle_netMoments)

                self.particles.updateSDF(self.mesh.nodeArray,
                                         self.model.q['x'],
                                         self.model.ebq_global['x'],
                                         self.phi_s,
                                         self.particle_signed_distances,
                                         self.particle_signed_distance_normals,
                                         self.particle_velocities,
                                         self.particle_centroids,
                                         self.ebq_global_phi_s,
                                         self.ebq_global_grad_phi_s,
                                         self.ebq_particle_velocity_s)

                # Temporary hack... see note in attachModels
                for i in range(self.particles.size()):
                    self.particle_centroids[i,:] = self.particles[i].x()
            else:
                self.phi_s[:] = 1e10
                self.phisField = np.ones(self.model.q[('u', 0)].shape, 'd') * 1e10
                for i in range(self.nParticles):
                    if self.granular_sdf_Calc is not None:
                        vel = lambda x: self.granular_vel_Calc(x, i)
                        sdf = lambda x: self.granular_sdf_Calc(x, i)
                    else:
                        vel = lambda x: self.particle_velocityList[i](t, x)
                        sdf = lambda x: self.particle_sdfList[i](t, x)

                    for j in range(self.mesh.nodeArray.shape[0]):
                        vel_at_node = vel(self.mesh.nodeArray[j, :])
                        sdf_at_node, sdNormals = sdf(self.mesh.nodeArray[j, :])
                        if (abs(sdf_at_node) < abs(self.phi_s[j])):
                            self.phi_s[j] = sdf_at_node
                    for eN in range(self.model.q['x'].shape[0]):
                        for k in range(self.model.q['x'].shape[1]):
                            self.particle_signed_distances[i, eN, k], self.particle_signed_distance_normals[i, eN, k] = sdf(self.model.q['x'][eN, k])
                            self.particle_velocities[i, eN, k] = vel(self.model.q['x'][eN, k])
                            if (abs(self.particle_signed_distances[i, eN, k]) < abs(self.phisField[eN, k])):
                                self.phisField[eN, k] = self.particle_signed_distances[i, eN, k]
                    corresponding_point_on_boundary = np.zeros((3,),'d')
                    for ebN in range(self.model.ebq_global['x'].shape[0]):
                        for kb in range(self.model.ebq_global['x'].shape[1]):
                            sdf_at_quad_pt,sdNormals = sdf(self.model.ebq_global['x'][ebN,kb])
                            if ( abs(sdf_at_quad_pt) < abs(self.ebq_global_phi_s[ebN,kb]) ):
                                self.ebq_global_phi_s[ebN,kb]=sdf_at_quad_pt
                                self.ebq_global_grad_phi_s[ebN,kb,:]=sdNormals
                                for j in range(len(sdNormals)):
                                    corresponding_point_on_boundary[j] = self.model.ebq_global['x'][ebN,kb][j] - sdf_at_quad_pt*sdNormals[j]
                                self.ebq_particle_velocity_s[ebN,kb,:]=vel(corresponding_point_on_boundary)
                self.model.q[('phis')] = self.phisField
                 #Update velocity inside the particle
                for ci_g_dof,ci_fg_dof in self.model.dirichletConditions[0].global2freeGlobal.items():
                    if isinstance(self.model.u[0].femSpace,C0_AffineLinearOnSimplexWithNodalBasis):
                        xyz = self.model.mesh.nodeArray[ci_g_dof,:]
                    elif isinstance(self.model.u[0].femSpace,C0_AffineQuadraticOnSimplexWithNodalBasis):
                        xyz = self.model.u[0].femSpace.dofMap.lagrangeNodesArray[ci_g_dof,:]
                    else:
                        assert False,"Use P1 or P2 for velocity"
                    distance_to_solid = 1e10
                    for i in range(self.nParticles):
                        if self.granular_sdf_Calc is not None:
                            vel = lambda x: self.granular_vel_Calc(x, i)
                            sdf = lambda x: self.granular_sdf_Calc(x, i)
                        else:
                            vel = lambda x: self.particle_velocityList[i](t, x)
                            sdf = lambda x: self.particle_sdfList[i](t, x)
                        distance_to_i_particle,_ = sdf(xyz)
                        if distance_to_solid > distance_to_i_particle:
                            vel_at_xyz = vel(xyz)
                            distance_to_solid = distance_to_i_particle
                    for ci in range(self.nc):#since nc=nd
                        dof = self.model.offset[ci] + self.model.stride[ci]*ci_fg_dof
                        if self.model.isActiveDOF[dof] < 0.5:
                            self.model.u[ci].dof[ci_g_dof] = vel_at_xyz[ci]
        if self.model.comm.isMaster():
            self.timeHistory.write("%21.16e\n" % (t,))
            self.timeHistory.flush()
            self.wettedAreaHistory.write("%21.16e\n" % (self.wettedAreas[-1],))
            self.wettedAreaHistory.flush()
            self.forceHistory_p.write("%21.16e %21.16e %21.16e\n" % tuple(self.netForces_p[-1, :]))
            self.forceHistory_p.flush()
            self.forceHistory_v.write("%21.16e %21.16e %21.16e\n" % tuple(self.netForces_v[-1, :]))
            self.forceHistory_v.flush()
            self.momentHistory.write("%21.15e %21.16e %21.16e\n" % tuple(self.netMoments[-1, :]))
            self.momentHistory.flush()
            if self.nParticles:
                self.particle_surfaceAreaHistory.write("%21.16e\n" % tuple(self.particle_surfaceArea[:]))
                self.particle_surfaceAreaHistory.flush()
                self.particle_forceHistory.write("%21.16e %21.16e %21.16e\n" % tuple(self.particle_netForces[0, :]))
                self.particle_forceHistory.flush()
                self.particle_pforceHistory.write("%21.16e %21.16e %21.16e\n" % tuple(self.particle_netForces[0+self.nParticles, :]))
                self.particle_pforceHistory.flush()
                self.particle_vforceHistory.write("%21.16e %21.16e %21.16e\n" % tuple(self.particle_netForces[0+2*self.nParticles, :]))
                self.particle_vforceHistory.flush()
                self.particle_momentHistory.write("%21.15e %21.16e %21.16e\n" % tuple(self.particle_netMoments[0, :]))
                self.particle_momentHistory.flush()

    def archiveFields(self, lm):
        """Archive the solid-phase signed distance alongside the solution.

        Previously written from ``NumericalSolution.py`` through
        ``archiveFiniteElementResiduals`` -- phi_s is not a residual -- inside
        a bare ``try/except: pass`` that also swallowed a ``NameError`` on
        every call. See :mod:`proteus.ArchiveFields`.

        ``phi_s`` is allocated in :meth:`initializeMesh`, which runs during
        model setup and so before any archiving. The guard is here only
        because this method must not raise for a model that somehow archives
        before its mesh is initialized; unlike the code it replaces, it says
        so in the log rather than passing silently.
        """
        from ..ArchiveFields import ArchiveField

        yield from super().archiveFields(lm)
        phi_s = getattr(self, "phi_s", None)
        if phi_s is None:
            logEvent("phi_s is not allocated yet; not archiving it "
                     "(initializeMesh should have run first)", level=1)
        else:
            #phi_s is sized mesh.nodeArray.shape[0] -- one value per
            #mesh vertex -- so it belongs on the base mesh grid. Routing
            #it through this model's solution space declared that space's
            #DOF count instead: with a C0P2 velocity that wrote 25 values
            #into a DataItem claiming 81, which a viewer reads as garbage.
            yield ArchiveField("phi_s", phi_s, on_mesh_nodes=True)


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
                 stressTraceBoundaryConditionsSetterDictDict=None,
                 stabilization=None,
                 shockCapturing=None,
                 conservativeFluxDict=None,
                 numericalFluxType=None,
                 TimeIntegrationClass=None,
                 massLumping=False,
                 reactionLumping=False,
                 options=None,
                 name='RANS3PF',
                 reuse_trial_and_test_quadrature=True,
                 sd=True,
                 movingDomain=False,
                 bdyNullSpace=False):
        self.bdyNullSpace = bdyNullSpace
        self.firstStep = True
        self.eb_adjoint_sigma = coefficients.eb_adjoint_sigma
        # this is a hack to test the effect of using a constant smoothing width
        useConstant_he = coefficients.useConstant_he
        self.postProcessing = True
        #
        # set the objects describing the method and boundary conditions
        #
        self.movingDomain = coefficients.movingDomain
        self.tLast_mesh = None
        # mql: Kill_pressure_term? This is for debugging and for convergence of momentum equations
        if ('KILL_PRESSURE_TERM') in dir(options):
            self.KILL_PRESSURE_TERM = options.KILL_PRESSURE_TERM
        else:
            self.KILL_PRESSURE_TERM = False
        # mql: Check if materialParameters are declared. This is for convergence tests
        self.hasMaterialParametersAsFunctions = False
        if ('materialParameters') in dir(options):
            self.materialParameters = options.materialParameters
            self.hasMaterialParametersAsFunctions = True
        # mql: Check if forceTerms are declared
        if ('forceTerms') in dir(options):
            self.forceTerms = options.forceTerms
        else:
            self.forceTerms = None
        # mql: Check if analytical pressure function is declared
        if ('analyticalPressureSolution') in dir(options):
            self.analyticalPressureSolution = options.analyticalPressureSolution
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
        if isinstance(
                self.u[0].femSpace,
                C0_AffineQuadraticOnSimplexWithNodalBasis):
            self.Hess = True
        self.ua = {}  # analytical solutions
        self.phi = phiDict
        self.dphi = {}
        self.phi_s = phiDict
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
        self.ncDrag = np.zeros((self.nFreeDOF_global[0],self.nc),'d')
        self.betaDrag = np.zeros((self.nFreeDOF_global[0],),'d')
        self.vos_vel_nodes = np.zeros((self.nFreeDOF_global[0],),'d')
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
        # mql. Some structures needed for stabilization operators
        self.u_dof_old = np.zeros(self.u[0].dof.shape, 'd')
        self.v_dof_old = np.zeros(self.u[0].dof.shape, 'd')
        self.w_dof_old = np.zeros(self.u[0].dof.shape, 'd')
        self.u_dof_old_old = np.zeros(self.u[0].dof.shape, 'd')
        self.v_dof_old_old = np.zeros(self.u[0].dof.shape, 'd')
        self.w_dof_old_old = np.zeros(self.u[0].dof.shape, 'd')
        self.uStar_dof = np.zeros(self.u[0].dof.shape, 'd')
        self.vStar_dof = np.zeros(self.u[0].dof.shape, 'd')
        self.wStar_dof = np.zeros(self.u[0].dof.shape, 'd')
        # mesh
        self.ebqe['x'] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary,
             3),
            'd')
        self.ebq_global[
            ('totalFlux',
             0)] = np.zeros(
            (self.mesh.nElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary),
            'd')
        self.ebq_global[
            ('velocityAverage',
             0)] = np.zeros(
            (self.mesh.nElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary,
             self.nSpace_global),
            'd')
        self.q['p'] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('u', 0)] = np.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('u', 1)] = np.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('u', 2)] = np.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('m', 0)] = self.q[('u', 0)]
        self.q[('m', 1)] = self.q[('u', 1)]
        self.q[('m', 2)] = self.q[('u', 2)]
        self.q[('m_last', 0)] = np.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('m_last', 1)] = np.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('m_last', 2)] = np.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('m_tmp', 0)] = np.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('m_tmp', 1)] = np.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('m_tmp', 2)] = np.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('mt', 0)] = np.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('mt', 1)] = np.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('mt', 2)] = np.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q['dV'] = np.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q['dV_last'] = -1000 * \
            np.ones((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('dV_u', 0)] = self.q[('dV')]
        self.q[('dV_u', 1)] = self.q[('dV')]
        self.q[('dV_u', 2)] = self.q[('dV')]
        self.q[('f', 0)] = np.zeros((self.mesh.nElements_global,
                                        self.nQuadraturePoints_element, self.nSpace_global), 'd')
        self.q[
            ('velocity',
             0)] = np.zeros(
                 (self.mesh.nElements_global,
                  self.nQuadraturePoints_element,
                  self.nSpace_global),
                 'd')
        self.q[
            ('uncorrectedVelocity',
             0)] = np.zeros(
                 (self.mesh.nElements_global,
                  self.nQuadraturePoints_element,
                  self.nSpace_global),
                 'd')
        self.q['velocity_solid'] = np.zeros(
            (self.mesh.nElements_global,
             self.nQuadraturePoints_element,
             self.nSpace_global),
            'd')
        # mql: create vectors to compute div(U)
        self.q['divU'] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        # mql: create vectors to compute uStar = 2*un-unm1
        self.q[
            ('velocityOld',
             0)] = np.zeros(
            (self.mesh.nElements_global,
             self.nQuadraturePoints_element,
             self.nSpace_global),
            'd')
        self.q[
            ('velocityStar',
             0)] = np.zeros(
                 (self.mesh.nElements_global,
                  self.nQuadraturePoints_element,
                  self.nSpace_global),
                 'd')
        self.q['phi_solid'] = np.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q['x'] = np.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element, 3), 'd')
        self.q[('cfl', 0)] = np.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('numDiff', 0, 0)] = np.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('numDiff', 1, 1)] = np.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('numDiff', 2, 2)] = np.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.ebqe[
            ('u',
             0)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary),
            'd')
        self.ebqe[
            ('u',
             1)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary),
            'd')
        self.ebqe[
            ('u',
             2)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary),
            'd')
        self.ebq[
            ('u',
             0)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary),
            'd')
        self.ebq[
            ('u',
             1)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary),
            'd')
        self.ebq[
            ('u',
             2)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary),
            'd')
        self.ebqe[
            ('advectiveFlux_bc_flag',
             0)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary),
            'i')
        self.ebqe[
            ('advectiveFlux_bc_flag',
             1)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary),
            'i')
        self.ebqe[
            ('advectiveFlux_bc_flag',
             2)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary),
            'i')
        self.ebqe[
            ('diffusiveFlux_bc_flag',
             0,
             0)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary),
            'i')
        self.ebqe[
            ('diffusiveFlux_bc_flag',
             1,
             1)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary),
            'i')
        self.ebqe[
            ('diffusiveFlux_bc_flag',
             2,
             2)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary),
            'i')
        self.ebqe[
            ('advectiveFlux_bc',
             0)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary),
            'd')
        self.ebqe[
            ('advectiveFlux_bc',
             1)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary),
            'd')
        self.ebqe[
            ('advectiveFlux_bc',
             2)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary),
            'd')
        self.ebqe[
            ('diffusiveFlux_bc',
             0,
             0)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary),
            'd')
        self.ebqe['penalty'] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary),
            'd')
        self.ebqe[
            ('diffusiveFlux_bc',
             1,
             1)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary),
            'd')
        self.ebqe[
            ('diffusiveFlux_bc',
             2,
             2)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary),
            'd')
        self.ebqe[
            ('velocity',
             0)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary,
             self.nSpace_global),
            'd')
        #self.ebqe[ #mql: I think these two are not needed. All the info is interleaved insided the "0"-th component
        #    ('velocity',
        #     1)] = np.zeros(
        #    (self.mesh.nExteriorElementBoundaries_global,
        #     self.nElementBoundaryQuadraturePoints_elementBoundary,
        #     self.nSpace_global),
        #    'd')
        #self.ebqe[
        #    ('velocity',
        #     2)] = np.zeros(
        #    (self.mesh.nExteriorElementBoundaries_global,
        #     self.nElementBoundaryQuadraturePoints_elementBoundary,
        #     self.nSpace_global),
        #    'd')
        self.ebqe[
            ('uncorrectedVelocity',
             0)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary,
             self.nSpace_global),
            'd')
        # VRANS start, defaults to RANS
        self.q[('r', 0)] = np.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q['eddy_viscosity'] = np.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        # VRANS end
        # RANS 2eq Models start
        self.q[
            ('grad(u)',
             0)] = np.zeros(
            (self.mesh.nElements_global,
             self.nQuadraturePoints_element,
             self.nSpace_global),
            'd')
        self.q[
            ('grad(u)',
             1)] = np.zeros(
            (self.mesh.nElements_global,
             self.nQuadraturePoints_element,
             self.nSpace_global),
            'd')
        self.q[
            ('grad(u)',
             2)] = np.zeros(
            (self.mesh.nElements_global,
             self.nQuadraturePoints_element,
             self.nSpace_global),
            'd')
        # probably don't need ebqe gradients
        self.ebqe[
            ('grad(u)',
             0)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary,
             self.nSpace_global),
            'd')
        self.ebqe[
            ('grad(u)',
             1)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary,
             self.nSpace_global),
            'd')
        self.ebqe[
            ('grad(u)',
             2)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary,
             self.nSpace_global),
            'd')
        self.ebq[
            ('grad(u)',
             0)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary,
             self.nSpace_global),
            'd')
        self.ebq[
            ('grad(u)',
             1)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary,
             self.nSpace_global),
            'd')
        self.ebq[
            ('grad(u)',
             2)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global,
             self.nElementBoundaryQuadraturePoints_elementBoundary,
             self.nSpace_global),
            'd')
        # RANS 2eq Models end
        self.points_elementBoundaryQuadrature = set()
        self.scalars_elementBoundaryQuadrature = set(
            [('u', ci) for ci in range(self.nc)])
        self.vectors_elementBoundaryQuadrature = set()
        self.tensors_elementBoundaryQuadrature = set()
        # use post processing tools to get conservative fluxes, None by default
        if self.postProcessing:
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
            self.q['abs(det(J))'] = np.zeros(
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
        #
        # show quadrature
        #
        log("Dumping quadrature shapes for model %s" % self.name, level=9)
        log("Element quadrature array (q)", level=9)
        for (k, v) in list(self.q.items()):
            log(str((k, v.shape)), level=9)
        log("Element boundary quadrature (ebq)", level=9)
        for (k, v) in list(self.ebq.items()):
            log(str((k, v.shape)), level=9)
        log("Global element boundary quadrature (ebq_global)", level=9)
        for (k, v) in list(self.ebq_global.items()):
            log(str((k, v.shape)), level=9)
        log("Exterior element boundary quadrature (ebqe)", level=9)
        for (k, v) in list(self.ebqe.items()):
            log(str((k, v.shape)), level=9)
        log("Interpolation points for nonlinear diffusion potential (phi_ip)", level=9)
        for (k, v) in list(self.phi_ip.items()):
            log(str((k, v.shape)), level=9)
        #
        # allocate residual and Jacobian storage
        #
        #
        # allocate residual and Jacobian storage
        #
        self.elementResidual = [np.zeros(
            (self.mesh.nElements_global,
             self.nDOF_test_element[ci]),
            'd')]
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

        # mql: material parameters defined by a function at quad points
        self.q['density'] = np.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q['dynamic_viscosity'] = np.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.ebqe['density'] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe['dynamic_viscosity'] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        # Initialize material parameters
        if self.hasMaterialParametersAsFunctions:
            self.updateMaterialParameters()
        # mql: force terms
        self.q[('force', 0)] = np.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('force', 1)] = np.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('force', 2)] = np.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')

        comm = Comm.get()
        self.comm = comm
        if comm.size() > 1:
            assert numericalFluxType is not None and numericalFluxType.useWeakDirichletConditions, "You must use a numerical flux to apply weak boundary conditions for parallel runs"

        log("initalizing numerical flux")
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
        # Aux quantity at DOFs to be filled by optimized code (MQL)
        self.quantDOFs = np.zeros(self.u[0].dof.shape, 'd')
        self.isBoundary_1D = None
        
        # set penalty terms
        log("initializing numerical flux penalty")
        self.numericalFlux.penalty_constant = self.coefficients.eb_penalty_constant
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
        log("setting up post-processing")
        from proteus import PostProcessingTools
        self.velocityPostProcessor = PostProcessingTools.VelocityPostProcessingChooser(
            self)
        log(memory("velocity postprocessor", "OneLevelTransport"), level=4)
        # helper for writing out data storage
        log("initializing archiver")
        from proteus import Archiver
        self.elementQuadratureDictionaryWriter = Archiver.XdmfWriter()
        self.elementBoundaryQuadratureDictionaryWriter = Archiver.XdmfWriter()
        self.exteriorElementBoundaryQuadratureDictionaryWriter = Archiver.XdmfWriter()
        log("flux bc objects")
        for ci, fbcObject in list(self.fluxBoundaryConditionsObjectsDict.items()):
            self.ebqe[('advectiveFlux_bc_flag', ci)] = np.zeros(
                self.ebqe[('advectiveFlux_bc', ci)].shape, 'i')
            for t, g in list(fbcObject.advectiveFluxBoundaryConditionsDict.items()):
                if ci in self.coefficients.advection:
                    self.ebqe[
                        ('advectiveFlux_bc', ci)][
                        t[0], t[1]] = g(
                        self.ebqe[
                            ('x')][
                            t[0], t[1]], self.timeIntegration.t)
                    self.ebqe[('advectiveFlux_bc_flag', ci)][t[0], t[1]] = 1
            for ck, diffusiveFluxBoundaryConditionsDict in list(fbcObject.diffusiveFluxBoundaryConditionsDictDict.items()):
                self.ebqe[('diffusiveFlux_bc_flag', ck, ci)] = np.zeros(
                    self.ebqe[('diffusiveFlux_bc', ck, ci)].shape, 'i')
                for t, g in list(diffusiveFluxBoundaryConditionsDict.items()):
                    self.ebqe[
                        ('diffusiveFlux_bc', ck, ci)][
                        t[0], t[1]] = g(
                        self.ebqe[
                            ('x')][
                            t[0], t[1]], self.timeIntegration.t)
                    self.ebqe[
                        ('diffusiveFlux_bc_flag', ck, ci)][
                        t[0], t[1]] = 1
        self.numericalFlux.setDirichletValues(self.ebqe)
        if self.movingDomain:
            self.MOVING_DOMAIN = 1.0
        else:
            self.MOVING_DOMAIN = 0.0
        if self.mesh.nodeVelocityArray is None:
            self.mesh.nodeVelocityArray = np.zeros(
                self.mesh.nodeArray.shape, 'd')
        # cek/ido todo replace python loops in modules with optimized code if
        # possible/necessary
        log("dirichlet conditions")
        self.forceStrongConditions = coefficients.forceStrongDirichlet
        self.dirichletConditionsForceDOF = {}
        if self.forceStrongConditions:
            for cj in range(self.nc):
                self.dirichletConditionsForceDOF[cj] = DOFBoundaryConditions(
                    self.u[cj].femSpace,
                    dofBoundaryConditionsSetterDict[cj],
                    weakDirichletConditions=False)
        log("final allocations")
        compKernelFlag = 0
        if self.coefficients.useConstant_he:
            self.elementDiameter = self.mesh.elementDiametersArray.copy()
            self.elementDiameter[:] = max(self.mesh.elementDiametersArray)
        else:
            self.elementDiameter = self.mesh.elementDiametersArray
        if self.nSpace_global == 2:
            import copy
            self.u[2] = self.u[1].copy()
            self.u[2].name = 'w'
            self.timeIntegration.m_tmp[
                2] = self.timeIntegration.m_tmp[1].copy()
            self.timeIntegration.beta_bdf[
                2] = self.timeIntegration.beta_bdf[1].copy()
            self.coefficients.sdInfo[(0, 2)] = (np.array(
                [0, 1, 2], dtype='i'), np.array([0, 1], dtype='i'))
            self.coefficients.sdInfo[(1, 2)] = (np.array(
                [0, 1, 2], dtype='i'), np.array([0, 1], dtype='i'))
            self.coefficients.sdInfo[(2, 0)] = (np.array(
                [0, 1, 2], dtype='i'), np.array([0, 1], dtype='i'))
            self.coefficients.sdInfo[(2, 0)] = (np.array(
                [0, 1, 2], dtype='i'), np.array([0, 1], dtype='i'))
            self.coefficients.sdInfo[(2, 1)] = (np.array(
                [0, 1, 2], dtype='i'), np.array([0, 1], dtype='i'))
            self.coefficients.sdInfo[(2, 2)] = (np.array(
                [0, 1, 2], dtype='i'), np.array([0, 1], dtype='i'))
            self.offset.append(self.offset[1])
            self.stride.append(self.stride[1])
            self.numericalFlux.isDOFBoundary[
                2] = self.numericalFlux.isDOFBoundary[1].copy()
            self.numericalFlux.ebqe[
                ('u', 2)] = self.numericalFlux.ebqe[
                ('u', 1)].copy()
            log("calling RANS3PF2D ctor")
            self.rans3pf = cRANS3PF2D.cppRANS3PF2D_base(
                self.nSpace_global,
                self.nQuadraturePoints_element,
                self.u[0].femSpace.elementMaps.localFunctionSpace.dim,
                self.u[0].femSpace.referenceFiniteElement.localFunctionSpace.dim,
                self.testSpace[0].referenceFiniteElement.localFunctionSpace.dim,
                self.nElementBoundaryQuadraturePoints_elementBoundary,
                compKernelFlag,
                self.coefficients.aDarcy,
                self.coefficients.betaForch,
                self.coefficients.grain,
                self.coefficients.packFraction,
                self.coefficients.packMargin,
                self.coefficients.maxFraction,
                self.coefficients.frFraction,
                self.coefficients.sigmaC,
                self.coefficients.C3e,
                self.coefficients.C4e,
                self.coefficients.eR,
                self.coefficients.fContact,
                self.coefficients.mContact,
                self.coefficients.nContact,
                self.coefficients.angFriction,
                self.coefficients.vos_limiter,
                self.coefficients.mu_fr_limiter,
                )
        else:
            log("calling  RANS3PF_base ctor")
            self.rans3pf = cRANS3PF.cppRANS3PF_base(
                self.nSpace_global,
                self.nQuadraturePoints_element,
                self.u[0].femSpace.elementMaps.localFunctionSpace.dim,
                self.u[0].femSpace.referenceFiniteElement.localFunctionSpace.dim,
                self.testSpace[0].referenceFiniteElement.localFunctionSpace.dim,
                self.nElementBoundaryQuadraturePoints_elementBoundary,
                compKernelFlag,
                self.coefficients.aDarcy,
                self.coefficients.betaForch,
                self.coefficients.grain,
                self.coefficients.packFraction,
                self.coefficients.packMargin,
                self.coefficients.maxFraction,
                self.coefficients.frFraction,
                self.coefficients.sigmaC,
                self.coefficients.C3e,
                self.coefficients.C4e,
                self.coefficients.eR,
                self.coefficients.fContact,
                self.coefficients.mContact,
                self.coefficients.nContact,
                self.coefficients.angFriction,
                self.coefficients.vos_limiter,
                self.coefficients.mu_fr_limiter,
                )

        self.phisErrorNodal = self.u[0].dof.copy()
        self.velocityErrorNodal = self.u[0].dof.copy()

        # Added by mql for discrete upwinding stab
        self.entropyResidualPerNode = np.zeros(self.u[0].dof.shape, 'd')
        self.laggedEntropyResidualPerNode = np.zeros(self.u[0].dof.shape, 'd')
        self.uStar_dMatrix = None
        self.vStar_dMatrix = None
        self.wStar_dMatrix = None
        self.numDOFs = None

    def updateMaterialParameters(self):
        x = self.q[('x')][:, :, 0]
        y = self.q[('x')][:, :, 1]
        z = self.q[('x')][:, :, 2]
        X = {0: x,
             1: y,
             2: z}
        t = self.timeIntegration.t
        self.q['density'][:] = self.materialParameters['density'](X, t)
        self.q['dynamic_viscosity'][:] = self.materialParameters['dynamic_viscosity'](X, t)
        # BOUNDARY
        ebqe_X = {0: self.ebqe['x'][:, :, 0],
                  1: self.ebqe['x'][:, :, 1],
                  2: self.ebqe['x'][:, :, 2]}
        self.ebqe['density'][:] = self.materialParameters['density'](ebqe_X, t)
        self.ebqe['dynamic_viscosity'][:] = self.materialParameters['dynamic_viscosity'](ebqe_X, t)

    def updateForceTerms(self):
        x = self.q[('x')][:, :, 0]
        y = self.q[('x')][:, :, 1]
        z = self.q[('x')][:, :, 2]
        X = {0: x,
             1: y,
             2: z}
        t = self.timeIntegration.t
        self.q[('force', 0)][:] = self.forceTerms[0](X, t)
        self.q[('force', 1)][:] = self.forceTerms[1](X, t)
        try:
            self.q[('force', 2)][:] = self.forceTerms[2](X, t)
        except:
            pass

    def getSparsityPatternForComponents(self):
        nSpace = self.nSpace_global
        self.nnz_1D = self.nnz // nSpace // nSpace
        self.numDOFs_1D = (len(self.rowptr) - 1) // nSpace
        self.rowptr_1D = np.zeros(self.numDOFs_1D + 1, 'i')  # NOTE: rowptr_1D[0]=0
        self.colind_1D = np.zeros(self.nnz_1D, 'i')
        # fill vector rowptr_scalar
        for i in range(1, self.rowptr_1D.size):
            self.rowptr_1D[i]=(self.rowptr_1D[i - 1] +
                               (self.rowptr[nSpace * (i - 1) + 1] -
                                self.rowptr[nSpace * (i - 1)])//nSpace)
        # fill vector colind_cMatrix
        ith_row = 0
        for i in range(len(self.rowptr)-1):  # 0 to total num of DOFs (i.e. num of rows of jacobian)
            if (i % nSpace == 0):  # Just consider the rows related to the u variable
                for j, offset in enumerate(range(self.rowptr[i], self.rowptr[i + 1])):
                    offset_1D = list(range(self.rowptr_1D[ith_row],
                                           self.rowptr_1D[ith_row + 1]))
                    if (j % nSpace == 0):
                        self.colind_1D[offset_1D[j//nSpace]] = self.colind[offset]//nSpace
                ith_row += 1

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
                        self.ebqe[
                            ('advectiveFlux_bc', ci)][
                            t[0], t[1]] = g(
                            self.ebqe[
                                ('x')][
                                t[0], t[1]], self.timeIntegration.t)
                        self.ebqe[
                            ('advectiveFlux_bc_flag', ci)][
                            t[0], t[1]] = 1
                for ck, diffusiveFluxBoundaryConditionsDict in list(fbcObject.diffusiveFluxBoundaryConditionsDictDict.items()):
                    for t, g in list(diffusiveFluxBoundaryConditionsDict.items()):
                        self.ebqe[
                            ('diffusiveFlux_bc', ck, ci)][
                            t[0], t[1]] = g(
                            self.ebqe[
                                ('x')][
                                t[0], t[1]], self.timeIntegration.t)
                        self.ebqe[
                            ('diffusiveFlux_bc_flag', ck, ci)][
                            t[0], t[1]] = 1
        r.fill(0.0)
        self.Ct_sge = 4.0
        self.Cd_sge = 36.0
        # TODO how to request problem specific evaluations from coefficient
        # class
        if 'evaluateForcingTerms' in dir(self.coefficients):
            self.coefficients.evaluateForcingTerms(
                self.timeIntegration.t,
                self.q,
                self.mesh,
                self.u[0].femSpace.elementMaps.psi,
                self.mesh.elementNodesArray)
        self.coefficients.wettedAreas[:] = 0.0
        self.coefficients.netForces_p[:, :] = 0.0
        self.coefficients.netForces_v[:, :] = 0.0
        self.coefficients.netMoments[:, :] = 0.0
        self.coefficients.particle_netForces[:, :] = 0.0
        self.coefficients.particle_netMoments[:, :] = 0.0
        self.coefficients.particle_surfaceArea[:] = 0.0

        if self.forceStrongConditions and self.firstStep == False:
            for cj in range(len(self.dirichletConditionsForceDOF)):
                for dofN, g in list(self.dirichletConditionsForceDOF[
                        cj].DOFBoundaryConditionsDict.items()):
                    self.u[cj].dof[dofN] = g(self.dirichletConditionsForceDOF[cj].DOFBoundaryPointDict[
                        dofN], self.timeIntegration.t)  # + self.MOVING_DOMAIN * self.mesh.nodeVelocityArray[dofN, cj - 1]

        if self.coefficients.set_vos:
            self.coefficients.set_vos(self.q['x'], self.coefficients.q_vos)
            self.coefficients.set_vos(self.ebqe['x'], self.coefficients.ebqe_vos)

        self.pressureModel.u[0].femSpace.elementMaps.getBasisValuesRef(self.elementQuadraturePoints)
        self.pressureModel.u[0].femSpace.elementMaps.getBasisGradientValuesRef(self.elementQuadraturePoints)
        self.pressureModel.u[0].femSpace.getBasisValuesRef(self.elementQuadraturePoints)
        self.pressureModel.u[0].femSpace.getBasisGradientValuesRef(self.elementQuadraturePoints)


        try:
            if self.coefficients.use_sbm > 0:
                self.isActiveDOF[:] = 0.0
            else:
                self.isActiveDOF[:] = 1.0
        except AttributeError:
            if self.coefficients.use_sbm > 0:
                self.isActiveDOF = np.zeros_like(r)
            else:
                self.isActiveDOF = np.ones_like(r)
        self.ncDrag[:]=0.0
        self.betaDrag[:]=0.0
        self.vos_vel_nodes[:]=0.0

        if self.uStar_dMatrix is None:
            self.getSparsityPatternForComponents()
            self.uStar_dMatrix = np.zeros(self.nnz_1D)
            self.vStar_dMatrix = np.zeros(self.nnz_1D)
            self.wStar_dMatrix = np.zeros(self.nnz_1D)
        #
        if self.isBoundary_1D is None:
            self.isBoundary_1D = np.zeros(self.u[0].dof.shape[0])
            argsDict_ = cArgumentsDict.ArgumentsDict()
            argsDict_["mesh_dof"] = self.mesh.nodeArray
            argsDict_["mesh_l2g"] = self.mesh.elementNodesArray
            argsDict_["mesh_trial_trace_ref"] = self.pressureModel.u[0].femSpace.elementMaps.psi_trace
            argsDict_["mesh_grad_trial_trace_ref"] = self.pressureModel.u[0].femSpace.elementMaps.grad_psi_trace
            argsDict_["dS_ref"] = self.elementBoundaryQuadratureWeights[('u', 0)]
            argsDict_["vel_test_trace_ref"] = self.u[0].femSpace.psi_trace
            argsDict_["normal_ref"] = self.u[0].femSpace.elementMaps.boundaryNormals
            argsDict_["boundaryJac_ref"] = self.u[0].femSpace.elementMaps.boundaryJacobians
            argsDict_["vel_l2g"] = self.u[0].femSpace.dofMap.l2g
            argsDict_["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
            argsDict_["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
            argsDict_["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
            argsDict_["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
            argsDict_["isBoundary_1D"] = self.isBoundary_1D
            self.rans3pf.getBoundaryDOFs(argsDict_)
            self.isBoundary_1D[:] = 1.0*(self.isBoundary_1D > 0)
            self.quantDOFs[:] = self.isBoundary_1D

        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["mesh_trial_ref"] = self.pressureModel.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.pressureModel.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["mesh_velocity_dof"] = self.mesh.nodeVelocityArray
        argsDict["MOVING_DOMAIN"] = self.MOVING_DOMAIN
        argsDict["PSTAB"] = self.coefficients.PSTAB
        argsDict["mesh_l2g"] = self.mesh.elementNodesArray
        argsDict["x_ref"] = self.elementQuadraturePoints
        argsDict["dV_ref"] = self.elementQuadratureWeights[('u', 0)]
        argsDict["nDOF_per_element_pressure"] = self.pressureModel.u[0].femSpace.referenceFiniteElement.localFunctionSpace.dim
        argsDict["p_trial_ref"] = self.pressureModel.u[0].femSpace.psi
        argsDict["p_grad_trial_ref"] = self.pressureModel.u[0].femSpace.grad_psi
        argsDict["p_test_ref"] = self.pressureModel.u[0].femSpace.psi
        argsDict["p_grad_test_ref"] = self.pressureModel.u[0].femSpace.grad_psi
        argsDict["q_p"] = self.pressureModel.q_p_sharp
        argsDict["q_grad_p"] = self.pressureModel.q_grad_p_sharp
        argsDict["ebqe_p"] = self.pressureModel.ebqe_p_sharp
        argsDict["ebqe_grad_p"] = self.pressureModel.ebqe_grad_p_sharp
        argsDict["vel_trial_ref"] = self.u[0].femSpace.psi
        argsDict["vel_grad_trial_ref"] = self.u[0].femSpace.grad_psi
        argsDict["vel_hess_trial_ref"] = self.u[0].femSpace.Hessian_psi
        argsDict["vel_test_ref"] = self.u[0].femSpace.psi
        argsDict["vel_grad_test_ref"] = self.u[0].femSpace.grad_psi
        argsDict["mesh_trial_trace_ref"] = self.pressureModel.u[0].femSpace.elementMaps.psi_trace
        argsDict["mesh_grad_trial_trace_ref"] = self.pressureModel.u[0].femSpace.elementMaps.grad_psi_trace
        argsDict["dS_ref"] = self.elementBoundaryQuadratureWeights[('u', 0)]
        argsDict["p_trial_trace_ref"] = self.pressureModel.u[0].femSpace.psi_trace
        argsDict["p_grad_trial_trace_ref"] = self.pressureModel.u[0].femSpace.grad_psi_trace
        argsDict["p_test_trace_ref"] = self.pressureModel.u[0].femSpace.psi_trace
        argsDict["p_grad_test_trace_ref"] = self.pressureModel.u[0].femSpace.grad_psi_trace
        argsDict["vel_trial_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["vel_grad_trial_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["vel_test_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["vel_grad_test_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["normal_ref"] = self.u[0].femSpace.elementMaps.boundaryNormals
        argsDict["boundaryJac_ref"] = self.u[0].femSpace.elementMaps.boundaryJacobians
        argsDict["eb_adjoint_sigma"] = self.eb_adjoint_sigma
        argsDict["elementDiameter"] = self.elementDiameter
        argsDict["nodeDiametersArray"] = self.mesh.nodeDiametersArray
        argsDict["hFactor"] = self.stabilization.hFactor
        argsDict["nElements_global"] = self.mesh.nElements_global
        argsDict["nElements_owned"] = int(self.mesh.nElements_owned)
        argsDict["nElementBoundaries_global"] = self.mesh.nElementBoundaries_global
        argsDict["nElementBoundaries_owned"] = int(self.mesh.nElementBoundaries_owned)
        argsDict["nNodes_owned"] = int(self.mesh.nNodes_owned)
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
        argsDict["C_dc"] = float(self.shockCapturing.shockCapturingFactor)
        argsDict["C_b"] = self.numericalFlux.penalty_constant
        argsDict["eps_solid"] = self.coefficients.epsFact_solid
        argsDict["ebq_global_phi_solid"] = self.coefficients.ebq_global_phi_s
        argsDict["ebq_global_grad_phi_solid"] = self.coefficients.ebq_global_grad_phi_s
        argsDict["ebq_particle_velocity_solid"] = self.coefficients.ebq_particle_velocity_s
        argsDict["phi_solid_nodes"] = self.coefficients.phi_s
        argsDict["phi_solid"] = self.coefficients.q_phi_solid
        argsDict["q_velocity_solid"] = self.coefficients.q_velocity_solid
        argsDict["q_velocityStar_solid"] = self.coefficients.q_velocityStar_solid
        argsDict["q_vos"] = self.coefficients.q_vos
        argsDict["q_dvos_dt"] = self.coefficients.q_dvos_dt
        argsDict["q_grad_vos"] = self.coefficients.q_grad_vos
        argsDict["q_dragAlpha"] = self.coefficients.q_dragAlpha
        argsDict["q_dragBeta"] = self.coefficients.q_dragBeta
        argsDict["q_mass_source"] = self.q[('r', 0)]
        argsDict["q_turb_var_0"] = self.coefficients.q_turb_var[0]
        argsDict["q_turb_var_1"] = self.coefficients.q_turb_var[1]
        argsDict["q_turb_var_grad_0"] = self.coefficients.q_turb_var_grad[0]
        argsDict["q_eddy_viscosity"] = self.q['eddy_viscosity']
        argsDict["p_l2g"] = self.pressureModel.u[0].femSpace.dofMap.l2g
        argsDict["vel_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["p_dof"] = self.pressureModel.p_sharp_dof
        argsDict["u_dof"] = self.u[0].dof
        argsDict["v_dof"] = self.u[1].dof
        argsDict["w_dof"] = self.u[2].dof
        argsDict["u_dof_old"] = self.u_dof_old
        argsDict["v_dof_old"] = self.v_dof_old
        argsDict["w_dof_old"] = self.w_dof_old
        argsDict["u_dof_old_old"] = self.u_dof_old_old
        argsDict["v_dof_old_old"] = self.v_dof_old_old
        argsDict["w_dof_old_old"] = self.w_dof_old_old
        argsDict["uStar_dof"] = self.uStar_dof
        argsDict["vStar_dof"] = self.vStar_dof
        argsDict["wStar_dof"] = self.wStar_dof
        argsDict["g"] = self.coefficients.g
        argsDict["useVF"] = self.coefficients.useVF
        argsDict["vf"] = self.coefficients.q_vf
        argsDict["phi"] = self.coefficients.q_phi
        argsDict["phi_dof"] = self.coefficients.phi_dof
        argsDict["normal_phi"] = self.coefficients.q_n
        argsDict["kappa_phi"] = self.coefficients.q_kappa
        argsDict["q_mom_u_acc"] = self.timeIntegration.m_tmp[0]
        argsDict["q_mom_v_acc"] = self.timeIntegration.m_tmp[1]
        argsDict["q_mom_w_acc"] = self.timeIntegration.m_tmp[2]
        argsDict["q_mass_adv"] = self.q[('f', 0)]
        argsDict["q_mom_u_acc_beta_bdf"] = self.timeIntegration.beta_bdf[0]
        argsDict["q_mom_v_acc_beta_bdf"] = self.timeIntegration.beta_bdf[1]
        argsDict["q_mom_w_acc_beta_bdf"] = self.timeIntegration.beta_bdf[2]
        argsDict["q_dV"] = self.q['dV']
        argsDict["q_dV_last"] = self.q['dV_last']
        argsDict["q_velocity_sge"] = self.q[('velocityStar', 0)]
        argsDict["ebqe_velocity_star"] = self.coefficients.ebqe_velocity_last
        argsDict["q_cfl"] = self.q[('cfl', 0)]
        argsDict["q_numDiff_u"] = self.q[('numDiff', 0, 0)]
        argsDict["q_numDiff_v"] = self.q[('numDiff', 1, 1)]
        argsDict["q_numDiff_w"] = self.q[('numDiff', 2, 2)]
        argsDict["q_numDiff_u_last"] = self.shockCapturing.numDiff_last[0]
        argsDict["q_numDiff_v_last"] = self.shockCapturing.numDiff_last[1]
        argsDict["q_numDiff_w_last"] = self.shockCapturing.numDiff_last[2]
        argsDict["sdInfo_u_u_rowptr"] = self.coefficients.sdInfo[(0, 0)][0]
        argsDict["sdInfo_u_u_colind"] = self.coefficients.sdInfo[(0, 0)][1]
        argsDict["sdInfo_u_v_rowptr"] = self.coefficients.sdInfo[(0, 1)][0]
        argsDict["sdInfo_u_v_colind"] = self.coefficients.sdInfo[(0, 1)][1]
        argsDict["sdInfo_u_w_rowptr"] = self.coefficients.sdInfo[(0, 2)][0]
        argsDict["sdInfo_u_w_colind"] = self.coefficients.sdInfo[(0, 2)][1]
        argsDict["sdInfo_v_v_rowptr"] = self.coefficients.sdInfo[(1, 1)][0]
        argsDict["sdInfo_v_v_colind"] = self.coefficients.sdInfo[(1, 1)][1]
        argsDict["sdInfo_v_u_rowptr"] = self.coefficients.sdInfo[(1, 0)][0]
        argsDict["sdInfo_v_u_colind"] = self.coefficients.sdInfo[(1, 0)][1]
        argsDict["sdInfo_v_w_rowptr"] = self.coefficients.sdInfo[(1, 2)][0]
        argsDict["sdInfo_v_w_colind"] = self.coefficients.sdInfo[(1, 2)][1]
        argsDict["sdInfo_w_w_rowptr"] = self.coefficients.sdInfo[(2, 2)][0]
        argsDict["sdInfo_w_w_colind"] = self.coefficients.sdInfo[(2, 2)][1]
        argsDict["sdInfo_w_u_rowptr"] = self.coefficients.sdInfo[(2, 0)][0]
        argsDict["sdInfo_w_u_colind"] = self.coefficients.sdInfo[(2, 0)][1]
        argsDict["sdInfo_w_v_rowptr"] = self.coefficients.sdInfo[(2, 1)][0]
        argsDict["sdInfo_w_v_colind"] = self.coefficients.sdInfo[(2, 1)][1]
        argsDict["offset_p"] = self.pressureModel.offset[0]
        argsDict["offset_u"] = self.offset[0]
        argsDict["offset_v"] = self.offset[1]
        argsDict["offset_w"] = self.offset[2]
        argsDict["stride_p"] = self.pressureModel.stride[0]
        argsDict["stride_u"] = self.stride[0]
        argsDict["stride_v"] = self.stride[1]
        argsDict["stride_w"] = self.stride[2]
        argsDict["globalResidual"] = r
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundariesArray"] = self.mesh.elementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["ebqe_vf_ext"] = self.coefficients.ebqe_vf
        argsDict["bc_ebqe_vf_ext"] = self.coefficients.bc_ebqe_vf
        argsDict["ebqe_phi_ext"] = self.coefficients.ebqe_phi
        argsDict["bc_ebqe_phi_ext"] = self.coefficients.bc_ebqe_phi
        argsDict["ebqe_normal_phi_ext"] = self.coefficients.ebqe_n
        argsDict["ebqe_kappa_phi_ext"] = self.coefficients.ebqe_kappa
        argsDict["ebqe_vos_ext"] = self.coefficients.ebqe_vos
        argsDict["ebqe_turb_var_0"] = self.coefficients.ebqe_turb_var[0]
        argsDict["ebqe_turb_var_1"] = self.coefficients.ebqe_turb_var[1]
        argsDict["isDOFBoundary_p"] = self.pressureModel.numericalFlux.isDOFBoundary[0]
        argsDict["isDOFBoundary_u"] = self.numericalFlux.isDOFBoundary[0]
        argsDict["isDOFBoundary_v"] = self.numericalFlux.isDOFBoundary[1]
        argsDict["isDOFBoundary_w"] = self.numericalFlux.isDOFBoundary[2]
        argsDict["isAdvectiveFluxBoundary_p"] = self.pressureModel.numericalFlux.ebqe[('advectiveFlux_bc_flag', 0)]
        argsDict["isAdvectiveFluxBoundary_u"] = self.ebqe[('advectiveFlux_bc_flag', 0)]
        argsDict["isAdvectiveFluxBoundary_v"] = self.ebqe[('advectiveFlux_bc_flag', 1)]
        argsDict["isAdvectiveFluxBoundary_w"] = self.ebqe[('advectiveFlux_bc_flag', 2)]
        argsDict["isDiffusiveFluxBoundary_u"] = self.ebqe[('diffusiveFlux_bc_flag', 0, 0)]
        argsDict["isDiffusiveFluxBoundary_v"] = self.ebqe[('diffusiveFlux_bc_flag', 1, 1)]
        argsDict["isDiffusiveFluxBoundary_w"] = self.ebqe[('diffusiveFlux_bc_flag', 2, 2)]
        argsDict["ebqe_bc_p_ext"] = self.pressureModel.numericalFlux.ebqe[('u', 0)]
        argsDict["ebqe_bc_flux_mass_ext"] = self.pressureModel.numericalFlux.ebqe[('advectiveFlux_bc', 0)]
        argsDict["ebqe_bc_flux_mom_u_adv_ext"] = self.ebqe[('advectiveFlux_bc', 0)]
        argsDict["ebqe_bc_flux_mom_v_adv_ext"] = self.ebqe[('advectiveFlux_bc', 1)]
        argsDict["ebqe_bc_flux_mom_w_adv_ext"] = self.ebqe[('advectiveFlux_bc', 2)]
        argsDict["ebqe_bc_u_ext"] = self.numericalFlux.ebqe[('u', 0)]
        argsDict["ebqe_bc_flux_u_diff_ext"] = self.ebqe[('diffusiveFlux_bc', 0, 0)]
        argsDict["ebqe_penalty_ext"] = self.ebqe['penalty']
        argsDict["ebqe_bc_v_ext"] = self.numericalFlux.ebqe[('u', 1)]
        argsDict["ebqe_bc_flux_v_diff_ext"] = self.ebqe[('diffusiveFlux_bc', 1, 1)]
        argsDict["ebqe_bc_w_ext"] = self.numericalFlux.ebqe[('u', 2)]
        argsDict["ebqe_bc_flux_w_diff_ext"] = self.ebqe[('diffusiveFlux_bc', 2, 2)]
        argsDict["q_x"] = self.q['x']
        argsDict["q_velocity"] = self.q[('velocity', 0)]
        argsDict["ebqe_velocity"] = self.ebqe[('velocity', 0)]
        argsDict["q_grad_u"] = self.q[('grad(u)', 0)]
        argsDict["q_grad_v"] = self.q[('grad(u)', 1)]
        argsDict["q_grad_w"] = self.q[('grad(u)', 2)]
        argsDict["q_divU"] = self.q['divU']
        argsDict["ebqe_grad_u"] = self.ebqe[('grad(u)', 0)]
        argsDict["ebqe_grad_v"] = self.ebqe[('grad(u)', 1)]
        argsDict["ebqe_grad_w"] = self.ebqe[('grad(u)', 2)]
        argsDict["flux"] = self.ebq_global[('totalFlux', 0)]
        argsDict["elementResidual_p_save"] = self.elementResidual[0]
        argsDict["elementFlags"] = self.mesh.elementMaterialTypes
        argsDict["boundaryFlags"] = self.mesh.elementBoundaryUserFlags
        argsDict["barycenters"] = self.coefficients.barycenters
        argsDict["wettedAreas"] = self.coefficients.wettedAreas
        argsDict["netForces_p"] = self.coefficients.netForces_p
        argsDict["netForces_v"] = self.coefficients.netForces_v
        argsDict["netMoments"] = self.coefficients.netMoments
        argsDict["q_rho"] = self.coefficients.q_rho
        argsDict["ebqe_rho"] = self.coefficients.ebqe_rho
        argsDict["q_nu"] = self.coefficients.q_nu
        argsDict["ebqe_nu"] = self.coefficients.ebqe_nu
        argsDict["nParticles"] = self.coefficients.nParticles
        argsDict["particle_epsFact"] = self.coefficients.particle_epsFact
        argsDict["particle_alpha"] = self.coefficients.particle_alpha
        argsDict["particle_beta"] = self.coefficients.particle_beta
        argsDict["particle_penalty_constant"] = self.coefficients.particle_penalty_constant
        argsDict["particle_signed_distances"] = self.coefficients.particle_signed_distances
        argsDict["particle_signed_distance_normals"] = self.coefficients.particle_signed_distance_normals
        argsDict["particle_velocities"] = self.coefficients.particle_velocities
        argsDict["particle_centroids"] = self.coefficients.particle_centroids
        argsDict["particle_netForces"] = self.coefficients.particle_netForces
        argsDict["particle_netMoments"] = self.coefficients.particle_netMoments
        argsDict["particle_surfaceArea"] = self.coefficients.particle_surfaceArea
        argsDict["particle_nitsche"] = self.coefficients.particle_nitsche
        argsDict["use_ball_as_particle"] = self.coefficients.use_ball_as_particle
        argsDict["ball_center"] = self.coefficients.ball_center
        argsDict["ball_radius"] = self.coefficients.ball_radius
        argsDict["ball_velocity"] = self.coefficients.ball_velocity
        argsDict["ball_angular_velocity"] = self.coefficients.ball_angular_velocity
        argsDict["phisError"] = self.q['phisError']
        argsDict["phisErrorNodal"] = self.phisErrorNodal
        argsDict["USE_SUPG"] = int(self.coefficients.USE_SUPG)
        argsDict["ARTIFICIAL_VISCOSITY"] = self.coefficients.ARTIFICIAL_VISCOSITY
        argsDict["cMax"] = self.coefficients.cMax
        argsDict["cE"] = float(self.coefficients.cE)
        argsDict["MULTIPLY_EXTERNAL_FORCE_BY_DENSITY"] = self.coefficients.MULTIPLY_EXTERNAL_FORCE_BY_DENSITY
        argsDict["forcex"] = self.q[('force', 0)]
        argsDict["forcey"] = self.q[('force', 1)]
        argsDict["forcez"] = self.q[('force', 2)]
        argsDict["KILL_PRESSURE_TERM"] = int(self.KILL_PRESSURE_TERM)
        argsDict["dt"] = self.timeIntegration.dt
        argsDict["quantDOFs"] = self.quantDOFs
        argsDict["MATERIAL_PARAMETERS_AS_FUNCTION"] = int(self.hasMaterialParametersAsFunctions)
        argsDict["density_as_function"] = self.q['density']
        argsDict["dynamic_viscosity_as_function"] = self.q['dynamic_viscosity']
        argsDict["ebqe_density_as_function"] = self.ebqe['density']
        argsDict["ebqe_dynamic_viscosity_as_function"] = self.ebqe['dynamic_viscosity']
        argsDict["order_polynomial"] = float(self.u[0].femSpace.order)
        argsDict["isActiveDOF"] = self.isActiveDOF
        argsDict["USE_SBM"] = self.coefficients.use_sbm
        argsDict["ncDrag"] = self.ncDrag
        argsDict["betaDrag"] = self.betaDrag
        argsDict["vos_vel_nodes"] = self.vos_vel_nodes
        argsDict["entropyResidualPerNode"] = self.entropyResidualPerNode
        argsDict["laggedEntropyResidualPerNode"] = self.laggedEntropyResidualPerNode
        argsDict["uStar_dMatrix"] = self.uStar_dMatrix
        argsDict["vStar_dMatrix"] = self.vStar_dMatrix
        argsDict["wStar_dMatrix"] = self.wStar_dMatrix
        argsDict["numDOFs_1D"] = self.numDOFs_1D
        argsDict["NNZ_1D"] = self.nnz_1D
        argsDict["csrRowIndeces_1D"] = self.csrRowIndeces[(0, 0)] // self.nSpace_global // self.nSpace_global
        argsDict["csrColumnOffsets_1D"] = self.csrColumnOffsets[(0, 0)]//self.nSpace_global
        argsDict["rowptr_1D"] = self.rowptr_1D
        argsDict["colind_1D"] = self.colind_1D
        argsDict["isBoundary_1D"] = self.isBoundary_1D
        argsDict["INT_BY_PARTS_PRESSURE"] = self.coefficients.INT_BY_PARTS_PRESSURE

        self.rans3pf.calculateResidual(
            argsDict,
            self.coefficients.useExact)

        r*=self.isActiveDOF
#         print "***********",np.amin(r),np.amax(r),np.amin(self.isActiveDOF),np.amax(self.isActiveDOF)
        # mql: Save the solution in 'u' to allow SimTools.py to compute the errors
        for dim in range(self.nSpace_global):
            self.q[('u', dim)][:] = self.q[('velocity', 0)][:, :, dim]

        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        
        comm.Allreduce(self.coefficients.wettedAreas.copy(),self.coefficients.wettedAreas)
        comm.Allreduce(self.coefficients.netForces_p.copy(),self.coefficients.netForces_p)
        comm.Allreduce(self.coefficients.netForces_v.copy(),self.coefficients.netForces_v)
        comm.Allreduce(self.coefficients.netMoments.copy(),self.coefficients.netMoments)
        
        comm.Allreduce(self.coefficients.particle_netForces.copy(),self.coefficients.particle_netForces)
        comm.Allreduce(self.coefficients.particle_netMoments.copy(),self.coefficients.particle_netMoments)
        comm.Allreduce(self.coefficients.particle_surfaceArea.copy(),self.coefficients.particle_surfaceArea)
        
        for i in range(self.coefficients.nParticles):
            logEvent("particle i=" + repr(i)+ " force " + repr(self.coefficients.particle_netForces[i]))
            logEvent("particle i=" + repr(i)+ " moment " + repr(self.coefficients.particle_netMoments[i]))
            logEvent("particle i=" + repr(i)+ " surfaceArea " + repr(self.coefficients.particle_surfaceArea[i]))
            logEvent("particle i=" + repr(i)+ " stress force " + repr(self.coefficients.particle_netForces[i+self.coefficients.nParticles]))
            logEvent("particle i=" + repr(i)+ " pressure force " + repr(self.coefficients.particle_netForces[i+2*self.coefficients.nParticles]))

        if self.forceStrongConditions:
            for cj in range(len(self.dirichletConditionsForceDOF)):
                for dofN, g in list(self.dirichletConditionsForceDOF[cj].DOFBoundaryConditionsDict.items()):
                    r[self.offset[cj] + self.stride[cj] * dofN] = self.u[cj].dof[dofN] - g(self.dirichletConditionsForceDOF[cj].DOFBoundaryPointDict[
                        dofN], self.timeIntegration.t)  # - self.MOVING_DOMAIN * self.mesh.nodeVelocityArray[dofN, cj - 1]

        cflMax = globalMax(self.q[('cfl', 0)].max()) * self.timeIntegration.dt
        log("Maximum CFL = " + str(cflMax), level=2)
        if self.stabilization:
            self.stabilization.accumulateSubgridMassHistory(self.q)
        log("Global residual", level=9, data=r)
        # mwf decide if this is reasonable for keeping solver statistics
        self.nonlinear_function_evaluations += 1

    def getJacobian(self, jacobian):
        cfemIntegrals.zeroJacobian_CSR(self.nNonzerosInJacobian,
                                       jacobian)

        if self.nSpace_global == 2:
            self.csrRowIndeces[(0, 2)] = self.csrRowIndeces[(0, 1)]
            self.csrColumnOffsets[(0, 2)] = self.csrColumnOffsets[(0, 1)]
            self.csrRowIndeces[(0, 2)] = self.csrRowIndeces[(0, 1)]
            self.csrColumnOffsets[(0, 2)] = self.csrColumnOffsets[(0, 1)]
            self.csrRowIndeces[(1, 2)] = self.csrRowIndeces[(0, 1)]
            self.csrColumnOffsets[(1, 2)] = self.csrColumnOffsets[(0, 1)]
            self.csrRowIndeces[(2, 0)] = self.csrRowIndeces[(1, 0)]
            self.csrColumnOffsets[(2, 0)] = self.csrColumnOffsets[(1, 0)]
            self.csrRowIndeces[(2, 0)] = self.csrRowIndeces[(1, 0)]
            self.csrColumnOffsets[(2, 0)] = self.csrColumnOffsets[(1, 0)]
            self.csrRowIndeces[(2, 1)] = self.csrRowIndeces[(1, 0)]
            self.csrColumnOffsets[(2, 1)] = self.csrColumnOffsets[(1, 0)]
            self.csrRowIndeces[(2, 2)] = self.csrRowIndeces[(1, 0)]
            self.csrColumnOffsets[(2, 2)] = self.csrColumnOffsets[(1, 0)]
            self.csrColumnOffsets_eb[(0, 2)] = self.csrColumnOffsets[(0, 1)]
            self.csrColumnOffsets_eb[(0, 2)] = self.csrColumnOffsets[(0, 1)]
            self.csrColumnOffsets_eb[(1, 2)] = self.csrColumnOffsets[(0, 1)]
            self.csrColumnOffsets_eb[(2, 0)] = self.csrColumnOffsets[(0, 1)]
            self.csrColumnOffsets_eb[(2, 0)] = self.csrColumnOffsets[(0, 1)]
            self.csrColumnOffsets_eb[(2, 1)] = self.csrColumnOffsets[(0, 1)]
            self.csrColumnOffsets_eb[(2, 2)] = self.csrColumnOffsets[(0, 1)]

        (rowptr, colind, globalJacobian) = jacobian.getCSRrepresentation()
        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["mesh_trial_ref"] = self.pressureModel.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.pressureModel.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["mesh_velocity_dof"] = self.mesh.nodeVelocityArray
        argsDict["MOVING_DOMAIN"] = self.MOVING_DOMAIN
        argsDict["PSTAB"] = self.coefficients.PSTAB
        argsDict["mesh_l2g"] = self.mesh.elementNodesArray
        argsDict["x_ref"] = self.elementQuadraturePoints
        argsDict["dV_ref"] = self.elementQuadratureWeights[('u', 0)]
        argsDict["p_trial_ref"] = self.pressureModel.u[0].femSpace.psi
        argsDict["p_grad_trial_ref"] = self.pressureModel.u[0].femSpace.grad_psi
        argsDict["p_test_ref"] = self.pressureModel.u[0].femSpace.psi
        argsDict["p_grad_test_ref"] = self.pressureModel.u[0].femSpace.grad_psi
        argsDict["q_p"] = self.pressureModel.q_p_sharp
        argsDict["q_grad_p"] = self.pressureModel.q_grad_p_sharp
        argsDict["ebqe_p"] = self.pressureModel.ebqe_p_sharp
        argsDict["ebqe_grad_p"] = self.pressureModel.ebqe_grad_p_sharp
        argsDict["vel_trial_ref"] = self.u[0].femSpace.psi
        argsDict["vel_grad_trial_ref"] = self.u[0].femSpace.grad_psi
        argsDict["vel_hess_trial_ref"] = self.u[0].femSpace.Hessian_psi
        argsDict["vel_test_ref"] = self.u[0].femSpace.psi
        argsDict["vel_grad_test_ref"] = self.u[0].femSpace.grad_psi
        argsDict["mesh_trial_trace_ref"] = self.pressureModel.u[0].femSpace.elementMaps.psi_trace
        argsDict["mesh_grad_trial_trace_ref"] = self.pressureModel.u[0].femSpace.elementMaps.grad_psi_trace
        argsDict["dS_ref"] = self.pressureModel.elementBoundaryQuadratureWeights[('u', 0)]
        argsDict["p_trial_trace_ref"] = self.pressureModel.u[0].femSpace.psi_trace
        argsDict["p_grad_trial_trace_ref"] = self.pressureModel.u[0].femSpace.grad_psi_trace
        argsDict["p_test_trace_ref"] = self.pressureModel.u[0].femSpace.psi_trace
        argsDict["p_grad_test_trace_ref"] = self.pressureModel.u[0].femSpace.grad_psi_trace
        argsDict["vel_trial_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["vel_grad_trial_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["vel_test_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["vel_grad_test_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["normal_ref"] = self.u[0].femSpace.elementMaps.boundaryNormals
        argsDict["boundaryJac_ref"] = self.u[0].femSpace.elementMaps.boundaryJacobians
        argsDict["eb_adjoint_sigma"] = self.eb_adjoint_sigma
        argsDict["elementDiameter"] = self.elementDiameter
        argsDict["nodeDiametersArray"] = self.mesh.nodeDiametersArray
        argsDict["hFactor"] = self.stabilization.hFactor
        argsDict["nElements_global"] = self.mesh.nElements_global
        argsDict["nElements_owned"] = int(self.mesh.nElements_owned)
        argsDict["nElementBoundaries_global"] = self.mesh.nElementBoundaries_global
        argsDict["nElementBoundaries_owned"] = int(self.mesh.nElementBoundaries_owned)
        argsDict["nNodes_owned"] = int(self.mesh.nNodes_owned)
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
        argsDict["C_dg"] = float(self.shockCapturing.shockCapturingFactor)
        argsDict["C_b"] = self.numericalFlux.penalty_constant
        argsDict["eps_solid"] = self.coefficients.epsFact_solid
        argsDict["ebq_global_phi_solid"] = self.coefficients.ebq_global_phi_s
        argsDict["ebq_global_grad_phi_solid"] = self.coefficients.ebq_global_grad_phi_s
        argsDict["ebq_particle_velocity_solid"] = self.coefficients.ebq_particle_velocity_s
        argsDict["phi_solid_nodes"] = self.coefficients.phi_s
        argsDict["phi_solid"] = self.coefficients.q_phi_solid
        argsDict["q_velocity_solid"] = self.coefficients.q_velocity_solid
        argsDict["q_velocityStar_solid"] = self.coefficients.q_velocityStar_solid
        argsDict["q_vos"] = self.coefficients.q_vos
        argsDict["q_dvos_dt"] = self.coefficients.q_dvos_dt
        argsDict["q_grad_vos"] = self.coefficients.q_grad_vos
        argsDict["q_dragAlpha"] = self.coefficients.q_dragAlpha
        argsDict["q_dragBeta"] = self.coefficients.q_dragBeta
        argsDict["q_mass_source"] = self.pressureModel.q[('r', 0)]
        argsDict["q_turb_var_0"] = self.coefficients.q_turb_var[0]
        argsDict["q_turb_var_1"] = self.coefficients.q_turb_var[1]
        argsDict["q_turb_var_grad_0"] = self.coefficients.q_turb_var_grad[0]
        argsDict["p_l2g"] = self.pressureModel.u[0].femSpace.dofMap.l2g
        argsDict["vel_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["p_dof"] = self.pressureModel.u[0].dof
        argsDict["u_dof"] = self.u[0].dof
        argsDict["v_dof"] = self.u[1].dof
        argsDict["w_dof"] = self.u[2].dof
        argsDict["g"] = self.coefficients.g
        argsDict["useVF"] = self.coefficients.useVF
        argsDict["vf"] = self.coefficients.q_vf
        argsDict["phi"] = self.coefficients.q_phi
        argsDict["phi_dof"] = self.coefficients.phi_dof
        argsDict["normal_phi"] = self.coefficients.q_n
        argsDict["kappa_phi"] = self.coefficients.q_kappa
        argsDict["q_mom_u_acc_beta_bdf"] = self.timeIntegration.beta_bdf[0]
        argsDict["q_mom_v_acc_beta_bdf"] = self.timeIntegration.beta_bdf[1]
        argsDict["q_mom_w_acc_beta_bdf"] = self.timeIntegration.beta_bdf[2]
        argsDict["q_dV"] = self.q['dV']
        argsDict["q_dV_last"] = self.q['dV_last']
        argsDict["q_velocity_sge"] = self.q[('velocityStar', 0)]
        argsDict["ebqe_velocity_star"] = self.coefficients.ebqe_velocity_last
        argsDict["q_cfl"] = self.q[('cfl', 0)]
        argsDict["q_numDiff_u_last"] = self.shockCapturing.numDiff_last[0]
        argsDict["q_numDiff_v_last"] = self.shockCapturing.numDiff_last[1]
        argsDict["q_numDiff_w_last"] = self.shockCapturing.numDiff_last[2]
        argsDict["sdInfo_u_u_rowptr"] = self.coefficients.sdInfo[(0, 0)][0]
        argsDict["sdInfo_u_u_colind"] = self.coefficients.sdInfo[(0, 0)][1]
        argsDict["sdInfo_u_v_rowptr"] = self.coefficients.sdInfo[(0, 1)][0]
        argsDict["sdInfo_u_v_colind"] = self.coefficients.sdInfo[(0, 1)][1]
        argsDict["sdInfo_u_w_rowptr"] = self.coefficients.sdInfo[(0, 2)][0]
        argsDict["sdInfo_u_w_colind"] = self.coefficients.sdInfo[(0, 2)][1]
        argsDict["sdInfo_v_v_rowptr"] = self.coefficients.sdInfo[(1, 1)][0]
        argsDict["sdInfo_v_v_colind"] = self.coefficients.sdInfo[(1, 1)][1]
        argsDict["sdInfo_v_u_rowptr"] = self.coefficients.sdInfo[(1, 0)][0]
        argsDict["sdInfo_v_u_colind"] = self.coefficients.sdInfo[(1, 0)][1]
        argsDict["sdInfo_v_w_rowptr"] = self.coefficients.sdInfo[(1, 2)][0]
        argsDict["sdInfo_v_w_colind"] = self.coefficients.sdInfo[(1, 2)][1]
        argsDict["sdInfo_w_w_rowptr"] = self.coefficients.sdInfo[(2, 2)][0]
        argsDict["sdInfo_w_w_colind"] = self.coefficients.sdInfo[(2, 2)][1]
        argsDict["sdInfo_w_u_rowptr"] = self.coefficients.sdInfo[(2, 0)][0]
        argsDict["sdInfo_w_u_colind"] = self.coefficients.sdInfo[(2, 0)][1]
        argsDict["sdInfo_w_v_rowptr"] = self.coefficients.sdInfo[(2, 1)][0]
        argsDict["sdInfo_w_v_colind"] = self.coefficients.sdInfo[(2, 1)][1]
        argsDict["csrRowIndeces_p_p"] = self.csrRowIndeces[(0, 0)]
        argsDict["csrColumnOffsets_p_p"] = self.csrColumnOffsets[(0, 0)]
        argsDict["csrRowIndeces_p_u"] = self.csrRowIndeces[(0, 0)]
        argsDict["csrColumnOffsets_p_u"] = self.csrColumnOffsets[(0, 0)]
        argsDict["csrRowIndeces_p_v"] = self.csrRowIndeces[(0, 1)]
        argsDict["csrColumnOffsets_p_v"] = self.csrColumnOffsets[(0, 1)]
        argsDict["csrRowIndeces_p_w"] = self.csrRowIndeces[(0, 2)]
        argsDict["csrColumnOffsets_p_w"] = self.csrColumnOffsets[(0, 2)]
        argsDict["csrRowIndeces_u_p"] = self.csrRowIndeces[(0, 0)]
        argsDict["csrColumnOffsets_u_p"] = self.csrColumnOffsets[(0, 0)]
        argsDict["csrRowIndeces_u_u"] = self.csrRowIndeces[(0, 0)]
        argsDict["csrColumnOffsets_u_u"] = self.csrColumnOffsets[(0, 0)]
        argsDict["csrRowIndeces_u_v"] = self.csrRowIndeces[(0, 1)]
        argsDict["csrColumnOffsets_u_v"] = self.csrColumnOffsets[(0, 1)]
        argsDict["csrRowIndeces_u_w"] = self.csrRowIndeces[(0, 2)]
        argsDict["csrColumnOffsets_u_w"] = self.csrColumnOffsets[(0, 2)]
        argsDict["csrRowIndeces_v_p"] = self.csrRowIndeces[(1, 0)]
        argsDict["csrColumnOffsets_v_p"] = self.csrColumnOffsets[(1, 0)]
        argsDict["csrRowIndeces_v_u"] = self.csrRowIndeces[(1, 0)]
        argsDict["csrColumnOffsets_v_u"] = self.csrColumnOffsets[(1, 0)]
        argsDict["csrRowIndeces_v_v"] = self.csrRowIndeces[(1, 1)]
        argsDict["csrColumnOffsets_v_v"] = self.csrColumnOffsets[(1, 1)]
        argsDict["csrRowIndeces_v_w"] = self.csrRowIndeces[(1, 2)]
        argsDict["csrColumnOffsets_v_w"] = self.csrColumnOffsets[(1, 2)]
        argsDict["csrRowIndeces_w_p"] = self.csrRowIndeces[(2, 0)]
        argsDict["csrColumnOffsets_w_p"] = self.csrColumnOffsets[(2, 0)]
        argsDict["csrRowIndeces_w_u"] = self.csrRowIndeces[(2, 0)]
        argsDict["csrColumnOffsets_w_u"] = self.csrColumnOffsets[(2, 0)]
        argsDict["csrRowIndeces_w_v"] = self.csrRowIndeces[(2, 1)]
        argsDict["csrColumnOffsets_w_v"] = self.csrColumnOffsets[(2, 1)]
        argsDict["csrRowIndeces_w_w"] = self.csrRowIndeces[(2, 2)]
        argsDict["csrColumnOffsets_w_w"] = self.csrColumnOffsets[(2, 2)]
        argsDict["globalJacobian"] = globalJacobian
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundariesArray"] = self.mesh.elementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["ebqe_vf_ext"] = self.coefficients.ebqe_vf
        argsDict["bc_ebqe_vf_ext"] = self.coefficients.bc_ebqe_vf
        argsDict["ebqe_phi_ext"] = self.coefficients.ebqe_phi
        argsDict["bc_ebqe_phi_ext"] = self.coefficients.bc_ebqe_phi
        argsDict["ebqe_normal_phi_ext"] = self.coefficients.ebqe_n
        argsDict["ebqe_kappa_phi_ext"] = self.coefficients.ebqe_kappa
        argsDict["ebqe_vos_ext"] = self.coefficients.ebqe_vos
        argsDict["ebqe_turb_var_0"] = self.coefficients.ebqe_turb_var[0]
        argsDict["ebqe_turb_var_1"] = self.coefficients.ebqe_turb_var[1]
        argsDict["isDOFBoundary_p"] = self.pressureModel.numericalFlux.isDOFBoundary[0]
        argsDict["isDOFBoundary_u"] = self.numericalFlux.isDOFBoundary[0]
        argsDict["isDOFBoundary_v"] = self.numericalFlux.isDOFBoundary[1]
        argsDict["isDOFBoundary_w"] = self.numericalFlux.isDOFBoundary[2]
        argsDict["isAdvectiveFluxBoundary_p"] = self.pressureModel.numericalFlux.ebqe[('advectiveFlux_bc_flag', 0)]
        argsDict["isAdvectiveFluxBoundary_u"] = self.ebqe[('advectiveFlux_bc_flag', 0)]
        argsDict["isAdvectiveFluxBoundary_v"] = self.ebqe[('advectiveFlux_bc_flag', 1)]
        argsDict["isAdvectiveFluxBoundary_w"] = self.ebqe[('advectiveFlux_bc_flag', 2)]
        argsDict["isDiffusiveFluxBoundary_u"] = self.ebqe[('diffusiveFlux_bc_flag', 0, 0)]
        argsDict["isDiffusiveFluxBoundary_v"] = self.ebqe[('diffusiveFlux_bc_flag', 1, 1)]
        argsDict["isDiffusiveFluxBoundary_w"] = self.ebqe[('diffusiveFlux_bc_flag', 2, 2)]
        argsDict["ebqe_bc_p_ext"] = self.pressureModel.numericalFlux.ebqe[('u', 0)]
        argsDict["ebqe_bc_flux_mass_ext"] = self.pressureModel.numericalFlux.ebqe[('advectiveFlux_bc', 0)]
        argsDict["ebqe_bc_flux_mom_u_adv_ext"] = self.ebqe[('advectiveFlux_bc', 0)]
        argsDict["ebqe_bc_flux_mom_v_adv_ext"] = self.ebqe[('advectiveFlux_bc', 1)]
        argsDict["ebqe_bc_flux_mom_w_adv_ext"] = self.ebqe[('advectiveFlux_bc', 2)]
        argsDict["ebqe_bc_u_ext"] = self.numericalFlux.ebqe[('u', 0)]
        argsDict["ebqe_bc_flux_u_diff_ext"] = self.ebqe[('diffusiveFlux_bc', 0, 0)]
        argsDict["ebqe_penalty_ext"] = self.ebqe['penalty']
        argsDict["ebqe_bc_v_ext"] = self.numericalFlux.ebqe[('u', 1)]
        argsDict["ebqe_bc_flux_v_diff_ext"] = self.ebqe[('diffusiveFlux_bc', 1, 1)]
        argsDict["ebqe_bc_w_ext"] = self.numericalFlux.ebqe[('u', 2)]
        argsDict["ebqe_bc_flux_w_diff_ext"] = self.ebqe[('diffusiveFlux_bc', 2, 2)]
        argsDict["csrColumnOffsets_eb_p_p"] = self.csrColumnOffsets_eb[(0, 0)]
        argsDict["csrColumnOffsets_eb_p_u"] = self.csrColumnOffsets_eb[(0, 0)]
        argsDict["csrColumnOffsets_eb_p_v"] = self.csrColumnOffsets_eb[(0, 1)]
        argsDict["csrColumnOffsets_eb_p_w"] = self.csrColumnOffsets_eb[(0, 2)]
        argsDict["csrColumnOffsets_eb_u_p"] = self.csrColumnOffsets_eb[(0, 0)]
        argsDict["csrColumnOffsets_eb_u_u"] = self.csrColumnOffsets_eb[(0, 0)]
        argsDict["csrColumnOffsets_eb_u_v"] = self.csrColumnOffsets_eb[(0, 1)]
        argsDict["csrColumnOffsets_eb_u_w"] = self.csrColumnOffsets_eb[(0, 2)]
        argsDict["csrColumnOffsets_eb_v_p"] = self.csrColumnOffsets_eb[(1, 0)]
        argsDict["csrColumnOffsets_eb_v_u"] = self.csrColumnOffsets_eb[(1, 0)]
        argsDict["csrColumnOffsets_eb_v_v"] = self.csrColumnOffsets_eb[(1, 1)]
        argsDict["csrColumnOffsets_eb_v_w"] = self.csrColumnOffsets_eb[(1, 2)]
        argsDict["csrColumnOffsets_eb_w_p"] = self.csrColumnOffsets_eb[(2, 0)]
        argsDict["csrColumnOffsets_eb_w_u"] = self.csrColumnOffsets_eb[(2, 0)]
        argsDict["csrColumnOffsets_eb_w_v"] = self.csrColumnOffsets_eb[(2, 1)]
        argsDict["csrColumnOffsets_eb_w_w"] = self.csrColumnOffsets_eb[(2, 2)]
        argsDict["elementFlags"] = self.mesh.elementMaterialTypes
        argsDict["nParticles"] = self.coefficients.nParticles
        argsDict["particle_epsFact"] = self.coefficients.particle_epsFact
        argsDict["particle_alpha"] = self.coefficients.particle_alpha
        argsDict["particle_beta"] = self.coefficients.particle_beta
        argsDict["particle_penalty_constant"] = self.coefficients.particle_penalty_constant
        argsDict["particle_signed_distances"] = self.coefficients.particle_signed_distances
        argsDict["particle_signed_distance_normals"] = self.coefficients.particle_signed_distance_normals
        argsDict["particle_velocities"] = self.coefficients.particle_velocities
        argsDict["particle_centroids"] = self.coefficients.particle_centroids
        argsDict["particle_nitsche"] = self.coefficients.particle_nitsche
        argsDict["use_ball_as_particle"] = self.coefficients.use_ball_as_particle
        argsDict["ball_center"] = self.coefficients.ball_center
        argsDict["ball_radius"] = self.coefficients.ball_radius
        argsDict["ball_velocity"] = self.coefficients.ball_velocity
        argsDict["ball_angular_velocity"] = self.coefficients.ball_angular_velocity
        argsDict["USE_SUPG"] = int(self.coefficients.USE_SUPG)
        argsDict["KILL_PRESSURE_TERM"] = int(self.KILL_PRESSURE_TERM)
        argsDict["dt"] = self.timeIntegration.dt
        argsDict["MATERIAL_PARAMETERS_AS_FUNCTION"] = int(self.hasMaterialParametersAsFunctions)
        argsDict["density_as_function"] = self.q['density']
        argsDict["dynamic_viscosity_as_function"] = self.q['dynamic_viscosity']
        argsDict["ebqe_density_as_function"] = self.ebqe['density']
        argsDict["ebqe_dynamic_viscosity_as_function"] = self.ebqe['dynamic_viscosity']
        argsDict["USE_SBM"] = self.coefficients.use_sbm
        argsDict["ARTIFICIAL_VISCOSITY"] = self.coefficients.ARTIFICIAL_VISCOSITY
        argsDict["uStar_dMatrix"] = self.uStar_dMatrix
        argsDict["vStar_dMatrix"] = self.vStar_dMatrix
        argsDict["wStar_dMatrix"] = self.wStar_dMatrix
        argsDict["numDOFs_1D"] = self.numDOFs_1D
        argsDict["offset_u"] = self.offset[0]
        argsDict["offset_v"] = self.offset[1]
        argsDict["offset_w"] = self.offset[2]
        argsDict["stride_u"] = self.stride[0]
        argsDict["stride_v"] = self.stride[1]
        argsDict["stride_w"] = self.stride[2]
        argsDict["rowptr_1D"] = self.rowptr_1D
        argsDict["colind_1D"] = self.colind_1D
        argsDict["rowptr"] = rowptr
        argsDict["colind"] = colind
        argsDict["INT_BY_PARTS_PRESSURE"] = self.coefficients.INT_BY_PARTS_PRESSURE
        self.rans3pf.calculateJacobian(
            argsDict,
            self.coefficients.useExact)

        if not self.forceStrongConditions and max(
            np.linalg.norm(
                self.u[0].dof, np.inf), np.linalg.norm(
                self.u[1].dof, np.inf), np.linalg.norm(
                self.u[2].dof, np.inf)) < 1.0e-8:
            self.pp_hasConstantNullSpace = True
        else:
            self.pp_hasConstantNullSpace = False
        # Load the Dirichlet conditions directly into residual
        if self.forceStrongConditions:
            for cj in range(self.nc):
                for dofN in list(self.dirichletConditionsForceDOF[
                        cj].DOFBoundaryConditionsDict.keys()):
                    global_dofN = self.offset[cj] + self.stride[cj] * dofN
                    for i in range(
                        self.rowptr[global_dofN], self.rowptr[
                            global_dofN + 1]):
                        if (self.colind[i] == global_dofN):
                            self.nzval[i] = 1.0
                        else:
                            self.nzval[i] = 0.0
                            # print "RBLES zeroing residual cj = %s dofN= %s
                            # global_dofN= %s " % (cj,dofN,global_dofN)
        # sb method
        for global_dofN in np.where(self.isActiveDOF==0.0)[0]:
            for i in range(
                    self.rowptr[global_dofN],
                    self.rowptr[global_dofN + 1]):
                if (self.colind[i] == global_dofN):
                    self.nzval[i] = 1.0
                else:
                    self.nzval[i] = 0.0
        log("Jacobian ", level=10, data=jacobian)
        # mwf decide if this is reasonable for solver statistics
        self.nonlinear_function_jacobian_evaluations += 1
        return jacobian

    def calculateElementQuadrature(self, domainMoved=False):
        """
        Calculate the physical location and weights of the quadrature rules
        and the shape information at the quadrature points.

        This function should be called only when the mesh changes.
        """
        if self.postProcessing:
            self.u[0].femSpace.elementMaps.getValues(
                self.elementQuadraturePoints, self.q['x'])
            self.u[0].femSpace.elementMaps.getJacobianValues(
                self.elementQuadraturePoints,
                self.q['J'],
                self.q['inverse(J)'],
                self.q['det(J)'])
            self.q['abs(det(J))'][:] = np.abs(self.q['det(J)'])
            self.u[0].femSpace.getBasisValues(
                self.elementQuadraturePoints, self.q[('v', 0)])
        self.u[0].femSpace.elementMaps.getBasisValuesRef(
            self.elementQuadraturePoints)
        self.u[0].femSpace.elementMaps.getBasisGradientValuesRef(
            self.elementQuadraturePoints)
        self.u[0].femSpace.getBasisValuesRef(self.elementQuadraturePoints)
        self.u[0].femSpace.getBasisGradientValuesRef(
            self.elementQuadraturePoints)
        self.u[0].femSpace.getBasisHessianValuesRef(
            self.elementQuadraturePoints)
        self.u[1].femSpace.getBasisValuesRef(self.elementQuadraturePoints)
        self.u[1].femSpace.getBasisGradientValuesRef(
            self.elementQuadraturePoints)
        self.coefficients.initializeElementQuadrature(
            self.timeIntegration.t, self.q)
        if self.stabilization is not None and not domainMoved:
            self.stabilization.initializeElementQuadrature(
                self.mesh, self.timeIntegration.t, self.q)
            self.stabilization.initializeTimeIntegration(self.timeIntegration)
        if self.shockCapturing is not None and not domainMoved:
            self.shockCapturing.initializeElementQuadrature(
                self.mesh, self.timeIntegration.t, self.q)

    def calculateElementBoundaryQuadrature(self, domainMoved=False):
        """
        Calculate the physical location and weights of the quadrature rules
        and the shape information at the quadrature points on element boundaries.

        This function should be called only when the mesh changes.
        """
        if self.postProcessing:
            self.u[0].femSpace.elementMaps.getValuesTrace(
                self.elementBoundaryQuadraturePoints, self.ebq['x'])
            self.u[0].femSpace.elementMaps.getJacobianValuesTrace(
                self.elementBoundaryQuadraturePoints,
                self.ebq['inverse(J)'],
                self.ebq['g'],
                self.ebq['sqrt(det(g))'],
                self.ebq['n'])
            cfemIntegrals.copyLeftElementBoundaryInfo(
                self.mesh.elementBoundaryElementsArray,
                self.mesh.elementBoundaryLocalElementBoundariesArray,
                self.mesh.exteriorElementBoundariesArray,
                self.mesh.interiorElementBoundariesArray,
                self.ebq['x'],
                self.ebq['n'],
                self.ebq_global['x'],
                self.ebq_global['n'])
            self.u[0].femSpace.elementMaps.getInverseValuesTrace(
                self.ebq['inverse(J)'], self.ebq['x'], self.ebq['hat(x)'])
            self.u[0].femSpace.elementMaps.getPermutations(self.ebq['hat(x)'])
            self.testSpace[0].getBasisValuesTrace(
                self.u[0].femSpace.elementMaps.permutations, self.ebq['hat(x)'], self.ebq[
                    ('w', 0)])
            self.u[0].femSpace.getBasisValuesTrace(
                self.u[0].femSpace.elementMaps.permutations, self.ebq['hat(x)'], self.ebq[
                    ('v', 0)])
            cfemIntegrals.calculateElementBoundaryIntegrationWeights(
                self.ebq['sqrt(det(g))'], self.elementBoundaryQuadratureWeights[
                    ('u', 0)], self.ebq[
                    ('dS_u', 0)])

    def calculateExteriorElementBoundaryQuadrature(self, domainMoved=False):
        """
        Calculate the physical location and weights of the quadrature rules
        and the shape information at the quadrature points on global element boundaries.

        This function should be called only when the mesh changes.
        """
        log("initalizing ebqe vectors for post-procesing velocity")
        if self.postProcessing:
            self.u[0].femSpace.elementMaps.getValuesGlobalExteriorTrace(
                self.elementBoundaryQuadraturePoints, self.ebqe['x'])
            self.u[0].femSpace.elementMaps.getJacobianValuesGlobalExteriorTrace(
                self.elementBoundaryQuadraturePoints,
                self.ebqe['inverse(J)'],
                self.ebqe['g'],
                self.ebqe['sqrt(det(g))'],
                self.ebqe['n'])
            cfemIntegrals.calculateIntegrationWeights(
                self.ebqe['sqrt(det(g))'], self.elementBoundaryQuadratureWeights[
                    ('u', 0)], self.ebqe[
                    ('dS_u', 0)])
        #
        # get physical locations of element boundary quadrature points
        #
        # assume all components live on the same mesh
        log("initalizing basis info")
        self.u[0].femSpace.elementMaps.getBasisValuesTraceRef(
            self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.elementMaps.getBasisGradientValuesTraceRef(
            self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.getBasisValuesTraceRef(
            self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.getBasisGradientValuesTraceRef(
            self.elementBoundaryQuadraturePoints)
        self.u[1].femSpace.getBasisValuesTraceRef(
            self.elementBoundaryQuadraturePoints)
        self.u[1].femSpace.getBasisGradientValuesTraceRef(
            self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.elementMaps.getValuesGlobalExteriorTrace(
            self.elementBoundaryQuadraturePoints, self.ebqe['x'])
        log("setting flux boundary conditions")
        if not domainMoved:
            self.fluxBoundaryConditionsObjectsDict = dict([(cj, FluxBoundaryConditions(self.mesh,
                                                                                       self.nElementBoundaryQuadraturePoints_elementBoundary,
                                                                                       self.ebqe[('x')],
                                                                                       self.advectiveFluxBoundaryConditionsSetterDict[cj],
                                                                                       self.diffusiveFluxBoundaryConditionsSetterDictDict[cj]))
                                                           for cj in list(self.advectiveFluxBoundaryConditionsSetterDict.keys())])
            log("initializing coefficients ebqe")
            self.coefficients.initializeGlobalExteriorElementBoundaryQuadrature(
                self.timeIntegration.t, self.ebqe)
        log("done with ebqe")

    def estimate_mt(self):
        pass

    def calculateSolutionAtQuadrature(self):
        pass

    def calculateAuxiliaryQuantitiesAfterStep(self):
        if self.postProcessing and self.conservativeFlux:
            argsDict = cArgumentsDict.ArgumentsDict()
            argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
            argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
            argsDict["nInteriorElementBoundaries_global"] = self.mesh.nInteriorElementBoundaries_global
            argsDict["interiorElementBoundariesArray"] = self.mesh.interiorElementBoundariesArray
            argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
            argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
            argsDict["mesh_dof"] = self.mesh.nodeArray
            argsDict["mesh_velocity_dof"] = self.mesh.nodeVelocityArray
            argsDict["MOVING_DOMAIN"] = self.MOVING_DOMAIN
            argsDict["mesh_l2g"] = self.mesh.elementNodesArray
            argsDict["mesh_trial_trace_ref"] = self.u[0].femSpace.elementMaps.psi_trace
            argsDict["mesh_grad_trial_trace_ref"] = self.u[0].femSpace.elementMaps.grad_psi_trace
            argsDict["normal_ref"] = self.u[0].femSpace.elementMaps.boundaryNormals
            argsDict["boundaryJac_ref"] = self.u[0].femSpace.elementMaps.boundaryJacobians
            argsDict["vel_l2g"] = self.u[0].femSpace.dofMap.l2g
            argsDict["u_dof"] = self.u[0].dof
            argsDict["v_dof"] = self.u[1].dof
            argsDict["w_dof"] = self.u[2].dof
            argsDict["vos_dof"] = self.coefficients.vos_dof
            argsDict["vel_trial_trace_ref"] = self.u[0].femSpace.psi_trace
            argsDict["ebqe_velocity"] = self.ebqe[('velocity', 0)]
            argsDict["velocityAverage"] = self.ebq_global[('velocityAverage', 0)]
            self.rans3pf.calculateVelocityAverage(argsDict)
            if self.movingDomain:
                log("Element Quadrature", level=3)
                self.calculateElementQuadrature(domainMoved=True)
                log("Element Boundary Quadrature", level=3)
                self.calculateElementBoundaryQuadrature(domainMoved=True)
                log("Global Exterior Element Boundary Quadrature", level=3)
                self.calculateExteriorElementBoundaryQuadrature(
                    domainMoved=True)
                for ci in range(
                        len(self.velocityPostProcessor.vpp_algorithms)):
                    for cj in list(self.velocityPostProcessor.vpp_algorithms[
                            ci].updateConservationJacobian.keys()):
                        self.velocityPostProcessor.vpp_algorithms[
                            ci].updateWeights()
                        self.velocityPostProcessor.vpp_algorithms[
                            ci].computeGeometricInfo()
                        self.velocityPostProcessor.vpp_algorithms[
                            ci].updateConservationJacobian[cj] = True

        self.q['velocityError'][:] = self.q[('velocity', 0)]
        OneLevelTransport.calculateAuxiliaryQuantitiesAfterStep(self)
        self.q['velocityError'] -= self.q[('velocity', 0)]
        if 'phis' in self.q and self.coefficients.granular_sdf_Calc is not None:
            self.q['phisError'] = self.q[('phis')]
        else:  # this needs to be fixed for the case that multiple bodies are present
            if ('phis', 0) in self.q:
                self.q['phisError'][:] = self.q[('phis', 0)]

    def updateAfterMeshMotion(self):
        pass
