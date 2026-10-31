"""
Optimized  Two-Phase Reynolds Averaged Navier-Stokes
"""
import math
import proteus
import sys
from proteus.mprans.cRANS2P import *
from proteus.mprans.cRANS2P2D import *
from proteus import Profiling
from proteus import LinearAlgebraTools as LAT
from proteus.Comm import (globalSum,
                          globalMax)
import numpy
from proteus import *
from proteus.Transport import *
from proteus.Transport import OneLevelTransport
from . import cArgumentsDict
import proteus.NumericalFlux
import proteus.Profiling
import proteus.ShockCapturing
import proteus.SubgridError
import proteus.Transport
import proteus.TransportCoefficients

class SubgridError(proteus.SubgridError.SGE_base):
    """
    Create a SubgridError  object for two-phase incompressible flow

    The VMS subgrid error approximation

    Parameters
    ----------
        coefficients  : proteus.TransportCoefficients.TC_base
            The coefficients object
        nd            : int
            Number of space  dimensions
        lag           : bool
            Use prior time step to calculate
        nStepsToDelay : int
            Lag only after nSteps
        hFactor       : float
            scaling factor based on order
        noPressureStabilization : bool
            turn off pressure stab
    """

    def __init__(self, coefficients, nd, lag=False, nStepsToDelay=1, hFactor=1.0, noPressureStabilization=False):
        self.noPressureStabilization = noPressureStabilization
        proteus.SubgridError.SGE_base.__init__(self, coefficients, nd, lag)
        coefficients.stencil[0].add(0)
        self.hFactor = hFactor
        self.nStepsToDelay = nStepsToDelay
        self.nSteps = 0
        if self.lag:
            logEvent("RANS2P.SubgridError: lagging requested but must lag the first step; switching lagging off and delaying")
            #Ensure nStepsToDelay value makes sense by throwing an error
            if(self.nStepsToDelay is None or self.nStepsToDelay < 1):
                sys.exit("RANS2P.SubgridError: nStepsToDelay cannot be None or < 1 with lagging, please specify an integer: nStepsToDelay>=1")

    def initializeElementQuadrature(self, mesh, t, cq):
        """
        Allocated or set additional arrays for values at element quadrature

        Parameters
        ----------
        mesh : proteus.MeshTools.Mesh
           The mesh for the domain
        t    : float
           The current time
        cq   : dict
           The dictionary of element quadrature arrrays
        """
        import copy
        self.cq = cq
        #default behavior is to set v_last to point to cq[('velocity')]
        self.v_last = self.cq[('velocity', 0)]

    def updateSubgridErrorHistory(self, initializationPhase=False):
        if self.lag:
            if self.nSteps > self.nStepsToDelay:
                #at this point, v_last must be a separate object
                #update the values of v_last based on cq[('velocity')]
                self.v_last[:] = self.cq[('velocity', 0)]
            elif self.nSteps == self.nStepsToDelay:
                logEvent("RANS2P.SubgridError: switched to lagged subgrid error")
                #create a separate object identical to cq[('velocity')] 
                self.v_last = self.cq[('velocity', 0)].copy()
            else:
                pass
            self.nSteps += 1


    def calculateSubgridError(self, q):
        pass


class NumericalFlux(proteus.NumericalFlux.NavierStokes_Advection_DiagonalUpwind_Diffusion_SIPG_exterior):
    hasInterior = False

    def __init__(self, vt, getPointwiseBoundaryConditions,
                 getAdvectiveFluxBoundaryConditions,
                 getDiffusiveFluxBoundaryConditions,
                 getPeriodicBoundaryConditions=None):
        proteus.NumericalFlux.NavierStokes_Advection_DiagonalUpwind_Diffusion_SIPG_exterior.__init__(self, vt, getPointwiseBoundaryConditions,
                                                                                                     getAdvectiveFluxBoundaryConditions,
                                                                                                     getDiffusiveFluxBoundaryConditions, getPeriodicBoundaryConditions)
        self.penalty_constant = 2.0
        self.includeBoundaryAdjoint = True
        self.boundaryAdjoint_sigma = 1.0
        self.hasInterior = False


class ShockCapturing(proteus.ShockCapturing.ShockCapturing_base):
    def __init__(self, coefficients, nd, shockCapturingFactor=0.25, lag=False, nStepsToDelay=1):
        proteus.ShockCapturing.ShockCapturing_base.__init__(self, coefficients, nd, shockCapturingFactor, lag)
        self.nStepsToDelay = nStepsToDelay
        self.nSteps = 0
        if self.lag:
            logEvent("RANS2P.ShockCapturing: lagging requested but must lag the first step; switching lagging off and delaying")
            #Ensure nStepsToDelay value makes sense by throwing an error
            if(self.nStepsToDelay is None or self.nStepsToDelay < 1):
                sys.exit("RANS2P.ShockCapturing: nStepsToDelay cannot be None or < 1 with lagging, please specify an integer: nStepsToDelay>=1")

    def initializeElementQuadrature(self, mesh, t, cq):
        self.mesh = mesh
        self.numDiff = {}
        self.numDiff_last = {}
        #default behavior is to set both numDiff_last and numDiff to point to cq[('numDiff')]
        for ci in range(1, 4):
            self.numDiff[ci] = cq[('numDiff', ci, ci)]
            self.numDiff_last[ci] = cq[('numDiff', ci, ci)]

    def updateShockCapturingHistory(self):
        if self.lag:
            if self.nSteps > self.nStepsToDelay:
                #numDiff_last is a different object
                #update the values of numDiff_last based on numDiff, 
                for ci in range(1, 4):
                    self.numDiff_last[ci][:] = self.numDiff[ci]
            elif self.nSteps == self.nStepsToDelay:
                logEvent("RANS2P.ShockCapturing: switched to lagged shock capturing")
                #if lagging, then create a separate object identical to numDiff 
                for ci in range(1, 4):
                    self.numDiff_last[ci] = self.numDiff[ci].copy()
            else:
                pass
            self.nSteps += 1

            logEvent("RANS2P: max numDiff_1 %e numDiff_2 %e numDiff_3 %e" % (globalMax(self.numDiff_last[1].max()),
                                                                         globalMax(self.numDiff_last[2].max()),
                                                                         globalMax(self.numDiff_last[3].max())))

INSIDE_FLUID_DOMAIN=10000.0#ensure no embedded solid boundaries by default

class Coefficients(proteus.TransportCoefficients.TC_base):
    """
    The coefficients for two incompresslble fluids governed by the Navier-Stokes equations and separated by a sharp interface represented by a level set function

    Notes
    -----
    The PRESSURE_PROJECTION_STABILIZATION flag allows the user to use
    the Bochev-Dohrmann-Gunzburger stabilization introduced in
    Stabilization of Low-Order Mixed Finite Elements for the  Stokes
    Equation.  This option should be turned off for most problems,
    but in some instances it may produced better preconditioning
    results than the full SGE approach.
    """
    from proteus.ctransportCoefficients import TwophaseNavierStokes_ST_LS_SO_2D_Evaluate
    from proteus.ctransportCoefficients import TwophaseNavierStokes_ST_LS_SO_3D_Evaluate
    from proteus.ctransportCoefficients import TwophaseNavierStokes_ST_LS_SO_2D_Evaluate_sd
    from proteus.ctransportCoefficients import TwophaseNavierStokes_ST_LS_SO_3D_Evaluate_sd

    def __init__(self,
                 epsFact=1.5,
                 sigma=72.8,
                 rho_0=998.2, nu_0=1.004e-6,
                 rho_1=1.205, nu_1=1.500e-5,
                 g=[0.0, 0.0, -9.8],
                 nd=3,
                 ME_model=0,
                 CLSVOF_model=None,
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
                 dragAlpha=0.0,
                 dragBeta=0.0,
                 setParamsFunc=None,  # uses setParamsFunc if given
                 dragAlphaTypes=None,  # otherwise can use element constant values
                 dragBetaTypes=None,  # otherwise can use element constant values
                 porosityTypes=None,
                 killNonlinearDrag=False,
                 epsFact_solid=None,
                 epsFact_porous=None,
                 eb_adjoint_sigma=1.0,
                 eb_penalty_constant=100.0,
                 forceStrongDirichlet=False,
                 turbulenceClosureModel=0,  # 0=No Model, 1=Smagorinksy, 2=Dynamic Smagorinsky, 3=K-Epsilon, 4=K-Omega
                 smagorinskyConstant=0.1,
                 barycenters=None,
                 NONCONSERVATIVE_FORM=1.0,
                 MOMENTUM_SGE=1.0,
                 PRESSURE_SGE=1.0,
                 VELOCITY_SGE=1.0,
                 PRESSURE_PROJECTION_STABILIZATION=0.0,
                 phaseFunction=None,
                 LAG_LES=1.0,
                 use_ball_as_particle=1,
                 ball_center=None,
                 ball_mass=None,
                 ball_stiffness=None,
                 ball_force_range=None,
                 particle_cfl=0.001,
                 particle_box=[1.,1.,1.],
                 ball_radius=None,
                 ball_velocity=None,
                 ball_angular_velocity=None,
                 ball_center_acceleration=None,
                 ball_angular_acceleration=None,
                 ball_density=None,
                 particle_velocities=None,
                 particle_centroids=None,
                 particle_sdfList=None,
                 particle_velocityList=None,
                 nParticles = 0,
                 particle_epsFact=3.0,
                 particle_alpha=1000.0,
                 particle_beta=1000.0,
                 particle_penalty_constant=100.0,
                 ghost_penalty_constant=0.1,
                 particle_nitsche=1.0,
                 nullSpace='NoNullSpace',
                 useExact=False,
                 analyticalSolution=None,
                 initialize=True,
                 force_x=None,
                 force_y=None,
                 force_z=None,
                 normalize_pressure=False,
                 useInternalParticleSolver=False):
        self.projection_direction=np.array([1.0,0.0,0.0])
        self.phi_s_isSet=False
        self.normalize_pressure=normalize_pressure
        self.force_x=force_x
        self.force_y=force_y
        self.force_z=force_z        
        self.analyticalSolution=analyticalSolution
        self.useExact=useExact
        self.use_ball_as_particle = use_ball_as_particle
        self.useInternalParticleSolver=useInternalParticleSolver
        self.nParticles = nParticles
        self.particle_nitsche = particle_nitsche
        self.particle_epsFact = particle_epsFact
        self.particle_alpha = particle_alpha
        self.particle_beta = particle_beta
        self.particle_penalty_constant = particle_penalty_constant
        self.ghost_penalty_constant = ghost_penalty_constant
        self.particle_netForces = np.zeros((3*self.nParticles, 3), 'd')#####[total_force_1,total_force_2,...,stress_1,stress_2,...,pressure_1,pressure_2,...]  
        self.particle_netMoments = np.zeros((self.nParticles, 3), 'd')
        self.particle_surfaceArea = np.zeros((self.nParticles,), 'd')
        self.particle_surfaceArea_projected = np.zeros((self.nParticles,), 'd')
        self.particle_volume = np.zeros((self.nParticles,), 'd')
        if ball_center is None:
            self.ball_center = 1e10*numpy.ones((self.nParticles,3),'d')
        else:
            self.ball_center = ball_center

        if ball_mass is None:
            self.ball_mass = 1e10*numpy.ones((self.nParticles,),'d')
        else:
            self.ball_mass = ball_mass

        self.ball_stiffness = ball_stiffness
        self.ball_force_range = ball_force_range
        self.particle_cfl = particle_cfl
        self.L = np.array(particle_box)
        if ball_radius is None:
            self.ball_radius = 1e10*numpy.ones((self.nParticles,1),'d')
        else:
            self.ball_radius = ball_radius

        if ball_velocity is None:
            self.ball_velocity = numpy.zeros((self.nParticles,3),'d')
        else:
            self.ball_velocity = ball_velocity

        if ball_angular_velocity is None:
            self.ball_angular_velocity = numpy.zeros((self.nParticles,3),'d')
        else:
            self.ball_angular_velocity = ball_angular_velocity

        if ball_center_acceleration is None:
            self.ball_center_acceleration = numpy.zeros((self.nParticles,3),'d')
        else:
            self.ball_center_acceleration = ball_center_acceleration

        if ball_angular_acceleration is None:
            self.ball_angular_acceleration = numpy.zeros((self.nParticles,3),'d')
        else:
            self.ball_angular_acceleration = ball_angular_acceleration

        if ball_density is None:
            self.ball_density = rho_0*numpy.ones((self.nParticles,1),'d')
        else:
            self.ball_density = ball_density
        if particle_centroids is None:
            self.particle_centroids = 1e10*numpy.zeros((self.nParticles,3),'d')
        else:
            self.particle_centroids = particle_centroids
        self.particle_sdfList = particle_sdfList
        self.particle_velocityList = particle_velocityList
        self.ball_center = ball_center
        self.ball_radius = ball_radius
        self.ball_velocity = ball_velocity
        self.ball_angular_velocity = ball_angular_velocity
        self.ball_center_acceleration = ball_center_acceleration
        self.ball_angular_acceleration = ball_angular_acceleration
        self.ball_density = ball_density
        self.particle_centroids = particle_centroids
        if nParticles > 0 and use_ball_as_particle and useInternalParticleSolver:
            self.ball_FT = np.zeros((nParticles, 6),'d')
            self.ball_last_FT = np.zeros((nParticles,  6),'d')
            self.ball_h = np.zeros((nParticles, 3),'d')
            self.ball_last_h = np.zeros((nParticles, 3),'d')
            self.ball_center_last = ball_center.copy()
            self.ball_last_velocity = self.ball_velocity.copy()
            self.ball_last_angular_velocity = self.ball_angular_velocity.copy()
            #
            self.ball_Q = np.zeros((nParticles, 3,3),'d')
            self.ball_last_Q = np.zeros((nParticles, 3,3),'d')
            self.ball_Omega = np.zeros((nParticles, 3,3),'d')
            self.ball_last_Omega = np.zeros((nParticles, 3,3),'d')
            #
            self.ball_u = np.zeros((nParticles,18),'d')#linear and angular velocity, linear displacement, and rotation matrix
            self.ball_last_u = np.zeros((nParticles,18),'d')
            self.ball_mom = np.zeros((nParticles,6),'d')
            self.ball_last_mom = np.zeros((nParticles,6),'d')
            self.ball_a = np.zeros((nParticles,6),'d')
            #
            self.ball_I = np.zeros((nParticles, 3, 3),'d')
            self.ball_f = np.zeros((nParticles, 3),'d')
            self.wall_f = np.zeros((nParticles, 3),'d')
            self.last_particle_netForces = np.zeros((nParticles,3),'d')
            self.last_particle_netMoments = np.zeros((nParticles,3),'d')
            self.ball_angular_velocity_avg = self.ball_angular_velocity.copy()
            for ip in range(nParticles):
                if len(particle_box) == 2:
                    self.particle_g=np.array([g[0],g[1],0.0])
                    self.ball_I[ip] = np.eye(3)
                    self.ball_I[ip,0,0] = (1.0/12.0)*self.ball_mass[ip]*(3*ball_radius[ip]**2 + 1.0)
                    self.ball_I[ip,1,1] = (1.0/12.0)*self.ball_mass[ip]*(3*ball_radius[ip]**2 + 1.0)
                    self.ball_I[ip,2,2] = 0.5*self.ball_mass[ip]*ball_radius[ip]**2
                else:
                    self.particle_g=np.array(g)
                    self.ball_I[ip] = np.eye(3)*(2.0/5.0)*self.ball_mass[ip]*self.ball_radius[ip]**2
                self.ball_Q[ip] = np.eye(3)
                self.ball_last_Q[ip] = np.eye(3)
                self.ball_u[ip,9:] = self.ball_Q[ip].flatten()
            self.ball_last_u[:,:] = self.ball_u

        #
        self.LAG_LES=LAG_LES
        self.phaseFunction=phaseFunction
        self.NONCONSERVATIVE_FORM=NONCONSERVATIVE_FORM
        self.MOMENTUM_SGE=MOMENTUM_SGE
        self.PRESSURE_SGE=PRESSURE_SGE
        self.VELOCITY_SGE=VELOCITY_SGE
        self.PRESSURE_PROJECTION_STABILIZATION=PRESSURE_PROJECTION_STABILIZATION
        self.barycenters=barycenters
        self.smagorinskyConstant = smagorinskyConstant
        self.turbulenceClosureModel = turbulenceClosureModel
        self.forceStrongDirichlet = forceStrongDirichlet
        self.eb_adjoint_sigma = eb_adjoint_sigma
        self.eb_penalty_constant = eb_penalty_constant
        self.movingDomain = movingDomain
        self.epsFact_solid = epsFact_solid
        self.epsFact_porous = epsFact_porous
        self.useConstant_he = useConstant_he
        self.useVF = useVF
        self.useRBLES = useRBLES
        self.useMetrics = useMetrics
        self.sd = sd
        self.epsFact_density = epsFact_density
        self.stokes = stokes
        self.ME_model = ME_model
        self.CLSVOF_model = CLSVOF_model
        self.LS_model = LS_model
        self.VF_model = VF_model
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
        self.g = numpy.array(g)
        self.nd = nd
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
        self.nullSpace = nullSpace
        if(nd == 2):
            self.variableNames=['p','u','v']
        else:
            self.variableNames= ['p', 'u', 'v','w']
        if initialize:
            self.initialize()

    def initialize(self):
        if self.epsFact_density is None:
            self.epsFact_density = self.epsFact
        if self.particle_epsFact is None:
            self.particle_epsFact = self.epsFact
        self.particle_netForces = np.zeros((3*self.nParticles, 3), 'd')#####[total_force_1,total_force_2,...,stress_1,stress_2,...,pressure_1,pressure_2,...]  
        self.particle_netMoments = np.zeros((self.nParticles, 3), 'd')
        self.particle_surfaceArea = np.zeros((self.nParticles,), 'd')
        self.particle_surfaceArea_projected = np.zeros((self.nParticles,), 'd')
        self.particle_volume = np.zeros((self.nParticles,), 'd')
        if self.ball_center is None:
            self.ball_center = 1e10*numpy.ones((self.nParticles,3),'d')
        if self.ball_radius is None:
            self.ball_radius = 1e10*numpy.ones((self.nParticles,1),'d')
        if self.ball_velocity is None:
            self.ball_velocity = numpy.zeros((self.nParticles,3),'d')
        if self.ball_angular_velocity is None:
            self.ball_angular_velocity = numpy.zeros((self.nParticles,3),'d')
        if self.ball_center_acceleration is None:
            self.ball_center_acceleration = numpy.zeros((self.nParticles,3),'d')
        if self.ball_angular_acceleration is None:
            self.ball_angular_acceleration = numpy.zeros((self.nParticles,3),'d')
        if self.ball_density is None:
            self.ball_density = self.rho_0*numpy.ones((self.nParticles,1),'d')
        if self.particle_centroids is None:
            self.particle_centroids = 1e10*numpy.zeros((self.nParticles,3),'d')
        if self.killNonlinearDrag:
            self.nonlinearDragFactor = 0.0
        mass = {}
        advection = {}
        diffusion = {}
        potential = {}
        reaction = {}
        hamiltonian = {}
        if self.nd == 2:
            variableNames = ['p', 'u', 'v']
            mass = {1: {1: 'linear'},
                    2: {2: 'linear'}}
            advection = {0: {0: 'linear',
                             1: 'linear',
                             2: 'linear'},
                         1: {0: 'nonlinear',
                             1: 'nonlinear',
                             2: 'nonlinear'},
                         2: {0: 'nonlinear',
                             1: 'nonlinear',
                             2: 'nonlinear'}}
            diffusion = {1: {1: {1: 'constant'}, 2: {2: 'constant'}},
                         2: {2: {2: 'constant'}, 1: {1: 'constant'}}}
            sdInfo = {(1, 1): (numpy.array([0, 1, 2], dtype='i'),
                               numpy.array([0, 1], dtype='i')),
                      (1, 2): (numpy.array([0, 0, 1], dtype='i'),
                               numpy.array([0], dtype='i')),
                      (2, 2): (numpy.array([0, 1, 2], dtype='i'),
                               numpy.array([0, 1], dtype='i')),
                      (2, 1): (numpy.array([0, 1, 1], dtype='i'),
                               numpy.array([1], dtype='i'))}
            potential = {1: {1: 'u'},
                         2: {2: 'u'}}
            reaction = {0: {0: 'constant'},
                        1: {1: 'nonlinear', 2: 'nonlinear'},
                        2: {1: 'nonlinear', 2: 'nonlinear'}}
            hamiltonian = {1: {0: 'linear'},
                           2: {0: 'linear'}}
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
                             useSparseDiffusion=self.sd,
                             movingDomain=self.movingDomain)
            self.vectorComponents = [1, 2]
            self.vectorName = "velocity"
        elif self.nd == 3:
            variableNames = ['p', 'u', 'v', 'w']
            mass = {1: {1: 'linear'},
                    2: {2: 'linear'},
                    3: {3: 'linear'}}
            advection = {0: {1: 'linear',
                             2: 'linear',
                             3: 'linear'},
                         1: {0: 'nonlinear',
                             1: 'nonlinear',
                             2: 'nonlinear',
                             3: 'nonlinear'},
                         2: {0: 'nonlinear',
                             1: 'nonlinear',
                             2: 'nonlinear',
                             3: 'nonlinear'},
                         3: {0: 'nonlinear',
                             1: 'nonlinear',
                             2: 'nonlinear',
                             3: 'nonlinear'}}
            diffusion = {1: {1: {1: 'constant'}, 2: {2: 'constant'}, 3: {3: 'constant'}},
                         2: {1: {1: 'constant'}, 2: {2: 'constant'}, 3: {3: 'constant'}},
                         3: {1: {1: 'constant'}, 2: {2: 'constant'}, 3: {3: 'constant'}}}
            sdInfo = {}
            sdInfo = {(1, 1): (numpy.array([0, 1, 2, 3], dtype='i'), numpy.array([0, 1, 2], dtype='i')),
                      (1, 2): (numpy.array([0, 0, 1, 1], dtype='i'), numpy.array([0], dtype='i')),
                      (1, 3): (numpy.array([0, 0, 0, 1], dtype='i'), numpy.array([0], dtype='i')),
                      (2, 1): (numpy.array([0, 1, 1, 1], dtype='i'), numpy.array([1], dtype='i')),
                      (2, 2): (numpy.array([0, 1, 2, 3], dtype='i'), numpy.array([0, 1, 2], dtype='i')),
                      (2, 3): (numpy.array([0, 0, 0, 1], dtype='i'), numpy.array([1], dtype='i')),
                      (3, 1): (numpy.array([0, 1, 1, 1], dtype='i'), numpy.array([2], dtype='i')),
                      (3, 2): (numpy.array([0, 0, 1, 1], dtype='i'), numpy.array([2], dtype='i')),
                      (3, 3): (numpy.array([0, 1, 2, 3], dtype='i'), numpy.array([0, 1, 2], dtype='i'))}
            potential = {1: {1: 'u'},
                         2: {2: 'u'},
                         3: {3: 'u'}}
            reaction = {0: {0: 'constant'},
                        1: {1: 'nonlinear', 2: 'nonlinear', 3: 'nonlinear'},
                        2: {1: 'nonlinear', 2: 'nonlinear', 3: 'nonlinear'},
                        3: {1: 'nonlinear', 2: 'nonlinear', 3: 'nonlinear'}}
            hamiltonian = {1: {0: 'linear'},
                           2: {0: 'linear'},
                           3: {0: 'linear'}}
            TC_base.__init__(self,
                             4,
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames,
                             sparseDiffusionTensors=sdInfo,
                             useSparseDiffusion=self.sd,
                             movingDomain=self.movingDomain)
            self.vectorComponents = [1, 2, 3]
            self.vectorName = "velocity"

    def attachModels(self, modelList):
        # level set
        self.model = modelList[self.ME_model]
        if self.analyticalSolution is not None:
            for eN in range(self.model.q['x'].shape[0]):
                for k in range(self.model.q['x'].shape[1]):
                    self.model.q[('u', 0)][eN,k] = self.analyticalSolution[0].uOfXT(self.model.q['x'][eN,k],0.)
                    self.model.q[('u', 1)][eN,k] = self.analyticalSolution[1].uOfXT(self.model.q['x'][eN,k],0.)
                    self.model.q[('u', 2)][eN,k] = self.analyticalSolution[2].uOfXT(self.model.q['x'][eN,k],0.)
                    if self.nd == 3:
                        self.model.q[('u', 3)][eN,k] = self.analyticalSolution[3].uOfXT(self.model.q['x'][eN,k],0.)
        self.model.q['phi_solid'] = self.q_phi_solid
        self.model.q['velocity_solid'] = self.q_velocity_solid
        self.model.q['phi_porous'] = self.q_phi_porous
        self.model.q['velocity_porous'] = self.q_velocity_porous
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
        else: # use NCLS-RDLS-VOF-MCorr instead
            if self.LS_model is not None:
                self.q_phi = modelList[self.LS_model].q[('u', 0)]
                self.phi_dof = modelList[self.LS_model].u[0].dof
                if ('u', 0) in modelList[self.LS_model].ebq:
                    self.ebq_phi = modelList[self.LS_model].ebq[('u', 0)]
                else:
                    self.ebq_phi = None
                self.ebqe_phi = modelList[self.LS_model].ebqe[('u', 0)]
                self.bc_ebqe_phi = modelList[self.LS_model].numericalFlux.ebqe[('u', 0)]
                # normal
                self.q_n = modelList[self.LS_model].q[('grad(u)', 0)]
                if ('grad(u)', 0) in modelList[self.LS_model].ebq:
                    self.ebq_n = modelList[self.LS_model].ebq[('grad(u)', 0)]
                else:
                    self.ebq_n = None
                self.ebqe_n = modelList[self.LS_model].ebqe[('grad(u)', 0)]
            else:
                self.q_phi = -10.0 * numpy.ones(self.model.q[('u', 1)].shape, 'd')
                self.phi_dof = -10.0 * numpy.ones_like(self.model.u[0].dof)
                self.ebqe_phi = -10.0 * numpy.ones(self.model.ebqe[('u', 1)].shape, 'd')
                self.bc_ebqe_phi = -10.0 * numpy.ones(self.model.ebqe[('u', 1)].shape, 'd')
                self.q_n = numpy.ones(self.model.q[('velocity', 0)].shape, 'd')
                self.ebqe_n = numpy.ones(self.model.ebqe[('velocity', 0)].shape, 'd')
            if self.VF_model is not None:
                self.q_vf = modelList[self.VF_model].q[('u', 0)]
                if ('u', 0) in modelList[self.VF_model].ebq:
                    self.ebq_vf = modelList[self.VF_model].ebq[('u', 0)]
                else:
                    self.ebq_vf = None
                self.ebqe_vf = modelList[self.VF_model].ebqe[('u', 0)]
                self.bc_ebqe_vf = modelList[self.VF_model].numericalFlux.ebqe[('u', 0)]
            else:
                self.q_vf = numpy.zeros(self.model.q[('u', 1)].shape, 'd')
                self.ebqe_vf = numpy.zeros(self.model.ebqe[('u', 1)].shape, 'd')
                self.bc_ebqe_vf = numpy.zeros(self.model.ebqe[('u', 1)].shape, 'd')
        # curvature
        if self.KN_model is not None:
            self.q_kappa = modelList[self.KN_model].q[('u', 0)]
            self.ebqe_kappa = modelList[self.KN_model].ebqe[('u', 0)]
            if ('u', 0) in modelList[self.KN_model].ebq:
                self.ebq_kappa = modelList[self.KN_model].ebq[('u', 0)]
            else:
                self.ebq_kappa = None
        else:
            self.q_kappa = -numpy.ones(self.model.q[('u', 1)].shape, 'd')
            self.ebqe_kappa = -numpy.ones(self.model.ebqe[('u', 1)].shape, 'd')
        # Turbulence Closures
        # only option for now is k-epsilon
        self.q_turb_var = {}
        self.q_turb_var_grad = {}
        self.ebqe_turb_var = {}
        if self.Closure_0_model is not None:
            self.q_turb_var[0] = modelList[self.Closure_0_model].q[('u', 0)]
            self.q_turb_var_grad[0] = modelList[self.Closure_0_model].q[('grad(u)', 0)]
            self.ebqe_turb_var[0] = modelList[self.Closure_0_model].ebqe[('u', 0)]
        else:
            self.q_turb_var[0] = numpy.ones(self.model.q[('u', 1)].shape, 'd')
            self.q_turb_var_grad[0] = numpy.ones(self.model.q[('grad(u)', 1)].shape, 'd')
            self.ebqe_turb_var[0] = numpy.ones(self.model.ebqe[('u', 1)].shape, 'd')
        if self.Closure_1_model is not None:
            self.q_turb_var[1] = modelList[self.Closure_1_model].q[('u', 0)]
            self.ebqe_turb_var[1] = modelList[self.Closure_1_model].ebqe[('u', 0)]
        else:
            self.q_turb_var[1] = numpy.ones(self.model.q[('u', 1)].shape, 'd')
            self.ebqe_turb_var[1] = numpy.ones(self.model.ebqe[('u', 1)].shape, 'd')
        if self.epsFact_solid is None:
            self.epsFact_solid = numpy.ones(self.model.mesh.elementMaterialTypes.max() + 1)
        assert len(self.epsFact_solid) > self.model.mesh.elementMaterialTypes.max(
        ), "epsFact_solid  array is not large  enough for the materials  in this mesh; length must be greater  than largest  material type ID"
        if self.epsFact_porous is None:
            self.epsFact_porous = numpy.ones(self.model.mesh.elementMaterialTypes.max() + 1)
        assert len(self.epsFact_porous) > self.model.mesh.elementMaterialTypes.max(
        ), "epsFact_porous  array is not large  enough for the materials  in this mesh; length must be greater  than largest  material type ID"
        if self.phaseFunction != None:
            from proteus.ctransportCoefficients import smoothedHeaviside
            if self.useConstant_he:
                self.elementDiameter = self.mesh.elementDiametersArray.copy()
                self.elementDiameter[:] = max(self.mesh.elementDiametersArray)
            else:
                self.elementDiameter = self.mesh.elementDiametersArray
            for i, quad_pts in enumerate(self.model.q['x']):
                for j, pt in enumerate(quad_pts):
                    he = self.elementDiameter[i]
                    self.q_phi[i, j] = self.phaseFunction(pt)
                    self.q_vf[i, j] = smoothedHeaviside(self.epsFact * he, self.phaseFunction(pt))
            for i, quad_pts in enumerate(self.model.ebqe['x']):
                # ebqe is indexed by exterior element boundary, not by element: use the diameter of the element
                # that owns the boundary. (On a partition with more exterior boundaries than elements the old
                # elementDiameter[i] ran off the end; elsewhere it silently used an unrelated element's size.)
                ebN = self.mesh.exteriorElementBoundariesArray[i]
                he = self.elementDiameter[self.mesh.elementBoundaryElementsArray[ebN, 0]]
                for j, pt in enumerate(quad_pts):
                    self.ebqe_phi[i, j] = self.phaseFunction(pt)
                    self.ebqe_vf[i, j] = smoothedHeaviside(self.epsFact * he, self.phaseFunction(pt))
        if self.use_ball_as_particle:
            self.particle_signed_distances        = 1e10*numpy.ones((1,self.model.q['x'].shape[0],self.model.q['x'].shape[1]),'d')
            self.particle_signed_distance_normals = 1e10*numpy.ones((1,self.model.q['x'].shape[0],self.model.q['x'].shape[1], 3),'d')
            self.particle_velocities              = 1e10*numpy.ones((1,self.model.q['x'].shape[0],self.model.q['x'].shape[1], 3),'d')
        else:
            self.particle_signed_distances        = 1e10*numpy.ones((self.nParticles,self.model.q['x'].shape[0],self.model.q['x'].shape[1]),'d')
            self.particle_signed_distance_normals = 1e10*numpy.ones((self.nParticles,self.model.q['x'].shape[0],self.model.q['x'].shape[1], 3),'d')
            self.particle_velocities              = 1e10*numpy.ones((self.nParticles,self.model.q['x'].shape[0],self.model.q['x'].shape[1], 3),'d')
        self.phisField                        = 1e10*numpy.ones((self.model.q['x'].shape[0],self.model.q['x'].shape[1]), 'd')
        self.ebqe_phi_s        = numpy.ones((self.model.ebqe['x'].shape[0],self.model.ebqe['x'].shape[1]),'d') * 1e10
        self.ebq_global_grad_phi_s   = numpy.ones((self.model.ebq_global['x'].shape[0],self.model.ebq_global['x'].shape[1],3),'d') * 1e10
        self.ebq_particle_velocity_s = numpy.ones((self.model.ebq_global['x'].shape[0],self.model.ebq_global['x'].shape[1],3),'d') * 1e10
        self.p_old_dof = self.model.u[0].dof.copy()
        self.u_old_dof = self.model.u[1].dof.copy()
        self.v_old_dof = self.model.u[2].dof.copy()
        self.w_old_dof = self.model.u[3].dof.copy()
        if self.nParticles > 0 and self.particle_sdfList != None and not self.use_ball_as_particle:
            self.phi_s_isSet=True
            self.phi_s[:] = 1e10
            self.phisField[:] = 1e10
            self.ebqe_phi_s[:] = 1e10
            self.ebq_global_grad_phi_s[:] = 1e10
            self.ebq_particle_velocity_s[:] = 1e10
            t=0.0
            for i in range(self.nParticles):
                try:
                    assert self.nParticles == 1
                    self.particle_sdfList[0](t, np.reshape(self.mesh.nodeArray, (self.mesh.nodeArray.size//3,3)),
                                             np.reshape(self.phi_s, (self.phi_s.size,)))
                    self.particle_sdfList[0](t, np.reshape(self.model.q['x'], (self.model.q['x'].size//3,3)),
                                             np.reshape(self.phisField, (self.phisField.size,)))
                    self.particle_sdfList[0](t, np.reshape(self.model.ebqe['x'], (self.model.ebqe['x'].size//3,3)),
                                             np.reshape(self.ebqe_phi_s, (self.ebqe_phi_s.size,)))
                    self.particle_velocities[...,:]=0.0
                    self.ebq_particle_velocity_s[...,:] = 0.0
                    self.particle_signed_distance_normals[...,:] = 0.0
                except:
                    vel = lambda x: self.particle_velocityList[i](t, x)
                    sdf = lambda x: self.particle_sdfList[i](t, x)
                    for j in range(self.mesh.nodeArray.shape[0]):
                        sdf_at_node, sdNormals = sdf(self.mesh.nodeArray[j, :])
                        if (sdf_at_node < self.phi_s[j]):
                            self.phi_s[j] = sdf_at_node
                    for eN in range(self.model.q['x'].shape[0]):
                        for k in range(self.model.q['x'].shape[1]):
                            self.particle_signed_distances[i, eN, k], self.particle_signed_distance_normals[i, eN, k,:] = sdf(self.model.q['x'][eN, k])
                            self.particle_velocities[i, eN, k,:] = vel(self.model.q['x'][eN, k])
                            if (self.particle_signed_distances[i, eN, k] < self.phisField[eN, k]):
                                self.phisField[eN, k] = self.particle_signed_distances[i, eN, k]
                    for ebNE in range(self.model.ebqe['x'].shape[0]):
                        for kb in range(self.model.ebqe['x'].shape[1]):
                            sdf_ebNE_kb,sdNormals = sdf(self.model.ebqe['x'][ebNE,kb])
                            if (sdf_ebNE_kb < self.ebqe_phi_s[ebNE,kb]):
                                self.ebqe_phi_s[ebNE,kb]=sdf_ebNE_kb

    def initializeMesh(self, mesh):
        self.phi_s = numpy.ones(mesh.nodeArray.shape[0], 'd')*1e10#
        # cek we eventually need to use the local element diameter
        self.eps_density = self.epsFact_density * mesh.h
        self.eps_viscosity = self.epsFact * mesh.h
        self.mesh = mesh
        self.elementMaterialTypes = mesh.elementMaterialTypes
        nBoundariesMax = int(globalMax(max(self.mesh.elementBoundaryUserFlags))) + 1
        self.wettedAreas = numpy.zeros((nBoundariesMax,), 'd')
        self.netForces_p = numpy.zeros((nBoundariesMax, 3), 'd')
        self.netForces_v = numpy.zeros((nBoundariesMax, 3), 'd')
        self.netMoments = numpy.zeros((nBoundariesMax, 3), 'd')
        if self.barycenters is None:
            self.barycenters = numpy.zeros((nBoundariesMax, 3), 'd')
        comm = Comm.get()
        import os
        if comm.isMaster():
            self.history_file = open(os.path.join(proteus.Profiling.logDir, "particles.txt"),"ab")
            self.timeHistory = open(os.path.join(proteus.Profiling.logDir, "timeHistory.txt"), "w")
            self.wettedAreaHistory = open(os.path.join(proteus.Profiling.logDir, "wettedAreaHistory.txt"), "w")
            self.forceHistory_p = open(os.path.join(proteus.Profiling.logDir, "forceHistory_p.txt"), "w")
            self.forceHistory_v = open(os.path.join(proteus.Profiling.logDir, "forceHistory_v.txt"), "w")
            self.momentHistory = open(os.path.join(proteus.Profiling.logDir, "momentHistory.txt"), "w")
            if self.nParticles:
                self.particle_forceHistory = open(os.path.join(proteus.Profiling.logDir, "particle_forceHistory.txt"), "w")
                self.particle_pforceHistory = open(os.path.join(proteus.Profiling.logDir, "particle_pforceHistory.txt"), "w")
                self.particle_vforceHistory = open(os.path.join(proteus.Profiling.logDir, "particle_vforceHistory.txt"), "w")
                self.particle_momentHistory = open(os.path.join(proteus.Profiling.logDir, "particle_momentHistory.txt"), "w")
        self.comm = comm
    # initialize so it can run as single phase

    def initializeElementQuadrature(self, t, cq):
        # VRANS
        self.numerical_viscosity = numpy.zeros(cq[('u', 1)].shape, 'd')
        self.q_phi_solid = INSIDE_FLUID_DOMAIN*numpy.ones(cq[('u', 1)].shape, 'd')
        self.q_velocity_solid = numpy.zeros(cq[('velocity', 0)].shape, 'd')
        self.q_phi_porous = INSIDE_FLUID_DOMAIN*numpy.ones(cq[('u', 1)].shape, 'd')
        self.q_velocity_porous = numpy.zeros(cq[('velocity', 0)].shape, 'd')
        self.q_porosity = numpy.ones(cq[('u', 1)].shape, 'd')
        self.q_dragAlpha = numpy.ones(cq[('u', 1)].shape, 'd')
        self.q_dragAlpha.fill(self.dragAlpha)
        self.q_dragBeta = numpy.ones(cq[('u', 1)].shape, 'd')
        self.q_dragBeta.fill(self.dragBeta)
        if self.setParamsFunc is not None:
            self.setParamsFunc(cq['x'], self.q_porosity, self.q_dragAlpha, self.q_dragBeta)
        else:
            # TODO make loops faster
            if self.porosityTypes is not None:
                for eN in range(self.q_porosity.shape[0]):
                    self.q_porosity[eN, :] = self.porosityTypes[self.elementMaterialTypes[eN]]
            if self.dragAlphaTypes is not None:
                for eN in range(self.q_dragAlpha.shape[0]):
                    self.q_dragAlpha[eN, :] = self.dragAlphaTypes[self.elementMaterialTypes[eN]]
            if self.dragBetaTypes is not None:
                for eN in range(self.q_dragBeta.shape[0]):
                    self.q_dragBeta[eN, :] = self.dragBetaTypes[self.elementMaterialTypes[eN]]
        cq['velocityError'] = cq[('velocity', 0)].copy()
        #

    def initializeElementBoundaryQuadrature(self, t, cebq, cebq_global):
        # VRANS
        self.ebq_porosity = numpy.ones(cebq['det(J)'].shape, 'd')
        self.ebq_dragAlpha = numpy.ones(cebq['det(J)'].shape, 'd')
        self.ebq_dragAlpha.fill(self.dragAlpha)
        self.ebq_dragBeta = numpy.ones(cebq['det(J)'].shape, 'd')
        self.ebq_dragBeta.fill(self.dragBeta)
        if self.setParamsFunc is not None:
            self.setParamsFunc(cebq['x'], self.ebq_porosity, self.ebq_dragAlpha, self.ebq_dragBeta)
        # TODO which mean to use or leave discontinuous
        # TODO make loops faster
        if self.porosityTypes is not None:
            for ebNI in range(self.mesh.nInteriorElementBoundaries_global):
                ebN = self.mesh.interiorElementBoundariesArray[ebNI]
                eN_left = self.mesh.elementBoundaryElementsArray[ebN, 0]
                eN_right = self.mesh.elementBoundaryElementsArray[ebN, 1]
                ebN_element_left = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN, 0]
                ebN_element_right = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN, 1]
                avg = 0.5 * (self.porosityTypes[self.elementMaterialTypes[eN_left]] +
                             self.porosityTypes[self.elementMaterialTypes[eN_right]])
                self.ebq_porosity[eN_left, ebN_element_left, :] = self.porosityTypes[self.elementMaterialTypes[eN_left]]
                self.ebq_porosity[eN_right, ebN_element_right, :] = self.porosityTypes[self.elementMaterialTypes[eN_right]]
            for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
                ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
                eN = self.mesh.elementBoundaryElementsArray[ebN, 0]
                ebN_element = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN, 0]
                self.ebq_porosity[eN, ebN_element, :] = self.porosityTypes[self.elementMaterialTypes[eN]]
        if self.dragAlphaTypes is not None:
            for ebNI in range(self.mesh.nInteriorElementBoundaries_global):
                ebN = self.mesh.interiorElementBoundariesArray[ebNI]
                eN_left = self.mesh.elementBoundaryElementsArray[ebN, 0]
                eN_right = self.mesh.elementBoundaryElementsArray[ebN, 1]
            ebN_element_left = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN, 0]
            ebN_element_right = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN, 1]
            avg = 0.5 * (self.dragAlphaTypes[self.elementMaterialTypes[eN_left]] +
                         self.dragAlphaTypes[self.elementMaterialTypes[eN_right]])
            self.ebq_dragAlpha[eN_left, ebN_element_left, :] = self.dragAlphaTypes[self.elementMaterialTypes[eN_left]]
            self.ebq_dragAlpha[eN_right, ebN_element_right, :] = self.dragAlphaTypes[self.elementMaterialTypes[eN_right]]
            for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
                ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
                eN = self.mesh.elementBoundaryElementsArray[ebN, 0]
                ebN_element = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN, 0]
                self.ebq_dragAlpha[eN, ebN_element, :] = self.dragAlphaTypes[self.elementMaterialTypes[eN]]
        if self.dragBetaTypes is not None:
            for ebNI in range(self.mesh.nInteriorElementBoundaries_global):
                ebN = self.mesh.interiorElementBoundariesArray[ebNI]
                eN_left = self.mesh.elementBoundaryElementsArray[ebN, 0]
                eN_right = self.mesh.elementBoundaryElementsArray[ebN, 1]
            ebN_element_left = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN, 0]
            ebN_element_right = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN, 1]
            avg = 0.5 * (self.dragBetaTypes[self.elementMaterialTypes[eN_left]] +
                         self.dragBetaTypes[self.elementMaterialTypes[eN_right]])
            self.ebq_dragBeta[eN_left, ebN_element_left, :] = self.dragBetaTypes[self.elementMaterialTypes[eN_left]]
            self.ebq_dragBeta[eN_right, ebN_element_right, :] = self.dragBetaTypes[self.elementMaterialTypes[eN_right]]
            for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
                ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
                eN = self.mesh.elementBoundaryElementsArray[ebN, 0]
                ebN_element = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN, 0]
                self.ebq_dragBeta[eN, ebN_element, :] = self.dragBetaTypes[self.elementMaterialTypes[eN]]
         #

    def initializeGlobalExteriorElementBoundaryQuadrature(self, t, cebqe):
        # VRANS
        logEvent("ebqe_global allocations in coefficients")
        self.ebqe_porosity = numpy.ones(cebqe[('u', 1)].shape, 'd')
        self.ebqe_dragAlpha = numpy.ones(cebqe[('u', 1)].shape, 'd')
        self.ebqe_dragAlpha.fill(self.dragAlpha)
        self.ebqe_dragBeta = numpy.ones(cebqe[('u', 1)].shape, 'd')
        self.ebqe_dragBeta.fill(self.dragBeta)
        logEvent("porosity and drag")
        # TODO make loops faster
        if self.setParamsFunc is not None:
            self.setParamsFunc(cebqe['x'], self.ebqe_porosity, self.ebqe_dragAlpha, self.ebqe_dragBeta)
        else:
            if self.porosityTypes is not None:
                for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
                    ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
                    eN = self.mesh.elementBoundaryElementsArray[ebN, 0]
                    self.ebqe_porosity[ebNE, :] = self.porosityTypes[self.elementMaterialTypes[eN]]
            if self.dragAlphaTypes is not None:
                for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
                    ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
                    eN = self.mesh.elementBoundaryElementsArray[ebN, 0]
                    self.ebqe_dragAlpha[ebNE, :] = self.dragAlphaTypes[self.elementMaterialTypes[eN]]
            if self.dragBetaTypes is not None:
                for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
                    ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
                    eN = self.mesh.elementBoundaryElementsArray[ebN, 0]
                    self.ebqe_dragBeta[ebNE, :] = self.dragBetaTypes[self.elementMaterialTypes[eN]]
        #

    def updateToMovingDomain(self, t, c):
        pass

    def evaluate(self, t, c):
        logEvent("Evaluating Coefficients")

    def preStep(self, t, firstStep=False):
        self.model.dt_last = self.model.timeIntegration.dt
        if self.analyticalSolution is not None:
            for eN in range(self.model.q['x'].shape[0]):
                for k in range(self.model.q['x'].shape[1]):
                    self.model.q[('u', 0)][eN,k] = self.analyticalSolution[0].uOfXT(self.model.q['x'][eN,k],t)
                    self.model.q[('u', 1)][eN,k] = self.analyticalSolution[1].uOfXT(self.model.q['x'][eN,k],t)
                    self.model.q[('u', 2)][eN,k] = self.analyticalSolution[2].uOfXT(self.model.q['x'][eN,k],t)
                    if self.nd == 3:
                        self.model.q[('u', 3)][eN,k] = self.analyticalSolution[3].uOfXT(self.model.q['x'][eN,k],t)

        if self.nParticles > 0 and self.use_ball_as_particle == 0 and not self.phi_s_isSet:
            self.phi_s[:] = 1e10
            self.phisField[:] = 1e10
            self.ebqe_phi_s[:] = 1e10
            self.ebq_global_grad_phi_s[:] = 1e10
            self.ebq_particle_velocity_s[:] = 1e10
            for i in range(self.nParticles):
                vel = lambda x: self.particle_velocityList[i](t, x)
                sdf = lambda x: self.particle_sdfList[i](t, x)
                for j in range(self.mesh.nodeArray.shape[0]):
                    sdf_at_node, sdNormals = sdf(self.mesh.nodeArray[j, :])
                    if (sdf_at_node < self.phi_s[j]):
                        self.phi_s[j] = sdf_at_node
                for eN in range(self.model.q['x'].shape[0]):
                    for k in range(self.model.q['x'].shape[1]):
                        self.particle_signed_distances[i, eN, k], self.particle_signed_distance_normals[i, eN, k,:] = sdf(self.model.q['x'][eN, k])
                        self.particle_velocities[i, eN, k,:] = vel(self.model.q['x'][eN, k])
                        if (self.particle_signed_distances[i, eN, k] < self.phisField[eN, k]):
                            self.phisField[eN, k] = self.particle_signed_distances[i, eN, k]
                for ebNE in range(self.model.ebqe['x'].shape[0]):
                    for kb in range(self.model.ebqe['x'].shape[1]):
                        sdf_ebNE_kb,sdNormals = sdf(self.model.ebqe['x'][ebNE,kb])
                        if (sdf_ebNE_kb < self.ebqe_phi_s[ebNE,kb]):
                            self.ebqe_phi_s[ebNE,kb]=sdf_ebNE_kb
        # if self.comm.isMaster():
        # print "wettedAreas"
        # print self.wettedAreas[:]
        # print "Forces_p"
        # print self.netForces_p[:,:]
        # print "Forces_v"
        # print self.netForces_v[:,:]

    def postStep(self, t, firstStep=False):
        self.model.dt_last = self.model.timeIntegration.dt
        self.model.q['dV_last'][:] = self.model.q['dV']
        self.model.isActiveElement_last[:] = self.model.isActiveElement
        if self.comm.isMaster():
            # logEvent("wettedAreas\n"+
            #          repr(self.wettedAreas[:]) +
            #          "\nForces_p\n" +
            #          repr(self.netForces_p[:,:]) +
            #          "\nForces_v\n" +
            #          repr(self.netForces_v[:,:]))
            self.timeHistory.write("%21.16e\n" % (t,))
            self.timeHistory.flush()
            for bN in range(self.wettedAreas.shape[0]):
                self.wettedAreaHistory.write("%21.16e\n" % (self.wettedAreas[bN],))
                self.forceHistory_p.write("%21.16e %21.16e %21.16e\n" % tuple(self.netForces_p[bN, :]))
                self.forceHistory_v.write("%21.16e %21.16e %21.16e\n" % tuple(self.netForces_v[bN, :]))
                self.momentHistory.write("%21.15e %21.16e %21.16e\n" % tuple(self.netMoments[bN, :]))
            self.wettedAreaHistory.flush()
            self.forceHistory_p.flush()
            self.forceHistory_v.flush()
            self.momentHistory.flush()
            if self.nParticles > 0:
                self.particle_forceHistory.write("%21.16e %21.16e %21.16e\n" % tuple(self.particle_netForces[0, :]))
                self.particle_forceHistory.flush()
                self.particle_vforceHistory.write("%21.16e %21.16e %21.16e\n" % tuple(self.particle_netForces[0+self.nParticles, :]))
                self.particle_vforceHistory.flush()
                self.particle_pforceHistory.write("%21.16e %21.16e %21.16e\n" % tuple(self.particle_netForces[0+2*self.nParticles, :]))
                self.particle_pforceHistory.flush()
                self.particle_momentHistory.write("%21.15e %21.16e %21.16e\n" % tuple(self.particle_netMoments[0, :]))
                self.particle_momentHistory.flush()

        if self.nParticles > 0 and self.use_ball_as_particle and self.useInternalParticleSolver:
            argsDict = cArgumentsDict.ArgumentsDict()
            argsDict["ball_FT"] = self.ball_FT
            argsDict["ball_last_FT"] = self.ball_last_FT
            argsDict["ball_h"] = self.ball_h
            argsDict["ball_last_h"] = self.ball_last_h
            argsDict["ball_center"] = self.ball_center
            argsDict["ball_center_last"] = self.ball_center_last
            argsDict["ball_velocity"] = self.ball_velocity
            argsDict["ball_angular_velocity"] = self.ball_angular_velocity
            argsDict["ball_last_velocity"] = self.ball_last_velocity
            argsDict["ball_last_angular_velocity"] = self.ball_last_angular_velocity
            argsDict["ball_Q"] = self.ball_Q
            argsDict["ball_last_Q"] = self.ball_last_Q
            argsDict["ball_Omega"] = self.ball_Omega
            argsDict["ball_last_Omega"] = self.ball_last_Omega
            argsDict["ball_u"] = self.ball_u
            argsDict["ball_last_u"] = self.ball_last_u
            argsDict["ball_mom"] = self.ball_mom
            argsDict["ball_last_mom"] = self.ball_last_mom
            argsDict["ball_a"] = self.ball_a
            argsDict["ball_I"] = self.ball_I
            argsDict["ball_mass"] = self.ball_mass
            argsDict["ball_radius"] = self.ball_radius
            argsDict["ball_f"] = self.ball_f
            argsDict["wall_f"] = self.wall_f
            argsDict["particle_netForces"] = self.particle_netForces
            argsDict["particle_netMoments"] = self.particle_netMoments
            argsDict["last_particle_netForces"] = self.last_particle_netForces
            argsDict["last_particle_netMoments"] = self.last_particle_netMoments
            argsDict["ball_angular_velocity_avg"] = self.ball_angular_velocity_avg
            argsDict["g"] = self.particle_g
            argsDict["L"] = self.L
            argsDict["dt"] = self.model.timeIntegration.dt
            argsDict["ball_force_range"] = self.ball_force_range
            argsDict["ball_stiffness"] = self.ball_stiffness
            argsDict["particle_cfl"] = self.particle_cfl
            argsDict["min_dt"] = min_dt = np.zeros((1,),'d')
            argsDict["nSteps"] = nSteps = np.zeros((1,),'i')
            if self.comm.rank() == 0:
                self.model.rans2p.step6DOF(argsDict)
            from mpi4py import MPI
            mpicomm = MPI.COMM_WORLD
            mpicomm.Bcast(self.ball_center)
            mpicomm.Bcast(self.ball_velocity)
            mpicomm.Bcast(self.ball_angular_velocity)
            logEvent("minimimum particle dt {0}".format(min_dt[0]))
            logEvent("particle sub-steps {0}".format(nSteps[0]))
            if self.comm.isMaster():
                np.savetxt(self.history_file, np.vstack((self.ball_center,self.ball_velocity, self.ball_angular_velocity)))
                self.history_file.flush()

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
                 name='RANS2P',
                 reuse_trial_and_test_quadrature=False,
                 sd=True,
                 movingDomain=False):
        if coefficients.useExact:
            self.hasCutCells=True
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
        # Simplicial Mesh
        self.mesh = self.u[0].femSpace.mesh  # assume the same mesh for  all components for now
        self.par_info = LinearAlgebraTools.ParInfo_petsc4py()
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
        self.q[('u', 0)] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
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
        #self.q[('dV_u',1)] = (1.0/self.mesh.nElements_global)*numpy.ones((self.mesh.nElements_global,self.nQuadraturePoints_element),'d')
        #self.q[('dV_u',2)] = (1.0/self.mesh.nElements_global)*numpy.ones((self.mesh.nElements_global,self.nQuadraturePoints_element),'d')
        #self.q[('dV_u',3)] = (1.0/self.mesh.nElements_global)*numpy.ones((self.mesh.nElements_global,self.nQuadraturePoints_element),'d')
        self.q['dV'] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q['dV_last'] = -1000 * numpy.ones((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('f', 0)] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element, self.nSpace_global), 'd')
        self.q[('velocity', 0)] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element, self.nSpace_global), 'd')
        self.q['velocity_solid'] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element, self.nSpace_global), 'd')
        self.q['phi_solid'] = INSIDE_FLUID_DOMAIN*numpy.ones((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q['velocity_porous'] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element, self.nSpace_global), 'd')
        self.q['phi_porous'] = INSIDE_FLUID_DOMAIN*numpy.ones((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q['x'] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element, 3), 'd')
        self.q[('cfl', 0)] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('numDiff', 1, 1)] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('numDiff', 2, 2)] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('numDiff', 3, 3)] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.ebqe[('u', 0)] = numpy.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('u', 1)] = numpy.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('u', 2)] = numpy.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('u', 3)] = numpy.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe['eddy_viscosity'] = numpy.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe['eddy_viscosity_last'] = numpy.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
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
        self.q['eddy_viscosity_last'] = numpy.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
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
        if numericalFluxType is not None and numericalFluxType.useWeakDirichletConditions:
            interleave_DOF=True
            for nDOF_trial_element_ci in self.nDOF_trial_element:
                if nDOF_trial_element_ci != self.nDOF_trial_element[0]:
                    interleave_DOF=False
        else:
            interleave_DOF=False
        self.setupFieldStrides(interleave_DOF)
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
        logEvent(memory("XdmfWriters", "OneLevelTransport"), level=4)
        logEvent("flux bc objects")
        try:
            for ci, fbcObject in list(self.fluxBoundaryConditionsObjectsDict.items()):
                self.ebqe[('advectiveFlux_bc_flag', ci)] = numpy.zeros(self.ebqe[('advectiveFlux_bc', ci)].shape, 'i')
                for t, g in list(fbcObject.advectiveFluxBoundaryConditionsDict.items()):
                    if ci in self.coefficients.advection:
                        self.ebqe[('advectiveFlux_bc', ci)][t[0], t[1]] = g(self.ebqe[('x')][t[0], t[1]], self.timeIntegration.t,self.ebqe['n'][t[0],t[1]])
                        self.ebqe[('advectiveFlux_bc_flag', ci)][t[0], t[1]] = 1
                for ck, diffusiveFluxBoundaryConditionsDict in list(fbcObject.diffusiveFluxBoundaryConditionsDictDict.items()):
                    self.ebqe[('diffusiveFlux_bc_flag', ck, ci)] = numpy.zeros(self.ebqe[('diffusiveFlux_bc', ck, ci)].shape, 'i')
                    for t, g in list(diffusiveFluxBoundaryConditionsDict.items()):
                        self.ebqe[('diffusiveFlux_bc', ck, ci)][t[0], t[1]] = g(self.ebqe[('x')][t[0], t[1]], self.timeIntegration.t,self.ebqe['n'][t[0],t[1]])
                        self.ebqe[('diffusiveFlux_bc_flag', ck, ci)][t[0], t[1]] = 1
        except:
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
            self.u[3].name="w"
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
                                         self.u[1].femSpace.referenceFiniteElement.localFunctionSpace.dim,
                                         self.testSpace[1].referenceFiniteElement.localFunctionSpace.dim,
                                         self.nElementBoundaryQuadraturePoints_elementBoundary,
                                         compKernelFlag)
        else:
            logEvent("calling  cRANS2P_base ctor")
            self.rans2p = cRANS2P_base(self.nSpace_global,
                                       self.nQuadraturePoints_element,
                                       self.u[0].femSpace.elementMaps.localFunctionSpace.dim,
                                       self.u[0].femSpace.referenceFiniteElement.localFunctionSpace.dim,
                                       self.testSpace[0].referenceFiniteElement.localFunctionSpace.dim,
                                       self.u[1].femSpace.referenceFiniteElement.localFunctionSpace.dim,
                                       self.testSpace[1].referenceFiniteElement.localFunctionSpace.dim,
                                       self.nElementBoundaryQuadraturePoints_elementBoundary,
                                       compKernelFlag)
        self.ball_u = self.u[1].dof.copy()
        self.ball_v = self.u[2].dof.copy()
        if self.nSpace_global == 3:
            self.ball_w = self.u[3].dof.copy()
        else:
            self.ball_w = self.u[2].dof.copy()
        self.errors = np.zeros((3,5),'d')
        self.velocityErrorNodal = self.u[0].dof.copy()
        logEvent('WARNING: The boundary fluxes at interpart boundaries are skipped if elementBoundaryMaterialType is 0 for RANS2P-based models. This means that DG methods are currently incompatible with RANS2P.')

        
        self.q[('force', 0)] = numpy.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('force', 1)] = numpy.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('force', 2)] = numpy.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        for eN in range(self.q['x'].shape[0]):
            for k in range(self.q['x'].shape[1]):
                if self.coefficients.force_x:
                    self.q[('force', 0)][eN,k] = self.coefficients.force_x(self.q['x'][eN,k])
                if self.coefficients.force_y:
                    self.q[('force', 1)][eN,k] = self.coefficients.force_y(self.q['x'][eN,k])
                if self.coefficients.force_z:
                    self.q[('force', 2)][eN,k] = self.coefficients.force_z(self.q['x'][eN,k])

    def getResidual(self, u, r):
        """
        Calculate the element residuals and add in to the global residual

        Parameters
        ----------
        u : :class:`numpy.ndarray`
        r : :class:`numpy.ndarray`
            Stores the calculated residual vector.
        """
        assert(np.all(np.isfinite(u)))
        assert(np.all(np.isfinite(r)))
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
            try:
                for ci, fbcObject in list(self.fluxBoundaryConditionsObjectsDict.items()):
                    for t, g in list(fbcObject.advectiveFluxBoundaryConditionsDict.items()):
                        if ci in self.coefficients.advection:
                            self.ebqe[('advectiveFlux_bc', ci)][t[0], t[1]] = g(self.ebqe[('x')][t[0], t[1]], self.timeIntegration.t,self.ebqe['n'][t[0],t[1]])
                            self.ebqe[('advectiveFlux_bc_flag', ci)][t[0], t[1]] = 1
                    for ck, diffusiveFluxBoundaryConditionsDict in list(fbcObject.diffusiveFluxBoundaryConditionsDictDict.items()):
                        for t, g in list(diffusiveFluxBoundaryConditionsDict.items()):
                            self.ebqe[('diffusiveFlux_bc', ck, ci)][t[0], t[1]] = g(self.ebqe[('x')][t[0], t[1]], self.timeIntegration.t,self.ebqe['n'][t[0],t[1]])
                            self.ebqe[('diffusiveFlux_bc_flag', ck, ci)][t[0], t[1]] = 1
            except:
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
        try:
            self.isActiveR[:] = 0.0
            self.isActiveDOF_p[:] = 0.0
            self.isActiveDOF_vel[:] = 0.0
            self.isActiveElement[:] = 0
        except AttributeError:
            self.isActiveR = np.zeros_like(r)
            self.isActiveDOF_p = np.zeros_like(self.u[0].dof)
            self.isActiveDOF_vel = np.zeros_like(self.u[1].dof)
            self.isActiveElement = np.zeros((self.mesh.nElements_global,),'i')
            self.isActiveElement_last = np.ones((self.mesh.nElements_global,),'i')
        self.Ct_sge = 4.0
        self.Cd_sge = 36.0
        self.coefficients.wettedAreas[:] = 0.0
        self.coefficients.netForces_p[:, :] = 0.0
        self.coefficients.netForces_v[:, :] = 0.0
        self.coefficients.netMoments[:, :] = 0.0
        self.coefficients.particle_netForces[:, :] = 0.0
        self.coefficients.particle_netMoments[:, :] = 0.0
        self.coefficients.particle_surfaceArea[:] = 0.0
        self.coefficients.particle_surfaceArea_projected[:] = 0.0
        self.coefficients.particle_volume[:] = 0.0
        if self.forceStrongConditions:
            try:
                for cj in range(len(self.dirichletConditionsForceDOF)):
                    for dofN, g in list(self.dirichletConditionsForceDOF[cj].DOFBoundaryConditionsDict.items()):
                        if cj == 0:
                            self.u[cj].dof[dofN] = g(self.dirichletConditionsForceDOF[cj].DOFBoundaryPointDict[dofN], self.timeIntegration.t,n=np.zeros((self.nSpace_global,),'d'))
                        else:
                            self.u[cj].dof[dofN] = g(self.dirichletConditionsForceDOF[cj].DOFBoundaryPointDict[dofN],
                                                     self.timeIntegration.t,n=np.zeros((self.nSpace_global,),'d'))
                            if self.MOVING_DOMAIN == 1.0:
                                self.u[cj].dof[dofN] += self.mesh.nodeVelocityArray[dofN, cj - 1]
            except:
                for cj in range(len(self.dirichletConditionsForceDOF)):
                    for dofN, g in list(self.dirichletConditionsForceDOF[cj].DOFBoundaryConditionsDict.items()):
                        if cj == 0:
                            self.u[cj].dof[dofN] = g(self.dirichletConditionsForceDOF[cj].DOFBoundaryPointDict[dofN], self.timeIntegration.t)
                        else:
                            self.u[cj].dof[dofN] = g(self.dirichletConditionsForceDOF[cj].DOFBoundaryPointDict[dofN],
                                                     self.timeIntegration.t)
                            if self.MOVING_DOMAIN == 1.0:
                                self.u[cj].dof[dofN] += self.mesh.nodeVelocityArray[dofN, cj - 1]
        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["NONCONSERVATIVE_FORM"] = float(self.coefficients.NONCONSERVATIVE_FORM)
        argsDict["MOMENTUM_SGE"] = float(self.coefficients.MOMENTUM_SGE)
        argsDict["PRESSURE_SGE"] = float(self.coefficients.PRESSURE_SGE)
        argsDict["VELOCITY_SGE"] = float(self.coefficients.VELOCITY_SGE)
        argsDict["PRESSURE_PROJECTION_STABILIZATION"] = self.coefficients.PRESSURE_PROJECTION_STABILIZATION
        argsDict["numerical_viscosity"] = self.coefficients.numerical_viscosity
        argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["mesh_velocity_dof"] = self.mesh.nodeVelocityArray
        argsDict["MOVING_DOMAIN"] = self.MOVING_DOMAIN
        argsDict["mesh_l2g"] = self.mesh.elementNodesArray
        argsDict["x_ref"] = self.elementQuadraturePoints
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
        argsDict["xb_ref"] = self.elementBoundaryQuadraturePoints
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
        argsDict["elementBoundaryDiameter"] = self.mesh.elementBoundaryDiametersArray
        argsDict["nodeDiametersArray"] = self.mesh.nodeDiametersArray
        argsDict["hFactor"] = self.stabilization.hFactor
        argsDict["nElements_global"] = self.mesh.nElements_global
        argsDict["nElementBoundaries_owned"] = int(self.mesh.nElementBoundaries_owned)
        argsDict["useRBLES"] = float(self.coefficients.useRBLES)
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
        argsDict["phi_solid"] = self.coefficients.q_phi_solid
        argsDict["q_velocity_solid"] = self.coefficients.q_velocity_solid
        argsDict["eps_porous"] = self.coefficients.epsFact_porous
        argsDict["phi_porous"] = self.coefficients.q_phi_porous
        argsDict["q_velocity_porous"] = self.coefficients.q_velocity_porous
        argsDict["q_porosity"] = self.coefficients.q_porosity
        argsDict["q_dragAlpha"] = self.coefficients.q_dragAlpha
        argsDict["q_dragBeta"] = self.coefficients.q_dragBeta
        argsDict["q_mass_source"] = self.q[('r', 0)]
        argsDict["q_turb_var_0"] = self.coefficients.q_turb_var[0]
        argsDict["q_turb_var_1"] = self.coefficients.q_turb_var[1]
        argsDict["q_turb_var_grad_0"] = self.coefficients.q_turb_var_grad[0]
        argsDict["LAG_LES"] = self.coefficients.LAG_LES
        argsDict["q_eddy_viscosity"] = self.q['eddy_viscosity']
        argsDict["q_eddy_viscosity_last"] = self.q['eddy_viscosity_last']
        argsDict["ebqe_eddy_viscosity"] = self.ebqe['eddy_viscosity']
        argsDict["ebqe_eddy_viscosity_last"] = self.ebqe['eddy_viscosity_last']
        argsDict["p_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["vel_l2g"] = self.u[1].femSpace.dofMap.l2g
        argsDict["rp_l2g"] = self.l2g[0]['freeGlobal']
        argsDict["rvel_l2g"] = self.l2g[1]['freeGlobal']
        argsDict["p_dof"] = self.u[0].dof
        argsDict["u_dof"] = self.u[1].dof
        argsDict["v_dof"] = self.u[2].dof
        argsDict["w_dof"] = self.u[3].dof
        argsDict["p_old_dof"] = self.coefficients.p_old_dof
        argsDict["u_old_dof"] = self.coefficients.u_old_dof
        argsDict["v_old_dof"] = self.coefficients.v_old_dof
        argsDict["w_old_dof"] = self.coefficients.w_old_dof
        argsDict["g"] = self.coefficients.g
        argsDict["useVF"] = self.coefficients.useVF
        argsDict["q_rho"] = self.q['rho']
        argsDict["vf"] = self.coefficients.q_vf
        argsDict["phi"] = self.coefficients.q_phi
        argsDict["phi_nodes"] = self.coefficients.phi_dof
        argsDict["normal_phi"] = self.coefficients.q_n
        argsDict["kappa_phi"] = self.coefficients.q_kappa
        argsDict["q_mom_u_acc"] = self.timeIntegration.m_tmp[1]
        argsDict["q_mom_v_acc"] = self.timeIntegration.m_tmp[2]
        argsDict["q_mom_w_acc"] = self.timeIntegration.m_tmp[3]
        argsDict["q_mass_adv"] = self.q[('f', 0)]
        argsDict["q_mom_u_acc_beta_bdf"] = self.timeIntegration.beta_bdf[1]
        argsDict["q_mom_v_acc_beta_bdf"] = self.timeIntegration.beta_bdf[2]
        argsDict["q_mom_w_acc_beta_bdf"] = self.timeIntegration.beta_bdf[3]
        argsDict["q_dV"] = self.q['dV']
        argsDict["q_dV_last"] = self.q['dV_last']
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
        argsDict["elementBoundariesArray"] = self.mesh.elementBoundariesArray
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
        argsDict["q_u_0"] = self.q[('u',0)]
        argsDict["q_u_1"] = self.q[('u',1)]
        argsDict["q_u_2"] = self.q[('u',2)]
        argsDict["q_u_3"] = self.q[('u',3)]
        argsDict["q_velocity"] = self.q[('velocity', 0)]
        argsDict["ebqe_velocity"] = self.ebqe[('velocity', 0)]
        argsDict["flux"] = self.ebq_global[('totalFlux', 0)]
        argsDict["elementResidual_p_save"] = self.elementResidual[0]
        argsDict["elementFlags"] = self.mesh.elementMaterialTypes
        argsDict["boundaryFlags"] = self.mesh.elementBoundaryUserFlags
        argsDict["barycenters"] = self.coefficients.barycenters
        argsDict["wettedAreas"] = self.coefficients.wettedAreas
        argsDict["netForces_p"] = self.coefficients.netForces_p
        argsDict["netForces_v"] = self.coefficients.netForces_v
        argsDict["netMoments"] = self.coefficients.netMoments
        argsDict["velocityError"] = self.q['velocityError']
        argsDict["velocityErrorNodal"] = self.velocityErrorNodal
        argsDict["forcex"] = self.q[('force', 0)]
        argsDict["forcey"] = self.q[('force', 1)]
        argsDict["forcez"] = self.q[('force', 2)]
        argsDict["use_ball_as_particle"] = int(self.coefficients.use_ball_as_particle)
        argsDict["ball_center"] = self.coefficients.ball_center
        argsDict["ball_radius"] = self.coefficients.ball_radius
        argsDict["ball_velocity"] = self.coefficients.ball_velocity
        argsDict["ball_angular_velocity"] = self.coefficients.ball_angular_velocity
        argsDict["ball_center_acceleration"] = self.coefficients.ball_center_acceleration
        argsDict["ball_angular_acceleration"] = self.coefficients.ball_angular_acceleration
        argsDict["ball_density"] = self.coefficients.ball_density
        argsDict["particle_signed_distances"] = self.coefficients.particle_signed_distances
        argsDict["particle_signed_distance_normals"] = self.coefficients.particle_signed_distance_normals
        argsDict["particle_velocities"] = self.coefficients.particle_velocities
        argsDict["particle_centroids"] = self.coefficients.particle_centroids
        argsDict["ebqe_phi_s"] = self.coefficients.ebqe_phi_s
        argsDict["ebq_global_grad_phi_s"] = self.coefficients.ebq_global_grad_phi_s
        argsDict["ebq_particle_velocity_s"] = self.coefficients.ebq_particle_velocity_s
        argsDict["nParticles"] = self.coefficients.nParticles
        argsDict["particle_netForces"] = self.coefficients.particle_netForces
        argsDict["particle_netMoments"] = self.coefficients.particle_netMoments
        argsDict["particle_surfaceArea"] = self.coefficients.particle_surfaceArea
        argsDict["particle_surfaceArea_projected"] = self.coefficients.particle_surfaceArea_projected
        argsDict["projection_direction"] = self.coefficients.projection_direction
        argsDict["particle_volume"] = self.coefficients.particle_volume
        argsDict["nElements_owned"] = int(self.mesh.nElements_owned)
        argsDict["particle_nitsche"] = self.coefficients.particle_nitsche
        argsDict["particle_epsFact"] = self.coefficients.particle_epsFact
        argsDict["particle_alpha"] = self.coefficients.particle_alpha
        argsDict["particle_beta"] = self.coefficients.particle_beta
        argsDict["particle_penalty_constant"] = self.coefficients.particle_penalty_constant
        argsDict["ghost_penalty_constant"] = self.coefficients.ghost_penalty_constant
        argsDict["phi_solid_nodes"] = self.coefficients.phi_s
        argsDict["distance_to_solids"] = self.coefficients.phisField
        argsDict["useExact"] = int(self.coefficients.useExact)
        argsDict["isActiveR"] = self.isActiveR
        argsDict["isActiveDOF_p"] = self.isActiveDOF_p
        argsDict["isActiveDOF_vel"] = self.isActiveDOF_vel
        argsDict["isActiveElement"] = self.isActiveElement
        argsDict["isActiveElement_last"] = self.isActiveElement_last
        argsDict["normalize_pressure"] = int(self.coefficients.normalize_pressure)
        argsDict["errors"]=self.errors
        argsDict["ball_u"]= self.ball_u
        argsDict["ball_v"]= self.ball_v
        argsDict["ball_w"]= self.ball_w
        self.rans2p.calculateResidual(argsDict)
        if self.forceStrongConditions:
            try:
                for cj in range(len(self.dirichletConditionsForceDOF)):
                    for dofN, g in list(self.dirichletConditionsForceDOF[cj].DOFBoundaryConditionsDict.items()):
                        if cj == 0:
                            r[self.offset[cj] + self.stride[cj] * dofN] = self.u[cj].dof[dofN] - \
                                g(self.dirichletConditionsForceDOF[cj].DOFBoundaryPointDict[dofN], self.timeIntegration.t, n=np.zeros((self.nSpace_global,),'d'))
                        else:
                            r[self.offset[cj] + self.stride[cj] * dofN] = self.u[cj].dof[dofN] - \
                                g(self.dirichletConditionsForceDOF[cj].DOFBoundaryPointDict[dofN], self.timeIntegration.t, n=np.zeros((self.nSpace_global,),'d')) 
                            if self.MOVING_DOMAIN == 1.0:
                                r[self.offset[cj] + self.stride[cj] * dofN] -= self.mesh.nodeVelocityArray[dofN, cj - 1]
            except:
                for cj in range(len(self.dirichletConditionsForceDOF)):
                    for dofN, g in list(self.dirichletConditionsForceDOF[cj].DOFBoundaryConditionsDict.items()):
                        if cj == 0:
                            r[self.offset[cj] + self.stride[cj] * dofN] = self.u[cj].dof[dofN] - \
                                g(self.dirichletConditionsForceDOF[cj].DOFBoundaryPointDict[dofN], self.timeIntegration.t)
                        else:
                            r[self.offset[cj] + self.stride[cj] * dofN] = self.u[cj].dof[dofN] - \
                                g(self.dirichletConditionsForceDOF[cj].DOFBoundaryPointDict[dofN], self.timeIntegration.t) 
                            if self.MOVING_DOMAIN == 1.0:
                                r[self.offset[cj] + self.stride[cj] * dofN] -= self.mesh.nodeVelocityArray[dofN, cj - 1]
                        #assert(abs(r[self.offset[cj] + self.stride[cj] * dofN]) < 1.0e-8)
                        
        #cek hack to fix singular system due pressure
        # comm = Comm.get()
        # if comm.rank() == 0:
        #     for i in range(self.u[0].dof.shape[0]):
        #         if self.isActiveR[i] != 1.0:
        #             self.isActiveR[i] == 0.0
        #             break
        try:
            #is sensitive to inactive DOF at velocity due to time derivative
            if self.coefficients.use_ball_as_particle:
                self.u[0].dof[:] = np.where(self.isActiveDOF_p==1.0, self.u[0].dof,0.0)
                self.u[1].dof[:] = np.where(self.isActiveDOF_vel==1.0, self.u[1].dof,self.ball_u)
                self.u[2].dof[:] = np.where(self.isActiveDOF_vel==1.0, self.u[2].dof,self.ball_v)
                if self.nSpace_global == 3:
                    self.u[3].dof[:] = np.where(self.isActiveDOF_vel==1.0, self.u[3].dof,self.ball_w)
            else:
                self.u[0].dof[:] = np.where(self.isActiveDOF_p==1.0, self.u[0].dof,0.0)
                self.u[1].dof[:] = np.where(self.isActiveDOF_vel==1.0, self.u[1].dof,0.0)
                self.u[2].dof[:] = np.where(self.isActiveDOF_vel==1.0, self.u[2].dof,0.0)
                if self.nSpace_global == 3:
                    self.u[3].dof[:] = np.where(self.isActiveDOF_vel==1.0, self.u[3].dof,0.0)
            r*=self.isActiveR
        except:
            assert((self.isActiveR == 1.0).all())
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        
        comm.Allreduce(self.coefficients.wettedAreas.copy(),self.coefficients.wettedAreas)
        comm.Allreduce(self.coefficients.netForces_p.copy(),self.coefficients.netForces_p)
        comm.Allreduce(self.coefficients.netForces_v.copy(),self.coefficients.netForces_v)
        comm.Allreduce(self.coefficients.netMoments.copy(),self.coefficients.netMoments)
        
        comm.Allreduce(self.coefficients.particle_netForces.copy(),self.coefficients.particle_netForces)
        comm.Allreduce(self.coefficients.particle_netMoments.copy(),self.coefficients.particle_netMoments)
        comm.Allreduce(self.coefficients.particle_surfaceArea.copy(),self.coefficients.particle_surfaceArea)
        comm.Allreduce(self.coefficients.particle_surfaceArea_projected.copy(),self.coefficients.particle_surfaceArea_projected)
        comm.Allreduce(self.coefficients.particle_volume.copy(),self.coefficients.particle_volume)

        if (self.coefficients.nParticles < 10):
            for i in range(self.coefficients.nParticles):
                logEvent("particle i=" + repr(i)+ " force " + repr(self.coefficients.particle_netForces[i]))
                logEvent("particle i=" + repr(i)+ " moment " + repr(self.coefficients.particle_netMoments[i]))
                logEvent("particle i=" + repr(i)+ " surfaceArea " + repr(self.coefficients.particle_surfaceArea[i]))
                logEvent("particle i=" + repr(i)+ " projected surfaceArea " + repr(self.coefficients.particle_surfaceArea_projected[i]))
                logEvent("particle i=" + repr(i)+ " particle volume" + repr(self.coefficients.particle_volume[i]))
                logEvent("particle i=" + repr(i)+ " stress force " + repr(self.coefficients.particle_netForces[i+self.coefficients.nParticles]))
                logEvent("particle i=" + repr(i)+ " pressure force " + repr(self.coefficients.particle_netForces[i+2*self.coefficients.nParticles]))

        cflMax = globalMax(self.q[('cfl', 0)].max()) * self.timeIntegration.dt
        logEvent("Maximum CFL = " + str(cflMax), level=2)
        if self.stabilization:
            self.stabilization.accumulateSubgridMassHistory(self.q)
        logEvent("Global residual", level=9, data=r)
        self.nonlinear_function_evaluations += 1
        if self.coefficients.analyticalSolution is not None:
            logEvent("""
p_1.append({:21.16e})
u_1.append({:21.16e})
v_1.append({:21.16e})
w_1.append({:21.16e})
velocity_1.append({:21.16e})
p_2.append({:21.16e})
u_2.append({:21.16e})
v_2.append({:21.16e})
w_2.append({:21.16e})
velocity_2.append({:21.16e})
p_I.append({:21.16e})
u_I.append({:21.16e})
v_I.append({:21.16e})
w_I.append({:21.16e})
velocity_I.append({:21.16e})
""".format(*self.errors.flatten().tolist()))
    
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
        argsDict["NONCONSERVATIVE_FORM"] = float(self.coefficients.NONCONSERVATIVE_FORM)
        argsDict["MOMENTUM_SGE"] = float(self.coefficients.MOMENTUM_SGE)
        argsDict["PRESSURE_SGE"] = float(self.coefficients.PRESSURE_SGE)
        argsDict["VELOCITY_SGE"] = float(self.coefficients.VELOCITY_SGE)
        argsDict["PRESSURE_PROJECTION_STABILIZATION"] = self.coefficients.PRESSURE_PROJECTION_STABILIZATION
        argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["mesh_velocity_dof"] = self.mesh.nodeVelocityArray
        argsDict["MOVING_DOMAIN"] = self.MOVING_DOMAIN
        argsDict["mesh_l2g"] = self.mesh.elementNodesArray
        argsDict["x_ref"] = self.elementQuadraturePoints
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
        argsDict["xb_ref"] = self.elementBoundaryQuadraturePoints
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
        argsDict["elementBoundaryDiameter"] = self.mesh.elementBoundaryDiametersArray
        argsDict["nodeDiametersArray"] = self.mesh.nodeDiametersArray
        argsDict["hFactor"] = self.stabilization.hFactor
        argsDict["nElements_global"] = self.mesh.nElements_global
        argsDict["useRBLES"] = float(self.coefficients.useRBLES)
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
        argsDict["phi_solid"] = self.coefficients.q_phi_solid
        argsDict["q_velocity_solid"] = self.coefficients.q_velocity_solid
        argsDict["eps_porous"] = self.coefficients.epsFact_porous
        argsDict["phi_porous"] = self.coefficients.q_phi_porous
        argsDict["q_velocity_porous"] = self.coefficients.q_velocity_porous
        argsDict["q_porosity"] = self.coefficients.q_porosity
        argsDict["q_dragAlpha"] = self.coefficients.q_dragAlpha
        argsDict["q_dragBeta"] = self.coefficients.q_dragBeta
        argsDict["q_mass_source"] = self.q[('r', 0)]
        argsDict["q_turb_var_0"] = self.coefficients.q_turb_var[0]
        argsDict["q_turb_var_1"] = self.coefficients.q_turb_var[1]
        argsDict["q_turb_var_grad_0"] = self.coefficients.q_turb_var_grad[0]
        argsDict["LAG_LES"] = self.coefficients.LAG_LES
        argsDict["q_eddy_viscosity_last"] = self.q['eddy_viscosity_last']
        argsDict["ebqe_eddy_viscosity_last"] = self.ebqe['eddy_viscosity_last']
        argsDict["p_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["vel_l2g"] = self.u[1].femSpace.dofMap.l2g
        argsDict["p_dof"] = self.u[0].dof
        argsDict["u_dof"] = self.u[1].dof
        argsDict["v_dof"] = self.u[2].dof
        argsDict["w_dof"] = self.u[3].dof
        argsDict["p_old_dof"] = self.coefficients.p_old_dof
        argsDict["u_old_dof"] = self.coefficients.u_old_dof
        argsDict["v_old_dof"] = self.coefficients.v_old_dof
        argsDict["w_old_dof"] = self.coefficients.w_old_dof
        argsDict["g"] = self.coefficients.g
        argsDict["useVF"] = self.coefficients.useVF
        argsDict["vf"] = self.coefficients.q_vf
        argsDict["phi"] = self.coefficients.q_phi
        argsDict["phi_nodes"] = self.coefficients.phi_dof
        argsDict["normal_phi"] = self.coefficients.q_n
        argsDict["kappa_phi"] = self.coefficients.q_kappa
        argsDict["q_mom_u_acc_beta_bdf"] = self.timeIntegration.beta_bdf[1]
        argsDict["q_mom_v_acc_beta_bdf"] = self.timeIntegration.beta_bdf[2]
        argsDict["q_mom_w_acc_beta_bdf"] = self.timeIntegration.beta_bdf[3]
        argsDict["q_dV"] = self.q['dV']
        argsDict["q_dV_last"] = self.q['dV_last']
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
        argsDict["elementBoundariesArray"] = self.mesh.elementBoundariesArray
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
        argsDict["elementFlags"] = self.mesh.elementMaterialTypes
        argsDict["boundaryFlags"] = self.mesh.elementBoundaryUserFlags
        argsDict["use_ball_as_particle"] = int(self.coefficients.use_ball_as_particle)
        argsDict["ball_center"] = self.coefficients.ball_center
        argsDict["ball_radius"] = self.coefficients.ball_radius
        argsDict["ball_velocity"] = self.coefficients.ball_velocity
        argsDict["ball_angular_velocity"] = self.coefficients.ball_angular_velocity
        argsDict["ball_center_acceleration"] = self.coefficients.ball_center_acceleration
        argsDict["ball_angular_acceleration"] = self.coefficients.ball_angular_acceleration
        argsDict["ball_density"] = self.coefficients.ball_density
        argsDict["particle_signed_distances"] = self.coefficients.particle_signed_distances
        argsDict["particle_signed_distance_normals"] = self.coefficients.particle_signed_distance_normals
        argsDict["particle_velocities"] = self.coefficients.particle_velocities
        argsDict["particle_centroids"] = self.coefficients.particle_centroids
        argsDict["ebqe_phi_s"] = self.coefficients.ebqe_phi_s
        argsDict["ebq_global_grad_phi_s"] = self.coefficients.ebq_global_grad_phi_s
        argsDict["ebq_particle_velocity_s"] = self.coefficients.ebq_particle_velocity_s
        argsDict["phi_solid_nodes"] = self.coefficients.phi_s
        argsDict["distance_to_solids"] = self.coefficients.phisField
        argsDict["nParticles"] = self.coefficients.nParticles
        argsDict["nElements_owned"] = int(self.mesh.nElements_owned)
        argsDict["particle_nitsche"] = self.coefficients.particle_nitsche
        argsDict["particle_epsFact"] = self.coefficients.particle_epsFact
        argsDict["particle_alpha"] = self.coefficients.particle_alpha
        argsDict["particle_beta"] = self.coefficients.particle_beta
        argsDict["particle_penalty_constant"] = self.coefficients.particle_penalty_constant
        argsDict["ghost_penalty_constant"] = self.coefficients.ghost_penalty_constant
        argsDict["useExact"] = int(self.coefficients.useExact)
        argsDict["isActiveR"] = self.isActiveR
        argsDict["isActiveDOF_p"] = self.isActiveDOF_p
        argsDict["isActiveDOF_vel"] = self.isActiveDOF_vel
        argsDict["isActiveElement"] = self.isActiveElement
        argsDict["isActiveElement_last"] = self.isActiveElement_last
        self.rans2p.calculateJacobian(argsDict)
        assert(np.all(np.isfinite(jacobian.getCSRrepresentation()[2])))
        if not self.forceStrongConditions and max(numpy.linalg.norm(self.u[1].dof, numpy.inf), numpy.linalg.norm(self.u[2].dof, numpy.inf), numpy.linalg.norm(self.u[3].dof, numpy.inf)) < 1.0e-8:
            self.pp_hasConstantNullSpace = True
        else:
            self.pp_hasConstantNullSpace = False
        # Load the Dirichlet conditions directly into residual
        if self.forceStrongConditions:
            for cj in range(self.nc):
                for dofN in list(self.dirichletConditionsForceDOF[cj].DOFBoundaryConditionsDict.keys()):
                    global_dofN = self.offset[cj] + self.stride[cj] * dofN
                    for i in range(self.rowptr[global_dofN], self.rowptr[global_dofN + 1]):
                        if (self.colind[i] == global_dofN):
                            self.nzval[i] = 1.0
                        else:
                            self.nzval[i] = 0.0
                            # print "RBLES zeroing residual cj = %s dofN= %s global_dofN= %s " % (cj,dofN,global_dofN)
        #cek hack to fix singular system due pressure
        # comm = Comm.get()
        # if comm.rank() == 0:
        #     for i in range(self.u[0].dof.shape[0]):
        #         if self.isActiveR[i] != 1.0:
        #             self.isActiveR[i] == 0.0
        #             break
        for global_dofN_a in np.argwhere(self.isActiveR==0.0):
            #assert(False)
            global_dofN = global_dofN_a[0]
            #print("inactive ", global_dofN)
            for i in range(
                    self.rowptr[global_dofN],
                    self.rowptr[global_dofN + 1]):
                if (self.colind[i] == global_dofN):
                    self.nzval[i] = 1.0
                else:
                    self.nzval[i] = 0.0
        #check that inactive DOF have no non-zero coefficients in active rows
        # for global_dofN_a in np.argwhere(self.isActiveR==1.0):
        #     global_dofN = global_dofN_a[0]
        #     for i in range(
        #             self.rowptr[global_dofN],
        #             self.rowptr[global_dofN + 1]):
        #         if(self.isActiveR[self.colind[i]] == 0.0):
        #             #pass
        #             assert(self.nzval[i] == 0.0), ("row", global_dofN, "column", self.colind[i], "val", self.nzval[i],
        #                                            self.offset[0], self.offset[1], self.offset[2], self.offset[3],
        #                                            self.stride[0], self.stride[1], self.stride[2], self.stride[3])

                
        logEvent("Jacobian ", level=10, data=jacobian)
        self.nonlinear_function_jacobian_evaluations += 1
        return jacobian

    def calculateElementQuadrature(self, domainMoved=False):
        """
        Calculate the physical location and weights of the quadrature rules
        and the shape information at the quadrature points.

        This function should be called only when the mesh changes.
        """
        if self.postProcessing:
            self.u[0].femSpace.elementMaps.getValues(self.elementQuadraturePoints,
                                                     self.q['x'])
            self.u[0].femSpace.elementMaps.getJacobianValues(self.elementQuadraturePoints,
                                                             self.q['J'],
                                                             self.q['inverse(J)'],
                                                             self.q['det(J)'])
            self.u[0].femSpace.getBasisValues(self.elementQuadraturePoints, self.q[('v', 0)])
        self.u[0].femSpace.elementMaps.getBasisValuesRef(self.elementQuadraturePoints)
        self.u[0].femSpace.elementMaps.getBasisGradientValuesRef(self.elementQuadraturePoints)
        self.u[0].femSpace.getBasisValuesRef(self.elementQuadraturePoints)
        self.u[0].femSpace.getBasisGradientValuesRef(self.elementQuadraturePoints)
        self.u[1].femSpace.getBasisValuesRef(self.elementQuadraturePoints)
        self.u[1].femSpace.getBasisGradientValuesRef(self.elementQuadraturePoints)
        self.coefficients.initializeElementQuadrature(self.timeIntegration.t, self.q)
        if self.stabilization is not None and not domainMoved:
            self.stabilization.initializeElementQuadrature(self.mesh, self.timeIntegration.t, self.q)
            self.stabilization.initializeTimeIntegration(self.timeIntegration)
        if self.shockCapturing is not None and not domainMoved:
            self.shockCapturing.initializeElementQuadrature(self.mesh, self.timeIntegration.t, self.q)

    def calculateElementBoundaryQuadrature(self, domainMoved=False):
        """
        Calculate the physical location and weights of the quadrature rules
        and the shape information at the quadrature points on element boundaries.

        This function should be called only when the mesh changes.
        """
        if self.postProcessing:
            self.u[0].femSpace.elementMaps.getValuesTrace(self.elementBoundaryQuadraturePoints,
                                                          self.ebq['x'])
            self.u[0].femSpace.elementMaps.getJacobianValuesTrace(self.elementBoundaryQuadraturePoints,
                                                                  self.ebq['inverse(J)'],
                                                                  self.ebq['g'],
                                                                  self.ebq['sqrt(det(g))'],
                                                                  self.ebq['n'])
            cfemIntegrals.copyLeftElementBoundaryInfo(self.mesh.elementBoundaryElementsArray,
                                                      self.mesh.elementBoundaryLocalElementBoundariesArray,
                                                      self.mesh.exteriorElementBoundariesArray,
                                                      self.mesh.interiorElementBoundariesArray,
                                                      self.ebq['x'],
                                                      self.ebq['n'],
                                                      self.ebq_global['x'],
                                                      self.ebq_global['n'])
            self.u[0].femSpace.elementMaps.getInverseValuesTrace(self.ebq['inverse(J)'], self.ebq['x'], self.ebq['hat(x)'])
            self.u[0].femSpace.elementMaps.getPermutations(self.ebq['hat(x)'])
            self.testSpace[0].getBasisValuesTrace(self.u[0].femSpace.elementMaps.permutations,
                                                  self.ebq['hat(x)'],
                                                  self.ebq[('w', 0)])
            self.u[0].femSpace.getBasisValuesTrace(self.u[0].femSpace.elementMaps.permutations,
                                                   self.ebq['hat(x)'],
                                                   self.ebq[('v', 0)])
            cfemIntegrals.calculateElementBoundaryIntegrationWeights(self.ebq['sqrt(det(g))'],
                                                                     self.elementBoundaryQuadratureWeights[('u', 0)],
                                                                     self.ebq[('dS_u', 0)])

    def calculateExteriorElementBoundaryQuadrature(self, domainMoved=False):
        """
        Calculate the physical location and weights of the quadrature rules
        and the shape information at the quadrature points on global element boundaries.

        This function should be called only when the mesh changes.
        """
        logEvent("initalizing ebqe vectors for post-procesing velocity")
        if self.postProcessing:
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
        #
        # get physical locations of element boundary quadrature points
        #
        # assume all components live on the same mesh
        logEvent("initalizing basis info")
        self.u[0].femSpace.elementMaps.getBasisValuesTraceRef(self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.elementMaps.getBasisGradientValuesTraceRef(self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.getBasisValuesTraceRef(self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.getBasisGradientValuesTraceRef(self.elementBoundaryQuadraturePoints)
        self.u[1].femSpace.getBasisValuesTraceRef(self.elementBoundaryQuadraturePoints)
        self.u[1].femSpace.getBasisGradientValuesTraceRef(self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.elementMaps.getValuesGlobalExteriorTrace(self.elementBoundaryQuadraturePoints,
                                                                    self.ebqe['x'])
        logEvent("setting flux boundary conditions")
        if not domainMoved:
            self.fluxBoundaryConditionsObjectsDict = dict([(cj, FluxBoundaryConditions(self.mesh,
                                                                                       self.nElementBoundaryQuadraturePoints_elementBoundary,
                                                                                       self.ebqe[('x')],
                                                                                       self.advectiveFluxBoundaryConditionsSetterDict[cj],
                                                                                       self.diffusiveFluxBoundaryConditionsSetterDictDict[cj]))
                                                           for cj in list(self.advectiveFluxBoundaryConditionsSetterDict.keys())])
            logEvent("initializing coefficients ebqe")
            self.coefficients.initializeGlobalExteriorElementBoundaryQuadrature(self.timeIntegration.t, self.ebqe)
        logEvent("done with ebqe")

    def estimate_mt(self):
        pass

    def calculateSolutionAtQuadrature(self):
        pass

    def calculateAuxiliaryQuantitiesAfterStep(self):
        if self.postProcessing and self.conservativeFlux:
            if self.coefficients.porosityTypes is None:
                self.coefficients.porosityTypes = np.ones((self.mesh.elementMaterialTypes.max()+1,),'d')
            
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
            argsDict["vel_l2g"] = self.u[1].femSpace.dofMap.l2g
            argsDict["u_dof"] = self.u[1].dof
            argsDict["v_dof"] = self.u[2].dof
            argsDict["w_dof"] = self.u[3].dof
            argsDict["vel_trial_trace_ref"] = self.u[1].femSpace.psi_trace
            argsDict["ebqe_velocity"] = self.ebqe[('velocity', 0)]
            argsDict["velocityAverage"] = self.ebq_global[('velocityAverage', 0)]
            argsDict["elementMaterialTypes"] = self.mesh.elementMaterialTypes
            argsDict["porosityTypes"] = self.coefficients.porosityTypes
            self.rans2p.calculateVelocityAverage(argsDict)
            
            if self.movingDomain:
                logEvent("Element Quadrature", level=3)
                self.calculateElementQuadrature(domainMoved=True)
                logEvent("Element Boundary Quadrature", level=3)
                self.calculateElementBoundaryQuadrature(domainMoved=True)
                logEvent("Global Exterior Element Boundary Quadrature", level=3)
                self.calculateExteriorElementBoundaryQuadrature(domainMoved=True)
                for ci in range(len(self.velocityPostProcessor.vpp_algorithms)):
                    for cj in list(self.velocityPostProcessor.vpp_algorithms[ci].updateConservationJacobian.keys()):
                        self.velocityPostProcessor.vpp_algorithms[ci].updateWeights()
                        self.velocityPostProcessor.vpp_algorithms[ci].computeGeometricInfo()
                        self.velocityPostProcessor.vpp_algorithms[ci].updateConservationJacobian[cj] = True
        self.q['velocityError'][:] = self.q[('velocity', 0)]
        OneLevelTransport.calculateAuxiliaryQuantitiesAfterStep(self)
        # if  self.coefficients.nd ==3:
        #     self.q[('cfl',0)][:] = np.sqrt(self.q[('velocity',0)][...,0]*self.q[('velocity',0)][...,0] +
        #                                    self.q[('velocity',0)][...,1]*self.q[('velocity',0)][...,1] +
        #                                    self.q[('velocity',0)][...,2]*self.q[('velocity',0)][...,2])/self.elementDiameter[:,np.newaxis]
        # else:
        #     self.q[('cfl',0)][:] = np.sqrt(self.q[('velocity',0)][...,0]*self.q[('velocity',0)][...,0] +
        #                                    self.q[('velocity',0)][...,1]*self.q[('velocity',0)][...,1])/self.elementDiameter[:,np.newaxis]
        self.q['velocityError'] -= self.q[('velocity', 0)]
        self.q['eddy_viscosity_last'][:] = self.q['eddy_viscosity']
        self.ebqe['eddy_viscosity_last'][:] = self.ebqe['eddy_viscosity']
        
    def updateAfterMeshMotion(self):
        pass


def getErgunDrag(porosity, meanGrainSize, viscosity):
    # cek hack, this doesn't seem right
    # cek todo look up correct Ergun model for alpha and beta
    voidFrac = 1.0 - porosity
    if voidFrac > 1.0e-6:
        dragBeta = porosity * porosity * porosity * meanGrainSize * 1.0e-2 / voidFrac
    if (porosity > epsZero and meanGrainSize > epsZero):
        dragAlpha = viscosity * 180.0 * voidFrac * voidFrac / (meanGrainSize * meanGrainSize * porosity)
