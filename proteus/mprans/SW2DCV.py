import proteus
from proteus import FemTools
from proteus import LinearAlgebraTools as LAT
from proteus.mprans.cSW2DCV import *
import numpy as np
import h5py
from proteus.Transport import OneLevelTransport, TC_base, NonlinearEquation
from proteus.Transport import Quadrature, logEvent, memory, BackwardEuler
from proteus.Transport import FluxBoundaryConditions, Comm, DOFBoundaryConditions
from proteus.Transport import cfemIntegrals, globalMax, SparseMat
from . import cArgumentsDict

class NumericalFlux(proteus.NumericalFlux.ShallowWater_2D):
    hasInterior = False

    def __init__(self, vt, getPointwiseBoundaryConditions,
                 getAdvectiveFluxBoundaryConditions,
                 getDiffusiveFluxBoundaryConditions,
                 getPeriodicBoundaryConditions=None,
                 h_eps=1.0e-8,
                 tol_u=1.0e-8):
        proteus.NumericalFlux.ShallowWater_2D.__init__(self, vt, getPointwiseBoundaryConditions,
                                                       getAdvectiveFluxBoundaryConditions,
                                                       getDiffusiveFluxBoundaryConditions,
                                                       getPeriodicBoundaryConditions,
                                                       h_eps,
                                                       tol_u)
        self.penalty_constant = 2.0
        self.includeBoundaryAdjoint = True
        self.boundaryAdjoint_sigma = 1.0
        self.hasInterior = False


class RKEV(proteus.TimeIntegration.SSP):
    from proteus import TimeIntegration
    """
    Wrapper for SSPRK time integration using EV

    ... more to come ...
    """

    def __init__(self, transport, timeOrder=1, runCFL=0.1, integrateInterpolationPoints=False):
        BackwardEuler.__init__(
            self, transport, integrateInterpolationPoints=integrateInterpolationPoints)
        self.trasport = transport
        self.runCFL = runCFL
        self.dtLast = None
        self.dtRatioMax = 2.0
        self.isAdaptive = True
        # About the cfl
        assert hasattr(
            transport, 'edge_based_cfl'), "No edge based cfl defined"
        self.edge_based_cfl = transport.edge_based_cfl
        # Stuff particular for SSP
        self.timeOrder = timeOrder  # order of approximation
        self.nStages = timeOrder  # number of stages total
        self.lstage = 0  # last stage completed
        # storage vectors (at old time step)
        self.u_dof_last = {}
        # per component lstage values
        self.u_dof_lstage = {}
        for ci in range(self.nc):
            self.u_dof_last[ci] = transport.u[ci].dof.copy()
            self.u_dof_lstage[ci] = transport.u[ci].dof.copy()

    def choose_dt(self):
        maxCFL = 1.0e-6
        # COMPUTE edge_based_cfl
        rowptr_cMatrix, colind_cMatrix, Cx = self.transport.cterm_global[0].getCSRrepresentation()
        rowptr_cMatrix, colind_cMatrix, Cy = self.transport.cterm_global[1].getCSRrepresentation()
        rowptr_cMatrix, colind_cMatrix, CTx = self.transport.cterm_global_transpose[0].getCSRrepresentation()
        rowptr_cMatrix, colind_cMatrix, CTy = self.transport.cterm_global_transpose[1].getCSRrepresentation()
        numDOFsPerEqn = self.transport.u[0].dof.size

        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["g"] = self.transport.coefficients.g
        argsDict["numDOFsPerEqn"] = numDOFsPerEqn
        argsDict["lumped_mass_matrix"] = self.transport.ML
        argsDict["h_dof_old"] = self.transport.u[0].dof
        argsDict["hu_dof_old"] = self.transport.u[1].dof
        argsDict["hv_dof_old"] = self.transport.u[2].dof
        argsDict["b_dof"] = self.transport.coefficients.b.dof
        argsDict["csrRowIndeces_DofLoops"] = rowptr_cMatrix
        argsDict["csrColumnOffsets_DofLoops"] = colind_cMatrix
        argsDict["hEps"] = self.transport.hEps
        argsDict["Cx"] = Cx
        argsDict["Cy"] = Cy
        argsDict["CTx"] = CTx
        argsDict["CTy"] = CTy
        argsDict["dLow"] = self.transport.dLow
        argsDict["run_cfl"] = self.runCFL
        argsDict["edge_based_cfl"] = self.transport.edge_based_cfl
        if self.transport.wall_dof is None:
            self.transport.setWallDOFs()
        argsDict["wall_dof"] = self.transport.wall_dof
        argsDict["tailwater_dof"] = self.transport.tailwater_dof
        argsDict["tailwater_depth_dof"] = self.transport.tailwater_depth_dof
        argsDict["porosity_dof"] = self.transport.coefficients.porosity_dof
        argsDict["debug"] = 0
        adjusted_maxCFL = self.transport.sw2d.calculateEdgeBasedCFL(argsDict)

        # owned DOFs only: a ghost row sees a truncated stencil (and wall
        # vector), so its CFL number is not the owner's
        par_dof = getattr(self.transport.u[0], "par_dof", None)
        n_owned = par_dof.dim_proc if par_dof is not None else self.edge_based_cfl.size
        maxCFL = max(maxCFL, globalMax(self.edge_based_cfl[:n_owned].max()))
        self.dt = self.runCFL/maxCFL

        if self.dtLast is None:
            self.dtLast = self.dt
        if self.dt/self.dtLast > self.dtRatioMax:
            self.dt = self.dtLast * self.dtRatioMax
        self.t = self.tLast + self.dt
        # Ignoring dif. time step levels
        self.substeps = [self.t for i in range(self.nStages)]

        assert (self.dt > 1E-8), ("Time step is probably getting too small: ", self.dt, adjusted_maxCFL)

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
        # Not needed for an implementation when alpha and beta are not used

    def updateStage(self):
        """
        Need to switch to use coefficients
        """
        self.lstage += 1
        assert self.timeOrder in [1, 2, 3]
        assert self.lstage > 0 and self.lstage <= self.timeOrder
        # print "within update stage...: ", self.lstage
        if self.timeOrder == 3:
            if self.lstage == 1:
                for ci in range(self.nc):
                    self.u_dof_lstage[ci][:] = self.transport.u[ci].dof
                # update u_dof_old
                self.transport.h_dof_old[:] = self.u_dof_lstage[0]
                self.transport.hu_dof_old[:] = self.u_dof_lstage[1]
                self.transport.hv_dof_old[:] = self.u_dof_lstage[2]
                logEvent("First stage of SSP33 method finished", level=4)
            elif self.lstage == 2:
                for ci in range(self.nc):
                    self.u_dof_lstage[ci][:] = self.transport.u[ci].dof
                    self.u_dof_lstage[ci] *= 1./4.
                    self.u_dof_lstage[ci] += 3. / 4. * self.u_dof_last[ci]
                # update u_dof_old
                self.transport.h_dof_old[:] = self.u_dof_lstage[0]
                self.transport.hu_dof_old[:] = self.u_dof_lstage[1]
                self.transport.hv_dof_old[:] = self.u_dof_lstage[2]
                logEvent("Second stage of SSP33 method finished", level=4)
            else:
                for ci in range(self.nc):
                    self.u_dof_lstage[ci][:] = self.transport.u[ci].dof
                    self.u_dof_lstage[ci][:] *= 2.0/3.0
                    self.u_dof_lstage[ci][:] += 1.0 / 3.0 * self.u_dof_last[ci]
                    # update solution to u[0].dof
                    self.transport.u[ci].dof[:] = self.u_dof_lstage[ci]
                # update u_dof_old
                self.transport.h_dof_old[:] = self.u_dof_last[0]
                self.transport.hu_dof_old[:] = self.u_dof_last[1]
                self.transport.hv_dof_old[:] = self.u_dof_last[2]
                logEvent("Third stage of SSP33 method finished", level=4)
        elif self.timeOrder == 2:
            if self.lstage == 1:
                logEvent("First stage of SSP22 method finished", level=4)
                for ci in range(self.nc):
                    self.u_dof_lstage[ci][:] = self.transport.u[ci].dof
                # Update u_dof_old
                self.transport.h_dof_old[:] = self.u_dof_lstage[0]
                self.transport.hu_dof_old[:] = self.u_dof_lstage[1]
                self.transport.hv_dof_old[:] = self.u_dof_lstage[2]
            else:
                for ci in range(self.nc):
                    self.u_dof_lstage[ci][:] = self.transport.u[ci].dof
                    self.u_dof_lstage[ci][:] *= 1./2.
                    self.u_dof_lstage[ci][:] += 1. / 2. * self.u_dof_last[ci]
                    # update solution to u[0].dof
                    self.transport.u[ci].dof[:] = self.u_dof_lstage[ci]
                # Update u_dof_old
                self.transport.h_dof_old[:] = self.u_dof_last[0]
                self.transport.hu_dof_old[:] = self.u_dof_last[1]
                self.transport.hv_dof_old[:] = self.u_dof_last[2]
                logEvent("Second stage of SSP22 method finished", level=4)
        else:
            assert self.timeOrder == 1
            logEvent("FE method finished", level=4)

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
        self.u_dof_lstage = {}
        for ci in range(self.nc):
            self.u_dof_lstage[ci] = self.transport.u[ci].dof.copy()
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
    """
    The coefficients for the shallow water equations
    """

    def __init__(self,
                 bathymetry,
                 g=9.81,
                 nd=2,
                 sd=True,
                 movingDomain=False,
                 useRBLES=0.0,
                 useMetrics=0.0,
                 modelIndex=0,
                 cE=1.0,
                 LUMPED_MASS_MATRIX=1,
                 LINEAR_FRICTION=0,
                 mannings=0.,
                 forceStrongConditions=True,
                 constrainedDOFs=None,
                 INTEGRATE_BY_PARTS_WALLS=1,
                 porosity=None,
                 drag_alpha=None,
                 drag_beta=None,
                 wall_function=None,
                 drag_law=None,
                 BOUNDS_RELAXATION=1,
                 BOUNDS_EXCLUDE_SELF=0,
                 WALL_RIEMANN=0,
                 tailwater=None,
                 LOW_ORDER_ONLY=0,
                 NODAL_H_BOUNDS=0,
                 NO_STAR_STATES=0):
        # Canopy model (each a scalar, a nodal array, or a function of [x, y]
        # returning nodal values; None means phi = 1, alpha = beta = 0):
        #   porosity phi: storage phi h, phi hu; flux phi h u; pressure
        #     phi g h grad(h + Z). For an array of cylinders of diameter D and
        #     m per unit area, phi = 1 - m pi D^2 / 4.
        #   Darcy-Forchheimer drag -(alpha + beta |u|) h u per unit plan
        #     area; for cylinders beta = C_D m D / 2.
        self.porosity = porosity
        self.drag_alpha = drag_alpha
        self.drag_beta = drag_beta
        self.porosity_dof = None
        self.drag_alpha_dof = None
        self.drag_beta_dof = None
        # wall_function([x, y]) -> bool: which boundary nodes are no-flux
        # walls (with INTEGRATE_BY_PARTS_WALLS). The other boundary nodes
        # keep the natural boundary flux, i.e. they are open (outflow).
        # None: all boundary nodes with reflecting_BCs, else nodes flagged 99.
        self.wall_function = wall_function
        # drag_law(state) -> nodal beta: a state-dependent Forchheimer
        # coefficient, re-evaluated before every step from the nodal state
        # {x, y, h, u, v, t, age, beta0, phi}, where age is the time since the
        # node first became wet (0 before that) and beta0 the static
        # drag_beta. None: beta stays drag_beta.
        self.drag_law = drag_law
        self.t_wet = None
        # 1: relax the convex-limiting bounds by the local smoothness
        # (second order at smooth extrema; permits O(h_e^2) wiggles);
        # 0: strict bar-state bounds.
        self.BOUNDS_RELAXATION = BOUNDS_RELAXATION
        # 1: local bounds from the neighbours' bar states only (ubar_ii excluded)
        self.BOUNDS_EXCLUDE_SELF = BOUNDS_EXCLUDE_SELF
        # 1: wall rows use the Rusanov flux against the mirror state (wall_dof = 2),
        # i.e. the wall pressure carries the impedance term h u_n (u_n + lambda)
        self.WALL_RIEMANN = WALL_RIEMANN
        # tailwater = {'where': f([x, y]) -> bool, 'stage': f(x, y, t) -> surface
        # elevation (a number or an array)}: the exterior boundary nodes selected
        # by 'where' (and not walls) get the Rusanov flux against a ghost state
        # with depth max(stage - Z, 0) and the outgoing Riemann invariant
        # u_n + 2 sqrt(g h) of the interior kept, so the stage is imposed where
        # the characteristics say it matters and ignored for supercritical
        # outflow. The stage is re-evaluated before every step. None: those
        # nodes keep the natural outflow (ghost state = interior state).
        self.tailwater = tailwater
        # 1 (diagnostic): return the first-order (graph viscosity) update, no limited antidiffusion
        self.LOW_ORDER_ONLY = LOW_ORDER_ONLY
        self.NODAL_H_BOUNDS = NODAL_H_BOUNDS
        self.NO_STAR_STATES = NO_STAR_STATES
        self.wet_eps = 1.0e-4
        # 1: walls (all boundaries with reflecting_BCs, or nodes flagged 99)
        # integrate the advective terms by parts with zero advective flux
        # through the wall; the pressure reaction comes from the
        # non-conservative pressure term. 0: the older nodal projection of
        # the momentum onto the wall tangent ("reflecting").
        self.INTEGRATE_BY_PARTS_WALLS = INTEGRATE_BY_PARTS_WALLS
        self.forceStrongConditions = forceStrongConditions
        self.constrainedDOFs = constrainedDOFs
        self.bathymetry = bathymetry
        self.useRBLES = useRBLES
        self.useMetrics = useMetrics
        self.sd = sd
        self.g = g
        self.nd = nd
        self.cE = cE
        self.LUMPED_MASS_MATRIX = LUMPED_MASS_MATRIX
        self.LINEAR_FRICTION = LINEAR_FRICTION
        self.mannings = mannings
        self.modelIndex = modelIndex
        mass = {}
        advection = {}
        diffusion = {}
        potential = {}
        reaction = {}
        hamiltonian = {}
        if nd == 2:
            variableNames = ['h', 'h_u', 'h_v']
            mass = {0: {0: 'linear'},
                    1: {0: 'linear', 1: 'linear'},
                    2: {0: 'linear', 2: 'linear'}}
            advection = {0: {0: 'nonlinear',
                             1: 'nonlinear',
                             2: 'nonlinear'},
                         1: {0: 'nonlinear',
                             1: 'nonlinear',
                             2: 'nonlinear'},
                         2: {0: 'nonlinear',
                             1: 'nonlinear',
                             2: 'nonlinear'}}
            diffusion = {1: {1: {1: 'constant'}, 2: {2: 'constant'}},
                         2: {2: {2: 'constant'}, 1: {1: 'constant'}}}
            sdInfo = {(1, 1): (np.array([0, 1, 2], dtype='i'),
                               np.array([0, 1], dtype='i')),
                      (1, 2): (np.array([0, 0, 1], dtype='i'),
                               np.array([0], dtype='i')),
                      (2, 2): (np.array([0, 1, 2], dtype='i'),
                               np.array([0, 1], dtype='i')),
                      (2, 1): (np.array([0, 1, 1], dtype='i'),
                               np.array([1], dtype='i'))}
            potential = {1: {1: 'u'},
                         2: {2: 'u'}}
            reaction = {1: {0: 'linear'},
                        2: {0: 'linear'}}
            TC_base.__init__(self,
                             3, # Number of components
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames,
                             sparseDiffusionTensors=sdInfo,
                             useSparseDiffusion=sd,
                             movingDomain=movingDomain,
                             )
            self.vectorComponents = [1, 2]
            self.vectorName = 'Momentum'
            
    def attachModels(self, modelList):
        self.model = modelList[self.modelIndex]
        # pass

    def initializeMesh(self, mesh):
        x = mesh.nodeArray[:, 0]
        y = mesh.nodeArray[:, 1]
        comm=Comm.get()        
        if self.bathymetry is None:
            self.b.dof = mesh.nodeArray[:, 2].copy()     
        elif type(self.bathymetry) is np.ndarray:
            #if mpi, use mapping old2new to rearrange b.dof vector globally
            if self.bathymetry.ndim==1:
                b_global = self.bathymetry.copy() 
            elif self.bathymetry.shape[1]==1:
                b_global = self.bathymetry[:,0].copy()
            elif self.bathymetry.shape[1]==3:
                b_global = self.bathymetry[:,2].copy()
            if comm.size() > 1: 
                fname= 'mappings.h5'
                with h5py.File(fname,'r') as f:
                    vname=list(f.keys())[0]
                    old2new=np.asarray(f[vname])       
                sub2glob=mesh.nodeNumbering_subdomain2global                                   
                indx=[]
                for i in range (0, len(sub2glob)):  
                    indx.append(np.where(old2new==sub2glob[i]))
                self.b.dof = np.ravel(b_global[indx])
                mesh.nodeArray[:,2] = self.b.dof
            else:
                self.b.dof = self.bathymetry
                mesh.nodeArray[:,2] = self.b.dof           
        else:
            self.b.dof = self.bathymetry([x, y])
            mesh.nodeArray[:,2] = self.b.dof     #if bathy is a function no need to pass subdomain info here
        assert mesh.nodeArray.shape[1]==3
        def nodal(f, default):
            if f is None:
                return np.full(x.shape, default, 'd')
            if callable(f):
                return np.asarray(f([x, y]), 'd') * np.ones(x.shape)
            return np.asarray(f, 'd') * np.ones(x.shape)
        self.porosity_dof = nodal(self.porosity, 1.0)
        self.drag_alpha_dof = nodal(self.drag_alpha, 0.0)
        self.drag_beta_dof = nodal(self.drag_beta, 0.0)
        assert self.porosity_dof.min() > 0. and self.porosity_dof.max() <= 1., \
            "porosity must lie in (0, 1]"

    def initializeElementQuadrature(self, t, cq):
        pass

    def initializeElementBoundaryQuadrature(self, t, cebq, cebq_global):
        pass

    def initializeGlobalExteriorElementBoundaryQuadrature(self, t, cebqe):
        pass

    def updateToMovingDomain(self, t, c):
        pass

    def evaluate(self, t, c):
        pass

    def preStep(self, t, firstStep=False):
        if firstStep:
            # Init boundaryIndex
            assert self.model.boundaryIndex is None and self.model.normalx is not None, "Check boundaryIndex, normalx and normaly"
            self.model.boundaryIndex = []
            for i in range(self.model.normalx.size):
                if self.model.normalx[i] != 0 or self.model.normaly[i] != 0:
                    self.model.boundaryIndex.append(i)
            self.model.boundaryIndex = np.array(self.model.boundaryIndex)

            # Init reflectingBoundaryIndex for partial reflecting boundaries
            self.model.reflectingBoundaryIndex = np.where(np.isin(self.model.mesh.nodeMaterialTypes, 99))[0].tolist()
            self.model.reflectingBoundaryIndex = np.array(self.model.reflectingBoundaryIndex)
        #
        self.model.h_dof_old[:] = self.model.u[0].dof
        self.model.hu_dof_old[:] = self.model.u[1].dof
        self.model.hv_dof_old[:] = self.model.u[2].dof
        if self.drag_law is not None:
            self.updateDragLaw(t)
        if self.tailwater is not None:
            self.model.updateTailwater(t)

    def updateDragLaw(self, t):
        """re-evaluate the nodal Forchheimer coefficient from the current state"""
        h = self.model.h_dof_old
        if self.t_wet is None:
            self.t_wet = np.full(h.shape, np.nan)
            self.drag_beta0_dof = self.drag_beta_dof.copy()
        newly = np.isnan(self.t_wet) & (h > self.wet_eps)
        self.t_wet[newly] = t
        age = np.where(np.isnan(self.t_wet), 0.0, t - self.t_wet)
        hs = np.maximum(h, self.wet_eps)
        X = self.model.mesh.nodeArray
        state = {"x": X[:, 0], "y": X[:, 1], "h": h, "t": t, "age": age,
                 "u": np.where(h > self.wet_eps, self.model.hu_dof_old / hs, 0.0),
                 "v": np.where(h > self.wet_eps, self.model.hv_dof_old / hs, 0.0),
                 "beta0": self.drag_beta0_dof, "phi": self.porosity_dof}
        self.drag_beta_dof[:] = np.asarray(self.drag_law(state), 'd')

    def postStep(self, t, firstStep=False):
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
                 stressTraceBoundaryConditionsSetterDictDict=None,
                 stabilization=None,
                 shockCapturing=None,
                 conservativeFluxDict=None,
                 numericalFluxType=None,
                 TimeIntegrationClass=None,
                 massLumping=False,
                 reactionLumping=False,
                 options=None,
                 name='SW2DCV',
                 reuse_trial_and_test_quadrature=True,
                 sd=True,
                 movingDomain=False,
                 bdyNullSpace=False):
        self.bdyNullSpace = bdyNullSpace
        self.inf_norm_hu = []  # To test 1D well balancing
        self.secondCallCalculateResidual = 0
        self.postProcessing = False  # this is a hack to test the effect of post-processing
        #
        # set the objects describing the method and boundary conditions
        #
        self.movingDomain = movingDomain
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
        # assume the same mesh for  all components for now
        self.mesh = self.u[0].femSpace.mesh
        self.testSpace = testSpaceDict
        self.dirichletConditions = dofBoundaryConditionsDict
        # explicit Dirichlet conditions for now, no Dirichlet BC constraints
        self.dirichletNodeSetList = None
        self.coefficients = coefficients
        # cek hack? give coefficients a bathymetry array
        import copy
        self.coefficients.b = self.u[0].copy()
        self.coefficients.b.name = 'b'
        self.coefficients.b.dof.fill(0.0)
        #
        self.coefficients.initializeMesh(self.mesh)
        self.nc = self.coefficients.nc
        for ci in range(self.nc):
            self.u[ci].femSpace.updateInterpolationPoints()
        self.stabilization = stabilization
        self.shockCapturing = shockCapturing
        # no velocity post-processing for now
        self.conservativeFlux = conservativeFluxDict
        self.fluxBoundaryConditions = fluxBoundaryConditionsDict
        self.advectiveFluxBoundaryConditionsSetterDict = advectiveFluxBoundaryConditionsSetterDict
        self.diffusiveFluxBoundaryConditionsSetterDictDict = diffusiveFluxBoundaryConditionsSetterDictDict
        # determine if we need element boundary storage
        self.elementBoundaryIntegrals = {}
        for ci in range(self.nc):
            self.elementBoundaryIntegrals[ci] = ((self.conservativeFlux is not None)
                                                  or (numericalFluxType is not None)
                                                  or (self.fluxBoundaryConditions[ci] == 'outFlow')
                                                  or (self.fluxBoundaryConditions[ci] == 'mixedFlow')
                                                  or (self.fluxBoundaryConditions[ci] == 'setFlow'))
        #
        # calculate some dimensions
        #
        # assume same space dim for all variables
        self.nSpace_global = self.u[0].femSpace.nSpace_global
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
        if self.shockCapturing is not None:
            for ci in self.shockCapturing.components:
                if elemQuadIsDict:
                    if ('numDiff', ci, ci) in elementQuadrature:
                        elementQuadratureDict[(
                            'numDiff', ci, ci)] = elementQuadrature[('numDiff', ci, ci)]
                    else:
                        elementQuadratureDict[(
                            'numDiff', ci, ci)] = elementQuadrature['default']
                else:
                    elementQuadratureDict[(
                        'numDiff', ci, ci)] = elementQuadrature
        if massLumping:
            for ci in list(self.coefficients.mass.keys()):
                elementQuadratureDict[('m', ci)] = Quadrature.SimplexLobattoQuadrature(
                    self.nSpace_global, 1)
            for I in self.coefficients.elementIntegralKeys:
                elementQuadratureDict[(
                    'stab',) + I[1:]] = Quadrature.SimplexLobattoQuadrature(self.nSpace_global, 1)
        if reactionLumping:
            for ci in list(self.coefficients.mass.keys()):
                elementQuadratureDict[('r', ci)] = Quadrature.SimplexLobattoQuadrature(
                    self.nSpace_global, 1)
            for I in self.coefficients.elementIntegralKeys:
                elementQuadratureDict[(
                    'stab',) + I[1:]] = Quadrature.SimplexLobattoQuadrature(self.nSpace_global, 1)
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
        self.nElementBoundaryQuadraturePoints_global = (self.mesh.nElements_global
                                                        * self.mesh.nElementBoundaries_element
                                                        * self.nElementBoundaryQuadraturePoints_elementBoundary)

        #
        # simplified allocations for test==trial and also check if space is mixed or not
        #
        self.q = {}
        self.ebq = {}
        self.ebq_global = {}
        self.ebqe = {}
        self.phi_ip = {}
        # To compute edge_based_cfl from within choose_dt of RKEV
        self.edge_based_cfl = np.zeros(self.u[0].dof.shape)
        self.dLow = None
        # Old DOFs
        # NOTE (Mql): It is important to link h_dof_old by reference with u[0].dof (and so on).
        # This is because  I need the initial condition to be passed to them as well (before calling calculateResidual).
        # During preStep I change this and copy the values instead of keeping the reference.
        self.h_dof_old = None
        self.hu_dof_old = None
        self.hv_dof_old = None
        # Vector for mass matrix
        self.check_positivity_water_height = True
        # mesh
        self.q['x'] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element, 3), 'd')
        self.ebqe['x'] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary, 3), 'd')
        self.ebq_global[('totalFlux', 0)] = np.zeros((self.mesh.nElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebq_global[('velocityAverage', 0)] = np.zeros((self.mesh.nElementBoundaries_global,
                                                               self.nElementBoundaryQuadraturePoints_elementBoundary, self.nSpace_global), 'd')
        self.q[('dV_u', 0)] = (1.0/self.mesh.nElements_global) * np.ones((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('dV_u', 1)] = (1.0/self.mesh.nElements_global) * np.ones((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('dV_u', 2)] = (1.0/self.mesh.nElements_global) * np.ones((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q['dV'] = self.q[('dV_u',0)]
        self.q[('u', 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('u', 1)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('u', 2)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('m', 0)] = self.q[('u', 0)]
        self.q[('m', 1)] = self.q[('u', 1)]
        self.q[('m', 2)] = self.q[('u', 2)]
        self.q[('m_last', 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('m_last', 1)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('m_last', 2)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('m_tmp', 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('m_tmp', 1)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('m_tmp', 2)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('f', 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element, self.nSpace_global), 'd')
        self.q[('velocity', 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element, self.nSpace_global), 'd')
        self.q[('cfl', 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.ebqe[('u', 0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('u', 1)] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('u', 2)] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('advectiveFlux_bc_flag', 0)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'i')
        self.ebqe[('advectiveFlux_bc_flag', 1)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'i')
        self.ebqe[('advectiveFlux_bc_flag', 2)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'i')
        self.ebqe[('diffusiveFlux_bc_flag', 1, 1)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'i')
        self.ebqe[('diffusiveFlux_bc_flag', 2, 2)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'i')
        self.ebqe[('advectiveFlux_bc', 0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('advectiveFlux_bc', 1)] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('advectiveFlux_bc', 2)] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('diffusiveFlux_bc', 1, 1)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('penalty')] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('diffusiveFlux_bc', 2, 2)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('velocity', 0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,
                                                  self.nElementBoundaryQuadraturePoints_elementBoundary, self.nSpace_global), 'd')
        self.ebqe[('velocity', 1)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,
                                                  self.nElementBoundaryQuadraturePoints_elementBoundary, self.nSpace_global), 'd')
        self.ebqe[('velocity', 2)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,
                                                  self.nElementBoundaryQuadraturePoints_elementBoundary, self.nSpace_global), 'd')
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
        
        del self.internalNodes
        self.internalNodes = None
        logEvent("Updating local to global mappings", 2)
        self.updateLocal2Global()
        logEvent("Building time integration object", 2)
        logEvent(memory("inflowBC, internalNodes,updateLocal2Global", "OneLevelTransport"), level=4)
        self.timeIntegration = TimeIntegrationClass(self)
        if options is not None:
            self.timeIntegration.setFromOptions(options)
        logEvent(memory("TimeIntegration", "OneLevelTransport"), level=4)
        logEvent("Calculating numerical quadrature formulas", 2)
        
        self.calculateQuadrature()
        self.setupFieldStrides()

        # hEps: this is use to regularize the flux and re-define the dry states
        self.eps = None
        self.hEps = None
        self.hReg = None
        self.ML = None  # lumped mass matrix
        self.MC_global = None  # consistent mass matrix
        ### Global C Matrices (mql)
        self.cterm_global = None
        self.cterm_transpose_global = None
        ### For convex limiting
        self.urelax = None
        self.drelax = None
        self.dH_minus_dL = None
        self.muH_minus_muL = None
        self.size_of_domain = None
        #
        self.hLow = None
        self.huLow = None
        self.hvLow = None
        #
        self.h_min = None
        self.h_max = None
        self.kin_max = None
        self.KE_tiny = None
        #
        self.extendedSourceTerm_hu = None
        self.extendedSourceTerm_hv = None
        self.new_SourceTerm_hu = None
        self.new_SourceTerm_hv = None
        ## for EV
        self.dij_small = None
        self.global_entropy_residual = None
        ## NORMALS
        self.COMPUTE_NORMALS = 1
        self.normalx = None
        self.normaly = None
        self.boundaryIndex = None
        self.reflectingBoundaryConditions = False
        self.reflectingBoundaryIndex = None
        self.wall_dof = None

        if 'reflecting_BCs' in dir(options) and options.reflecting_BCs == True:
            self.reflectingBoundaryConditions = True

        # Aux quantity at DOFs to be filled by optimized code (MQL)
        self.quantDOFs = None

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
        if self.postProcessing:
            self.q[('v', 0)] = self.tmpvt.q[('v', 0)]
            self.ebq[('v', 0)] = self.tmpvt.ebq[('v', 0)]
            self.ebq[('w', 0)] = self.tmpvt.ebq[('w', 0)]
            self.ebq['sqrt(det(g))'] = self.tmpvt.ebq['sqrt(det(g))']
            self.ebq['n'] = self.tmpvt.ebq['n']
            self.ebq[('dS_u', 0)] = self.tmpvt.ebq[('dS_u', 0)]
            self.ebqe['dS'] = self.tmpvt.ebqe['dS']
            self.ebqe['n'] = self.tmpvt.ebqe['n']
            self.ebq_global['n'] = self.tmpvt.ebq_global['n']
            self.ebq_global['x'] = self.tmpvt.ebq_global['x']
        from proteus import PostProcessingTools
        self.velocityPostProcessor = PostProcessingTools.VelocityPostProcessingChooser(self)
        logEvent(memory("velocity postprocessor", "OneLevelTransport"), level=4)
        # helper for writing out data storage
        from proteus import Archiver
        self.elementQuadratureDictionaryWriter = Archiver.XdmfWriter()
        self.elementBoundaryQuadratureDictionaryWriter = Archiver.XdmfWriter()
        self.exteriorElementBoundaryQuadratureDictionaryWriter = Archiver.XdmfWriter()
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
        # self.numericalFlux.setDirichletValues(self.ebqe)
        if self.movingDomain:
            self.MOVING_DOMAIN = 1.0
        else:
            self.MOVING_DOMAIN = 0.0
        # cek hack
        self.movingDomain = False
        self.MOVING_DOMAIN = 0.0
        if self.mesh.nodeVelocityArray is None:
            self.mesh.nodeVelocityArray = np.zeros(self.mesh.nodeArray.shape, 'd')
        # cek/ido todo replace python loops in modules with optimized code if possible/necessary
        self.forceStrongConditions = self.coefficients.forceStrongConditions
        self.dirichletConditionsForceDOF = {}
        if self.forceStrongConditions:
            for cj in range(self.nc):
                self.dirichletConditionsForceDOF[cj] = DOFBoundaryConditions(
                    self.u[cj].femSpace, dofBoundaryConditionsSetterDict[cj], weakDirichletConditions=False)

        compKernelFlag = 0
        self.elementDiameter = self.mesh.elementDiametersArray
        print(self.nSpace_global, " nSpace_global")
        self.sw2d = cSW2DCV_base(self.nSpace_global,
                                 self.nQuadraturePoints_element,
                                 self.u[0].femSpace.elementMaps.localFunctionSpace.dim,
                                 self.u[0].femSpace.referenceFiniteElement.localFunctionSpace.dim,
                                 self.testSpace[0].referenceFiniteElement.localFunctionSpace.dim,
                                 self.nElementBoundaryQuadraturePoints_elementBoundary,
                                 compKernelFlag)

        self.calculateResidual = self.sw2d.calculateResidual

        # define function to compute entropy viscosity residual
        self.calculateEV = self.sw2d.calculateEV

        if (self.coefficients.LUMPED_MASS_MATRIX):
            self.calculateJacobian = self.sw2d.calculateLumpedMassMatrix
        else:
            self.calculateJacobian = self.sw2d.calculateMassMatrix
        #
        self.dofsXCoord = None
        self.dofsYCoord = None
        self.constrainedDOFsIndices = None
        self.dataStructuresInitialized = False

        # PARALLEL VECTORS #
        self.par_normalx = None
        self.par_normaly = None
        self.par_ML = None
        # for source terms
        self.par_extendedSourceTerm_hu = None
        self.par_extendedSourceTerm_hv = None
        self.par_new_SourceTerm_hu = None
        self.par_new_SourceTerm_hv = None
        # for parallel entropy residual
        self.par_global_entropy_residual = None

    def FCTStep(self):
        # NOTE: this function is meant to be called within the solver
        comm = Comm.get()
        rowptr, colind, MassMatrix = self.MC_global.getCSRrepresentation()
        # Extract hnp1 from global solution u
        index = list(range(0, len(self.timeIntegration.u)))
        hIndex = index[0::3]
        huIndex = index[1::3]
        hvIndex = index[2::3]
        low_order = (self.hLow.copy(), self.huLow.copy(), self.hvLow.copy())
        # create limited solution
        limited_hnp1 = np.zeros(self.h_dof_old.shape)
        limited_hunp1 = np.zeros(self.h_dof_old.shape)
        limited_hvnp1 = np.zeros(self.h_dof_old.shape)

        self.KE_tiny = self.hEps * comm.globalMax(np.amax(self.kin_max))

        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["dt"] = self.timeIntegration.dt
        argsDict["NNZ"] = self.nnz
        argsDict["numDOFs"] = len(rowptr) - 1
        argsDict["lumped_mass_matrix"] = self.ML
        argsDict["h_old"] = self.h_dof_old
        argsDict["hu_old"] = self.hu_dof_old
        argsDict["hv_old"] = self.hv_dof_old
        argsDict["b_dof"] = self.coefficients.b.dof
        argsDict["high_order_hnp1"] = self.timeIntegration.u[hIndex]
        argsDict["high_order_hunp1"] = self.timeIntegration.u[huIndex]
        argsDict["high_order_hvnp1"] = self.timeIntegration.u[hvIndex]
        argsDict["extendedSourceTerm_hu"] = self.extendedSourceTerm_hu
        argsDict["extendedSourceTerm_hv"] = self.extendedSourceTerm_hv
        argsDict["limited_hnp1"] = limited_hnp1
        argsDict["limited_hunp1"] = limited_hunp1
        argsDict["limited_hvnp1"] = limited_hvnp1
        argsDict["csrRowIndeces_DofLoops"] = rowptr
        argsDict["csrColumnOffsets_DofLoops"] = colind
        argsDict["MassMatrix"] = MassMatrix
        argsDict["dH_minus_dL"] = self.dH_minus_dL
        argsDict["muH_minus_muL"] = self.muH_minus_muL
        argsDict["hEps"] = self.hEps
        argsDict["LUMPED_MASS_MATRIX"] = self.coefficients.LUMPED_MASS_MATRIX
        argsDict["NO_STAR_STATES"] = self.coefficients.NO_STAR_STATES
        argsDict["dLow"] = self.dLow
        argsDict["new_SourceTerm_hu"] = self.new_SourceTerm_hu
        argsDict["new_SourceTerm_hv"] = self.new_SourceTerm_hv
        argsDict["hLow"] = self.hLow
        argsDict["huLow"] = self.huLow
        argsDict["hvLow"] = self.hvLow
        argsDict["h_min"] = self.h_min
        argsDict["h_max"] = self.h_max
        argsDict["kin_max"] = self.kin_max
        argsDict["KE_tiny"] = self.KE_tiny
        # Two limiting passes, each in two kernel calls with ghost exchanges:
        # stage 0 computes R+- (exchanged), stage 1 computes l_ij and applies
        # the limited fluxes; hLow/huLow/hvLow are exchanged between passes.
        n_limit_iter = 2
        nnz = self.dH_minus_dL.size
        if self.FCT_all is None or self.FCT_all.size != 4 * nnz:
            self.FCT_all = np.zeros(4 * nnz, 'd')
        argsDict["FCT_all"] = self.FCT_all
        argsDict["Rneg_dof"] = self.Rneg_dof
        argsDict["Rpos_dof"] = self.Rpos_dof
        argsDict["num_neighbours"] = self.num_neighbours
        for limit_iter in range(n_limit_iter):
            argsDict["limit_iter"] = limit_iter
            argsDict["limit_stage"] = 0
            self.sw2d.convexLimiting(argsDict)
            self.par_Rneg.scatter_forward_insert()
            self.par_Rpos.scatter_forward_insert()
            argsDict["limit_stage"] = 1
            self.sw2d.convexLimiting(argsDict)
            if limit_iter < n_limit_iter - 1:
                self.par_hLow.scatter_forward_insert()
                self.par_huLow.scatter_forward_insert()
                self.par_hvLow.scatter_forward_insert()

        if self.coefficients.LOW_ORDER_ONLY:
            limited_hnp1[:], limited_hunp1[:], limited_hvnp1[:] = low_order
        # Pass the post processed hnp1 solution to global solution u
        self.timeIntegration.u[hIndex] = limited_hnp1
        self.timeIntegration.u[huIndex] = limited_hunp1
        self.timeIntegration.u[hvIndex] = limited_hvnp1

    def computeEV(self):
        entropy_residual = np.zeros(self.u[0].dof.shape)
        small = 0.0

        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["g"] = self.coefficients.g
        argsDict["h_dof_old"] = self.h_dof_old
        argsDict["hu_dof_old"] = self.hu_dof_old
        argsDict["hv_dof_old"] = self.hv_dof_old
        argsDict["b_dof"] = self.coefficients.b.dof
        argsDict["Cx"] = self.Cx
        argsDict["Cy"] = self.Cy
        argsDict["CTx"] = self.CTx
        argsDict["CTy"] = self.CTy
        argsDict["numDOFsPerEqn"] = self.numDOFsPerEqn
        argsDict["csrRowIndeces_DofLoops"] = self.rowptr_cMatrix
        argsDict["csrColumnOffsets_DofLoops"] = self.colind_cMatrix
        argsDict["lumped_mass_matrix"] = self.ML
        argsDict["eps"] = self.eps
        argsDict["hEps"] = self.hEps
        argsDict["global_entropy_residual"] = self.global_entropy_residual
        argsDict["dij_small"] = small

        # compute entropy residual
        self.sw2d.calculateEV(argsDict)

        # save things
        self.dij_small = globalMax(argsDict.dscalar["dij_small"])
    #

    def getDOFsCoord(self):
        # get x,y coordinates of all DOFs #
        self.dofsXCoord = np.zeros(self.u[0].dof.shape, 'd')
        self.dofsYCoord = np.zeros(self.u[0].dof.shape, 'd')
        self.dofsXCoord[:] = self.mesh.nodeArray[:, 0]
        self.dofsYCoord[:] = self.mesh.nodeArray[:, 1]
    #

    def getCMatrices(self):
        # since we only need cterm_global to persist, we can drop the other self.'s
        self.cterm = {}
        self.cterm_a = {}
        self.cterm_global = {}
        self.cterm_transpose = {}
        self.cterm_a_transpose = {}
        self.cterm_global_transpose = {}
        # Sparsity pattern for Jacobian
        rowptr, colind, nzval = self.jacobian.getCSRrepresentation()
        nnz = nzval.shape[-1]  # number of non-zero entries in sparse matrix

        ###########################################
        ##### SPARSITY PATTERN FOR C MATRICES #####
        ###########################################
        # construct nnz_cMatrix, czval_cMatrix, rowptr_cMatrix, colind_cMatrix C matrix
        nnz_cMatrix = nnz // 3 // 3  # This is always true for the SWEs in 2D
        nzval_cMatrix = np.zeros(nnz_cMatrix)  # This is enough since the values are filled later
        rowptr_cMatrix = np.zeros(self.u[0].dof.size + 1, 'i')  # NOTE that in particular rowptr_cMatrix[0]=0
        colind_cMatrix = np.zeros(nnz_cMatrix, 'i')
        # fill vector rowptr_cMatrix
        for i in range(1, rowptr_cMatrix.size):
            rowptr_cMatrix[i] = rowptr_cMatrix[i - 1] + (rowptr[3 * (i - 1) + 1] - rowptr[3 * (i - 1)])//3
            # = rowptr_cMatrix[i-1] + 1/3*(Number of columns of Jacobian's row 3*(i-1)=0, 3, 6, 9, 12, ... 3*(i-1), ..., 3*n-3)

        # fill vector colind_cMatrix
        i_cMatrix = 0  # ith row of cMatrix
        # 0 to total num of DOFs (i.e. num of rows of jacobian)
        for i in range(rowptr.size - 1):
            if (i % 3 == 0):  # Just consider the rows related to the h variable
                for j, offset in enumerate(range(rowptr[i], rowptr[i + 1])):
                    offset_cMatrix = list(range(rowptr_cMatrix[i_cMatrix], rowptr_cMatrix[i_cMatrix + 1]))
                    if (j % 3 == 0):
                        colind_cMatrix[offset_cMatrix[j//3]] = colind[offset]//3
                i_cMatrix += 1
        # END OF SPARSITY PATTERN FOR C MATRICES

        di = np.zeros((self.mesh.nElements_global,
                          self.nQuadraturePoints_element,
                          self.nSpace_global),
                         'd')  # direction of derivative
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
        #
        # lumped mass matrix
        #
        # assume a linear mass term, weighted by the porosity (storage is
        # phi h): dm = phi_h at the quadrature points
        l2g = self.u[0].femSpace.dofMap.l2g
        dm = np.einsum('eqj,ej->eq', self.q[('w', 0)],
                       self.coefficients.porosity_dof[l2g])
        elementMassMatrix = np.zeros((self.mesh.nElements_global,
                                      self.nDOF_test_element[0],
                                      self.nDOF_trial_element[0]), 'd')
        cfemIntegrals.updateMassJacobian_weak_lowmem(dm,
                                                     self.q[('w', 0)],
                                                     self.q[('w*dV_m', 0)],
                                                     elementMassMatrix)
        self.MC_a = nzval_cMatrix.copy()
        self.MC_global = SparseMat(self.nFreeDOF_global[0],
                                   self.nFreeDOF_global[0],
                                   nnz_cMatrix,
                                   self.MC_a,
                                   colind_cMatrix,
                                   rowptr_cMatrix)
        cfemIntegrals.zeroJacobian_CSR(nnz_cMatrix, self.MC_global)
        cfemIntegrals.updateGlobalJacobianFromElementJacobian_CSR(self.l2g[0]['nFreeDOF'],
                                                                  self.l2g[0]['freeLocal'],
                                                                  self.l2g[0]['nFreeDOF'],
                                                                  self.l2g[0]['freeLocal'],
                                                                  self.csrRowIndeces[(0, 0)] // 3 // 3,
                                                                  self.csrColumnOffsets[(0, 0)] // 3,
                                                                  elementMassMatrix,
                                                                  self.MC_global)

        diamD2 = np.sum(self.q['abs(det(J))'][:] * self.elementQuadratureWeights[('u', 0)])
        self.ML = np.zeros((self.nFreeDOF_global[0],), 'd')
        self.hReg = np.zeros((self.nFreeDOF_global[0],), 'd')
        for i in range(self.nFreeDOF_global[0]):
            self.ML[i] = self.MC_a[rowptr_cMatrix[i]:rowptr_cMatrix[i + 1]].sum()
            self.hReg[i] = self.ML[i] / diamD2 * self.u[0].dof.max()
        # np.testing.assert_almost_equal(self.ML.sum(), self.mesh.volume, err_msg="Trace of lumped mass matrix should be the domain volume",verbose=True)
        # np.testing.assert_almost_equal(self.ML.sum(), diamD2, err_msg="Trace of lumped mass matrix should be the domain volume",verbose=True)

        for d in range(self.nSpace_global):  # spatial dimensions
            # C matrices
            self.cterm[d] = np.zeros((self.mesh.nElements_global,
                                         self.nDOF_test_element[0],
                                         self.nDOF_trial_element[0]), 'd')
            self.cterm_a[d] = nzval_cMatrix.copy()
            self.cterm_global[d] = LAT.SparseMat(self.nFreeDOF_global[0],
                                                                self.nFreeDOF_global[0],
                                                                nnz_cMatrix,
                                                                self.cterm_a[d],
                                                                colind_cMatrix,
                                                                rowptr_cMatrix)
            cfemIntegrals.zeroJacobian_CSR(nnz_cMatrix, self.cterm_global[d])
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
                                                                      self.csrRowIndeces[(0, 0)] // 3 // 3,
                                                                      self.csrColumnOffsets[(0, 0)] // 3,
                                                                      self.cterm[d],
                                                                      self.cterm_global[d])
            # C Transpose matrices
            self.cterm_transpose[d] = np.zeros((self.mesh.nElements_global,
                                                   self.nDOF_test_element[0],
                                                   self.nDOF_trial_element[0]), 'd')
            self.cterm_a_transpose[d] = nzval_cMatrix.copy()
            self.cterm_global_transpose[d] = LAT.SparseMat(self.nFreeDOF_global[0],
                                                                          self.nFreeDOF_global[0],
                                                                          nnz_cMatrix,
                                                                          self.cterm_a_transpose[d],
                                                                          colind_cMatrix,
                                                                          rowptr_cMatrix)
            cfemIntegrals.zeroJacobian_CSR(nnz_cMatrix, self.cterm_global_transpose[d])
            di[:] = 0.0
            di[..., d] = -1.0
            cfemIntegrals.updateAdvectionJacobian_weak_lowmem(di,
                                                              self.q[('w', 0)],
                                                              self.q[('grad(w)*dV_f', 0)],
                                                              self.cterm_transpose[d])  # -int[(-di*grad(wi))*wj*dV]

            cfemIntegrals.updateGlobalJacobianFromElementJacobian_CSR(self.l2g[0]['nFreeDOF'],
                                                                      self.l2g[0]['freeLocal'],
                                                                      self.l2g[0]['nFreeDOF'],
                                                                      self.l2g[0]['freeLocal'],
                                                                      self.csrRowIndeces[(0, 0)] // 3 // 3,
                                                                      self.csrColumnOffsets[(0, 0)] // 3,
                                                                      self.cterm_transpose[d],
                                                                      self.cterm_global_transpose[d])
        #
        self.rowptr_cMatrix, self.colind_cMatrix, self.Cx = self.cterm_global[0].getCSRrepresentation()
        rowptr_cMatrix, colind_cMatrix, self.Cy = self.cterm_global[1].getCSRrepresentation()
        rowptr_cMatrix, colind_cMatrix, self.CTx = self.cterm_global_transpose[0].getCSRrepresentation()
        rowptr_cMatrix, colind_cMatrix, self.CTy = self.cterm_global_transpose[1].getCSRrepresentation()
        # (mql): I am assuming all variables live on the same FE space
        self.numDOFsPerEqn = self.u[0].dof.size
        self.numNonZeroEntries = len(self.Cx)
    #

    def updateConstrainedDOFs(self):
        # get indices for constrained DOFs
        if self.constrainedDOFsIndices is None:
            self.constrainedDOFsIndices = []
            self.constrainedDOFsIndices = self.coefficients.constrainedDOFs[0](self.dofsXCoord,
                                                                               self.dofsYCoord)
        for i in self.constrainedDOFsIndices:
            x = self.dofsXCoord[i]
            y = self.dofsYCoord[i]
            (h, hu, hv) = self.coefficients.constrainedDOFs[1](x,y,self.timeIntegration.t,
                                                               self.u[0].dof[i],
                                                               self.u[1].dof[i],
                                                               self.u[2].dof[i])
            if h is not None:
                self.u[0].dof[i] = h
            if hu is not None:
                self.u[1].dof[i] = hu
            if hv is not None:
                self.u[2].dof[i] = hv
    #

    def updateAllReflectingBoundaryConditions(self):
            self.forceStrongConditions = False
            for dummy, index in enumerate(self.boundaryIndex):
                vx = self.u[1].dof[index]
                vy = self.u[2].dof[index]
                vt = vx * self.normaly[index] - vy * self.normalx[index]
                self.u[1].dof[index] = vt * self.normaly[index]
                self.u[2].dof[index] = -vt * self.normalx[index]
    #

    def updatePartialReflectingBoundaryConditions(self):
            for dummy, index in enumerate(self.reflectingBoundaryIndex):
                vx = self.u[1].dof[index]
                vy = self.u[2].dof[index]
                vt = vx * self.normaly[index] - vy * self.normalx[index]
                self.u[1].dof[index] = vt * self.normaly[index]
                self.u[2].dof[index] = -vt * self.normalx[index]
    #

    def setWallDOFs(self):
        """1 on the DOFs whose rows treat the boundary as a no-flux wall (2 with
        WALL_RIEMANN: the wall flux is the Rusanov flux against the mirror state)."""
        self.wall_dof = np.zeros(self.u[0].dof.shape, 'd')
        self.tailwater_dof = np.zeros(self.u[0].dof.shape, 'd')
        self.tailwater_depth_dof = np.zeros(self.u[0].dof.shape, 'd')
        if not self.coefficients.INTEGRATE_BY_PARTS_WALLS:
            return
        wall_function = self.coefficients.wall_function
        if self.reflectingBoundaryConditions or wall_function is not None:
            ebN = self.mesh.exteriorElementBoundariesArray
            nodes = np.unique(self.mesh.elementBoundaryNodesArray[ebN].ravel())
            if wall_function is not None:
                X = self.mesh.nodeArray[nodes]
                nodes = nodes[np.asarray(wall_function([X[:, 0], X[:, 1]]), bool)]
            self.wall_dof[nodes] = 1.0
        else:
            self.wall_dof[self.mesh.nodeMaterialTypes == 99] = 1.0
        if self.coefficients.WALL_RIEMANN:
            self.wall_dof[self.wall_dof > 0.5] = 2.0
        self.setTailwaterDOFs()

    def setTailwaterDOFs(self):
        """1 on the open boundary nodes that carry a tailwater stage."""
        tw = self.coefficients.tailwater
        if tw is None:
            return
        ebN = self.mesh.exteriorElementBoundariesArray
        nodes = np.unique(self.mesh.elementBoundaryNodesArray[ebN].ravel())
        X = self.mesh.nodeArray[nodes]
        nodes = nodes[np.asarray(tw['where']([X[:, 0], X[:, 1]]), bool)]
        self.tailwater_dof[nodes] = 1.0
        self.wall_dof[nodes] = 0.0
        self.updateTailwater(0.0)

    def updateTailwater(self, t):
        """depth of the ghost state at the tailwater nodes: max(stage - Z, 0)"""
        tw = self.coefficients.tailwater
        X = self.mesh.nodeArray
        stage = np.asarray(tw['stage'](X[:, 0], X[:, 1], t), 'd') * np.ones(X.shape[0])
        self.tailwater_depth_dof[:] = np.maximum(stage - self.coefficients.b.dof, 0.0)
    #

    def initDataStructures(self):
        comm = Comm.get()
        self.setWallDOFs()
        # old vectors
        self.h_dof_old = np.copy(self.u[0].dof)
        self.hu_dof_old = np.copy(self.u[1].dof)
        self.hv_dof_old = np.copy(self.u[2].dof)
        # hEps
        self.eps = 1E-5
        self.hEps = self.eps * comm.globalMax(self.u[0].dof.max())
        # size_of_domain used in relaxation of bounds: the plan area of the
        # whole domain. In parallel each subdomain also holds overlap
        # elements, so sum only the owned elements and reduce across ranks
        # (summing mesh.volume double-counts the overlap).
        area_e = np.sum(self.q['abs(det(J))'] * self.elementQuadratureWeights[('u', 0)], axis=1)
        self.size_of_domain = comm.globalSum(area_e[:self.mesh.nElements_owned].sum())
        self.mesh.globalMesh.volume = self.size_of_domain
        # normal vectors
        self.normalx = np.zeros(self.u[0].dof.shape, 'd')
        self.normaly = np.zeros(self.u[0].dof.shape, 'd')
        # quantDOFs
        self.quantDOFs = np.zeros(self.u[0].dof.shape, 'd')
        # boundary Index: I do this in preStep since I need normalx and normaly to be initialized first
        # Allocate space for dLow (for the first stage in the SSP method)
        self.dLow = np.zeros(self.Cx.shape, 'd')
        # get coordinates of DOFs
        self.getDOFsCoord()
        # some vectors for convex limiting
        self.urelax = np.zeros(self.u[0].dof.shape, 'd')
        self.drelax = np.zeros(self.u[0].dof.shape, 'd')
        self.dH_minus_dL = np.zeros(self.Cx.shape, 'd')
        self.muH_minus_muL = np.zeros(self.Cx.shape, 'd')
        #
        self.hLow = np.zeros(self.u[0].dof.shape, 'd')
        self.huLow = np.zeros(self.u[0].dof.shape, 'd')
        self.hvLow = np.zeros(self.u[0].dof.shape, 'd')
        #
        self.h_min = np.zeros(self.u[0].dof.shape, 'd')
        self.h_max = np.zeros(self.u[0].dof.shape, 'd')
        self.kin_max = np.zeros(self.u[0].dof.shape, 'd')
        #
        self.extendedSourceTerm_hu = np.zeros(self.u[0].dof.shape, 'd')
        self.extendedSourceTerm_hv = np.zeros(self.u[0].dof.shape, 'd')
        self.new_SourceTerm_hu = np.zeros(self.u[0].dof.shape, 'd')
        self.new_SourceTerm_hv = np.zeros(self.u[0].dof.shape, 'd')
        self.global_entropy_residual = np.zeros(self.u[0].dof.shape, 'd')
        self.dij_small = 0.0

        # PARALLEL VECTORS #
        n=self.u[0].par_dof.dim_proc
        N=self.u[0].femSpace.dofMap.nDOF_all_processes
        nghosts = self.u[0].par_dof.nghosts
        subdomain2global=self.u[0].femSpace.dofMap.subdomain2global
        #
        self.par_urelax = LAT.ParVec_petsc4py(self.urelax,
                                                bs=1,
                                                n=n,N=N,nghosts=nghosts,
                                                subdomain2global=subdomain2global)
        self.par_drelax = LAT.ParVec_petsc4py(self.drelax,
                                                bs=1,
                                                n=n,N=N,nghosts=nghosts,
                                                subdomain2global=subdomain2global)
        self.par_hLow = LAT.ParVec_petsc4py(self.hLow,
                                                bs=1,
                                                n=n,N=N,nghosts=nghosts,
                                                subdomain2global=subdomain2global)
        self.par_huLow = LAT.ParVec_petsc4py(self.huLow,
                                                bs=1,
                                                n=n,N=N,nghosts=nghosts,
                                                subdomain2global=subdomain2global)
        self.par_hvLow = LAT.ParVec_petsc4py(self.hvLow,
                                                bs=1,
                                                n=n,N=N,nghosts=nghosts,
                                                subdomain2global=subdomain2global)
        #
        self.par_h_min = LAT.ParVec_petsc4py(self.h_min,
                                                bs=1,
                                                n=n,N=N,nghosts=nghosts,
                                                subdomain2global=subdomain2global)
        self.par_h_max = LAT.ParVec_petsc4py(self.h_max,
                                                bs=1,
                                                n=n,N=N,nghosts=nghosts,
                                                subdomain2global=subdomain2global)
        self.par_kin_max = LAT.ParVec_petsc4py(self.kin_max,
                                                bs=1,
                                                n=n,N=N,nghosts=nghosts,
                                                subdomain2global=subdomain2global)
        # nodal quantities the kernels read at neighbours: computed on every
        # local row, then the owners' values are copied to the ghosts
        self.delta_Sqd_h = np.zeros(self.u[0].dof.shape, 'd')
        self.delta_Sqd_kin = np.zeros(self.u[0].dof.shape, 'd')
        self.Rneg_dof = np.zeros(self.u[0].dof.shape, 'd')
        self.Rpos_dof = np.zeros(self.u[0].dof.shape, 'd')
        self.par_delta_Sqd_h = LAT.ParVec_petsc4py(self.delta_Sqd_h, bs=1, n=n, N=N, nghosts=nghosts,
                                                   subdomain2global=subdomain2global)
        self.par_delta_Sqd_kin = LAT.ParVec_petsc4py(self.delta_Sqd_kin, bs=1, n=n, N=N, nghosts=nghosts,
                                                     subdomain2global=subdomain2global)
        self.par_Rneg = LAT.ParVec_petsc4py(self.Rneg_dof, bs=1, n=n, N=N, nghosts=nghosts,
                                            subdomain2global=subdomain2global)
        self.par_Rpos = LAT.ParVec_petsc4py(self.Rpos_dof, bs=1, n=n, N=N, nghosts=nghosts,
                                            subdomain2global=subdomain2global)
        self.FCT_all = None  # FCT matrices and l_ij kept between limiter calls
        # neighbours per DOF (row length - 1); a ghost's local row is
        # truncated, so take the owner's count
        self.num_neighbours = (np.diff(self.rowptr_cMatrix) - 1).astype('d')
        self.par_num_neighbours = LAT.ParVec_petsc4py(self.num_neighbours, bs=1, n=n, N=N, nghosts=nghosts,
                                                      subdomain2global=subdomain2global)
        self.par_num_neighbours.scatter_forward_insert()
        #
        self.par_extendedSourceTerm_hu = LAT.ParVec_petsc4py(self.extendedSourceTerm_hu,
                                                bs=1,
                                                n=n,N=N,nghosts=nghosts,
                                                subdomain2global=subdomain2global)
        self.par_extendedSourceTerm_hv = LAT.ParVec_petsc4py(self.extendedSourceTerm_hv,
                                                bs=1,
                                                n=n,N=N,nghosts=nghosts,
                                                subdomain2global=subdomain2global)
        self.par_new_SourceTerm_hu = LAT.ParVec_petsc4py(self.new_SourceTerm_hu,
                                                bs=1,
                                                n=n,N=N,nghosts=nghosts,
                                                subdomain2global=subdomain2global)
        self.par_new_SourceTerm_hv = LAT.ParVec_petsc4py(self.new_SourceTerm_hv,
                                                bs=1,
                                                n=n,N=N,nghosts=nghosts,
                                                subdomain2global=subdomain2global)
        #
        self.par_global_entropy_residual = LAT.ParVec_petsc4py(self.global_entropy_residual,
                                                bs=1,
                                                n=n,N=N,nghosts=nghosts,
                                                subdomain2global=subdomain2global)
        self.par_normalx = LAT.ParVec_petsc4py(self.normalx,
                                                bs=1,
                                                n=n,N=N,nghosts=nghosts,
                                                subdomain2global=subdomain2global)
        self.par_normaly = LAT.ParVec_petsc4py(self.normaly,
                                                bs=1,
                                                n=n,N=N,nghosts=nghosts,
                                                subdomain2global=subdomain2global)
        self.par_ML = LAT.ParVec_petsc4py(self.ML,
                                                bs=1,
                                                n=n,N=N,nghosts=nghosts,
                                                subdomain2global=subdomain2global)
        self.par_ML.scatter_forward_insert()
        # Wall flags come from the subdomain's exterior boundary, which also
        # contains the partition boundary: owned flags are right, ghost flags
        # are not. Take the ghosts' flags from their owners.
        self.par_wall_dof = LAT.ParVec_petsc4py(self.wall_dof,
                                                bs=1,
                                                n=n,N=N,nghosts=nghosts,
                                                subdomain2global=subdomain2global)
        self.par_wall_dof.scatter_forward_insert()
        self.par_tailwater_dof = LAT.ParVec_petsc4py(self.tailwater_dof,
                                                     bs=1,
                                                     n=n,N=N,nghosts=nghosts,
                                                     subdomain2global=subdomain2global)
        self.par_tailwater_dof.scatter_forward_insert()
        
        if self.coefficients.BOUNDS_RELAXATION:
            self.urelax[:] = 1.0 + pow(np.sqrt(np.sqrt(self.ML / self.size_of_domain)),3)
            self.drelax[:] = 1.0 - pow(np.sqrt(np.sqrt(self.ML / self.size_of_domain)),3)
        else:
            self.urelax[:] = 1.0
            self.drelax[:] = 1.0
        self.par_urelax.scatter_forward_insert()
        self.par_drelax.scatter_forward_insert()
        #
        self.dataStructuresInitialized = True
    #

    def getResidual(self, u, r):
        """
        Calculate the element residuals and add in to the global residual
        """

        # COMPUTE C MATRIX #
        if self.cterm_global is None:
            self.getCMatrices()

        # INIT DATA STRUCTURES #
        if self.dataStructuresInitialized == False:
            self.initDataStructures()

        # LOAD THE UNKNOWNS INTO THE FINITE ELEMENT DOFs #
        self.timeIntegration.calculateCoefs()
        self.timeIntegration.calculateU(u)
        self.setUnknowns(self.timeIntegration.u)

        # SET TO ZERO RESIDUAL #
        r.fill(0.0)

        # REFLECTING BOUNDARY CONDITIONS ON ALL BOUNDARIES #
        # (only for the older nodal projection; integrated-by-parts walls
        #  need nothing here)
        legacy_walls = not self.coefficients.INTEGRATE_BY_PARTS_WALLS
        if legacy_walls and self.reflectingBoundaryConditions and self.boundaryIndex is not None:
            self.updateAllReflectingBoundaryConditions()

        # REFLECTING BOUNDARY CONDITIONS ON PARTIAL BOUNDARIES  #
        if legacy_walls and not self.reflectingBoundaryConditions and self.reflectingBoundaryIndex is not None:
            self.updatePartialReflectingBoundaryConditions()
        #

        # INIT BOUNDARY INDEX #
        # NOTE: this must be done after the first call to getResidual
        #   (to have normalx and normaly initialized)
        # I do this in preStep if firstStep=True

        # CONSTRAINT DOFs #
        if self.coefficients.constrainedDOFs is not None:
            self.updateConstrainedDOFs()
        #

        # DIRICHLET BOUNDARY CONDITIONS #
        if self.forceStrongConditions:
            for cj in range(len(self.dirichletConditionsForceDOF)):
                for dofN, g in list(self.dirichletConditionsForceDOF[cj].DOFBoundaryConditionsDict.items()):
                    
                    self.u[cj].dof[dofN] = g(self.dirichletConditionsForceDOF[cj].DOFBoundaryPointDict[dofN], self.timeIntegration.t)
        #       
        # CHECK POSITIVITY OF WATER HEIGHT
        if (self.check_positivity_water_height == True):
            assert self.u[0].dof.min() >= -self.eps * self.u[0].dof.max(), ("Negative water height: ", self.u[0].dof.min())

        # lets call calculate EV first and distribute
        self.computeEV()
        self.par_global_entropy_residual.scatter_forward_insert()

        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["mesh_l2g"] = self.mesh.elementNodesArray
        argsDict["dV_ref"] = self.elementQuadratureWeights[('u', 0)]
        argsDict["h_trial_ref"] = self.u[0].femSpace.psi
        argsDict["h_grad_trial_ref"] = self.u[0].femSpace.grad_psi
        argsDict["h_test_ref"] = self.u[0].femSpace.psi
        argsDict["h_grad_test_ref"] = self.u[0].femSpace.grad_psi
        argsDict["vel_trial_ref"] = self.u[1].femSpace.psi
        argsDict["vel_grad_trial_ref"] = self.u[1].femSpace.grad_psi
        argsDict["vel_test_ref"] = self.u[1].femSpace.psi
        argsDict["vel_grad_test_ref"] = self.u[1].femSpace.grad_psi
        argsDict["mesh_trial_trace_ref"] = self.u[0].femSpace.elementMaps.psi_trace
        argsDict["mesh_grad_trial_trace_ref"] = self.u[0].femSpace.elementMaps.grad_psi_trace
        argsDict["h_trial_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["h_grad_trial_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["h_test_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["h_grad_test_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["vel_trial_trace_ref"] = self.u[1].femSpace.psi_trace
        argsDict["vel_grad_trial_trace_ref"] = self.u[1].femSpace.grad_psi_trace
        argsDict["vel_test_trace_ref"] = self.u[1].femSpace.psi_trace
        argsDict["vel_grad_test_trace_ref"] = self.u[1].femSpace.grad_psi_trace
        argsDict["normal_ref"] = self.u[0].femSpace.elementMaps.boundaryNormals
        argsDict["boundaryJac_ref"] = self.u[0].femSpace.elementMaps.boundaryJacobians
        argsDict["elementDiameter"] = self.elementDiameter
        argsDict["nElements_global"] = self.mesh.nElements_global
        argsDict["g"] = self.coefficients.g
        argsDict["h_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["vel_l2g"] = self.u[1].femSpace.dofMap.l2g
        argsDict["h_dof_old"] = self.h_dof_old
        argsDict["hu_dof_old"] = self.hu_dof_old
        argsDict["hv_dof_old"] = self.hv_dof_old
        argsDict["b_dof"] = self.coefficients.b.dof
        argsDict["h_dof"] = self.u[0].dof
        argsDict["hu_dof"] = self.u[1].dof
        argsDict["hv_dof"] = self.u[2].dof
        argsDict["q_cfl"] = self.q[('cfl', 0)]
        argsDict["sdInfo_hu_hu_rowptr"] = self.coefficients.sdInfo[(1, 1)][0]
        argsDict["sdInfo_hu_hu_colind"] = self.coefficients.sdInfo[(1, 1)][1]
        argsDict["sdInfo_hu_hv_rowptr"] = self.coefficients.sdInfo[(1, 2)][0]
        argsDict["sdInfo_hu_hv_colind"] = self.coefficients.sdInfo[(1, 2)][1]
        argsDict["sdInfo_hv_hv_rowptr"] = self.coefficients.sdInfo[(2, 2)][0]
        argsDict["sdInfo_hv_hv_colind"] = self.coefficients.sdInfo[(2, 2)][1]
        argsDict["sdInfo_hv_hu_rowptr"] = self.coefficients.sdInfo[(2, 1)][0]
        argsDict["sdInfo_hv_hu_colind"] = self.coefficients.sdInfo[(2, 1)][1]
        argsDict["offset_h"] = self.offset[0]
        argsDict["offset_hu"] = self.offset[1]
        argsDict["offset_hv"] = self.offset[2]
        argsDict["stride_h"] = self.stride[0]
        argsDict["stride_hu"] = self.stride[1]
        argsDict["stride_hv"] = self.stride[2]
        argsDict["globalResidual"] = r
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["isDOFBoundary_h"] = self.numericalFlux.isDOFBoundary[0]
        argsDict["isDOFBoundary_hu"] = self.numericalFlux.isDOFBoundary[1]
        argsDict["isDOFBoundary_hv"] = self.numericalFlux.isDOFBoundary[2]
        argsDict["isAdvectiveFluxBoundary_h"] = self.ebqe[('advectiveFlux_bc_flag', 0)]
        argsDict["isAdvectiveFluxBoundary_hu"] = self.ebqe[('advectiveFlux_bc_flag', 1)]
        argsDict["isAdvectiveFluxBoundary_hv"] = self.ebqe[('advectiveFlux_bc_flag', 2)]
        argsDict["isDiffusiveFluxBoundary_hu"] = self.ebqe[('diffusiveFlux_bc_flag', 1, 1)]
        argsDict["isDiffusiveFluxBoundary_hv"] = self.ebqe[('diffusiveFlux_bc_flag', 2, 2)]
        argsDict["ebqe_bc_h_ext"] = self.numericalFlux.ebqe[('u', 0)]
        argsDict["ebqe_bc_flux_mass_ext"] = self.ebqe[('advectiveFlux_bc', 0)]
        argsDict["ebqe_bc_flux_mom_hu_adv_ext"] = self.ebqe[('advectiveFlux_bc', 1)]
        argsDict["ebqe_bc_flux_mom_hv_adv_ext"] = self.ebqe[('advectiveFlux_bc', 2)]
        argsDict["ebqe_bc_hu_ext"] = self.numericalFlux.ebqe[('u', 1)]
        argsDict["ebqe_bc_flux_hu_diff_ext"] = self.ebqe[('diffusiveFlux_bc', 1, 1)]
        argsDict["ebqe_penalty_ext"] = self.ebqe[('penalty')]
        argsDict["ebqe_bc_hv_ext"] = self.numericalFlux.ebqe[('u', 2)]
        argsDict["ebqe_bc_flux_hv_diff_ext"] = self.ebqe[('diffusiveFlux_bc', 2, 2)]
        argsDict["q_velocity"] = self.q[('velocity', 0)]
        argsDict["ebqe_velocity"] = self.ebqe[('velocity', 0)]
        argsDict["flux"] = self.ebq_global[('totalFlux', 0)]
        argsDict["elementResidual_h"] = self.elementResidual[0]
        argsDict["Cx"] = self.Cx
        argsDict["Cy"] = self.Cy
        argsDict["CTx"] = self.CTx
        argsDict["CTy"] = self.CTy
        argsDict["numDOFsPerEqn"] = self.numDOFsPerEqn
        argsDict["NNZ"] = self.numNonZeroEntries
        argsDict["csrRowIndeces_DofLoops"] = self.rowptr_cMatrix
        argsDict["csrColumnOffsets_DofLoops"] = self.colind_cMatrix
        argsDict["lumped_mass_matrix"] = self.ML
        argsDict["cfl_run"] = self.timeIntegration.runCFL
        argsDict["eps"] = self.eps
        argsDict["hEps"] = self.hEps
        argsDict["hnp1_at_quad_point"] = self.q[('u', 0)]
        argsDict["hunp1_at_quad_point"] = self.q[('u', 1)]
        argsDict["hvnp1_at_quad_point"] = self.q[('u', 2)]
        argsDict["extendedSourceTerm_hu"] = self.extendedSourceTerm_hu
        argsDict["extendedSourceTerm_hv"] = self.extendedSourceTerm_hv
        argsDict["dH_minus_dL"] = self.dH_minus_dL
        argsDict["muH_minus_muL"] = self.muH_minus_muL
        argsDict["cE"] = float(self.coefficients.cE)
        argsDict["LUMPED_MASS_MATRIX"] = self.coefficients.LUMPED_MASS_MATRIX
        argsDict["dt"] = self.timeIntegration.dt
        argsDict["LINEAR_FRICTION"] = self.coefficients.LINEAR_FRICTION
        argsDict["mannings"] = float(self.coefficients.mannings)
        argsDict["quantDOFs"] = self.quantDOFs
        argsDict["SECOND_CALL_CALCULATE_RESIDUAL"] = self.secondCallCalculateResidual
        argsDict["COMPUTE_NORMALS"] = self.COMPUTE_NORMALS
        argsDict["wall_dof"] = self.wall_dof
        argsDict["tailwater_dof"] = self.tailwater_dof
        argsDict["tailwater_depth_dof"] = self.tailwater_depth_dof
        argsDict["porosity_dof"] = self.coefficients.porosity_dof
        argsDict["drag_alpha_dof"] = self.coefficients.drag_alpha_dof
        argsDict["drag_beta_dof"] = self.coefficients.drag_beta_dof
        argsDict["normalx"] = self.normalx
        argsDict["normaly"] = self.normaly
        argsDict["dLow"] = self.dLow
        argsDict["lstage"] = self.timeIntegration.lstage
        argsDict["BOUNDS_EXCLUDE_SELF"] = self.coefficients.BOUNDS_EXCLUDE_SELF
        argsDict["NODAL_H_BOUNDS"] = self.coefficients.NODAL_H_BOUNDS
        argsDict["NO_STAR_STATES"] = self.coefficients.NO_STAR_STATES
        argsDict["new_SourceTerm_hu"] = self.new_SourceTerm_hu
        argsDict["new_SourceTerm_hv"] = self.new_SourceTerm_hv
        argsDict["global_entropy_residual"] = self.global_entropy_residual
        argsDict["dij_small"] = self.dij_small
        argsDict["hLow"] = self.hLow
        argsDict["huLow"] = self.huLow
        argsDict["hvLow"] = self.hvLow
        argsDict["h_min"] = self.h_min
        argsDict["h_max"] = self.h_max
        argsDict["kin_max"] = self.kin_max
        argsDict["size_of_domain"] = self.size_of_domain
        argsDict["urelax"] = self.urelax
        argsDict["drelax"] = self.drelax

        ## call calculate residual
        # delta_Sqd = sum over neighbours j of (q_i - q_j), for the bounds
        # relaxation; computed here so the ghost values can be exchanged
        rowptr = self.rowptr_cMatrix; colind = self.colind_cMatrix
        rows = np.repeat(np.arange(len(rowptr) - 1), np.diff(rowptr))
        h = self.h_dof_old
        hmax = np.where(h > self.hEps, h, self.hEps)
        kin = 0.5 * (self.hu_dof_old * self.hu_dof_old + self.hv_dof_old * self.hv_dof_old)
        kin = kin * (2.0 * h / (h * h + hmax * hmax))
        if self.coefficients.BOUNDS_RELAXATION:
            self.delta_Sqd_h[:] = np.add.reduceat(h[rows] - h[colind], rowptr[:-1])
            self.delta_Sqd_kin[:] = np.add.reduceat(kin[rows] - kin[colind], rowptr[:-1])
        else:
            self.delta_Sqd_h[:] = 0.0
            self.delta_Sqd_kin[:] = 0.0
        self.par_delta_Sqd_h.scatter_forward_insert()
        self.par_delta_Sqd_kin.scatter_forward_insert()
        argsDict["delta_Sqd_h"] = self.delta_Sqd_h
        argsDict["delta_Sqd_kin"] = self.delta_Sqd_kin
        self.calculateResidual(argsDict)

        ## distribute local bounds and low order solutions (with bar states)
        self.par_hLow.scatter_forward_insert()
        self.par_huLow.scatter_forward_insert()
        self.par_hvLow.scatter_forward_insert()
        #
        self.par_h_min.scatter_forward_insert()
        self.par_h_max.scatter_forward_insert()
        self.par_kin_max.scatter_forward_insert()

        ## distribute source terms (not sure if needed)
        self.par_extendedSourceTerm_hu.scatter_forward_insert()
        self.par_extendedSourceTerm_hv.scatter_forward_insert()
        ##
        self.par_new_SourceTerm_hu.scatter_forward_insert()
        self.par_new_SourceTerm_hv.scatter_forward_insert()

        if self.COMPUTE_NORMALS==1:
            self.par_normalx.scatter_forward_insert()
            self.par_normaly.scatter_forward_insert()
            self.COMPUTE_NORMALS = 0
        #

        # for reflecting conditions on all boundaries
        legacy_walls = not self.coefficients.INTEGRATE_BY_PARTS_WALLS
        if legacy_walls and self.reflectingBoundaryConditions and self.boundaryIndex is not None:
            for dummy, index in enumerate(self.boundaryIndex):
                r[self.offset[1]+self.stride[1]*index]=0.
                r[self.offset[2]+self.stride[2]*index]=0.
            #
        #

        # for reflecting conditions on partial boundaries
        if legacy_walls and not self.reflectingBoundaryConditions and self.reflectingBoundaryIndex is not None:
            for dummy, index in enumerate(self.reflectingBoundaryIndex):
                r[self.offset[1]+self.stride[1]*index]=0.
                r[self.offset[2]+self.stride[2]*index]=0.
            #
        #

        if self.forceStrongConditions:
            for cj in range(len(self.dirichletConditionsForceDOF)):
                for dofN, g in list(self.dirichletConditionsForceDOF[cj].DOFBoundaryConditionsDict.items()):
                    r[self.offset[cj] + self.stride[cj] * dofN] = 0.
        #
        if self.constrainedDOFsIndices is not None:
            for index in self.constrainedDOFsIndices:
                for cj in range(self.nc):
                    global_dofN = self.offset[cj] + self.stride[cj] * index
                    r[global_dofN] = 0.
        #
        logEvent("Global residual SWEs: ", level=9, data=r)
        # mwf decide if this is reasonable for keeping solver statistics
        self.nonlinear_function_evaluations += 1

    def getJacobian(self, jacobian):
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
        argsDict["h_trial_ref"] = self.u[0].femSpace.psi
        argsDict["h_grad_trial_ref"] = self.u[0].femSpace.grad_psi
        argsDict["h_test_ref"] = self.u[0].femSpace.psi
        argsDict["h_grad_test_ref"] = self.u[0].femSpace.grad_psi
        argsDict["vel_trial_ref"] = self.u[1].femSpace.psi
        argsDict["vel_grad_trial_ref"] = self.u[1].femSpace.grad_psi
        argsDict["vel_test_ref"] = self.u[1].femSpace.psi
        argsDict["vel_grad_test_ref"] = self.u[1].femSpace.grad_psi
        argsDict["mesh_trial_trace_ref"] = self.u[0].femSpace.elementMaps.psi_trace
        argsDict["mesh_grad_trial_trace_ref"] = self.u[0].femSpace.elementMaps.grad_psi_trace
        argsDict["dS_ref"] = self.elementBoundaryQuadratureWeights[('u', 0)]
        argsDict["h_trial_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["h_grad_trial_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["h_test_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["h_grad_test_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["vel_trial_trace_ref"] = self.u[1].femSpace.psi_trace
        argsDict["vel_grad_trial_trace_ref"] = self.u[1].femSpace.grad_psi_trace
        argsDict["vel_test_trace_ref"] = self.u[1].femSpace.psi_trace
        argsDict["vel_grad_test_trace_ref"] = self.u[1].femSpace.grad_psi_trace
        argsDict["normal_ref"] = self.u[0].femSpace.elementMaps.boundaryNormals
        argsDict["boundaryJac_ref"] = self.u[0].femSpace.elementMaps.boundaryJacobians
        argsDict["elementDiameter"] = self.elementDiameter
        argsDict["nElements_global"] = self.mesh.nElements_global
        argsDict["g"] = self.coefficients.g
        argsDict["h_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["vel_l2g"] = self.u[1].femSpace.dofMap.l2g
        argsDict["b_dof"] = self.coefficients.b.dof
        argsDict["h_dof"] = self.u[0].dof
        argsDict["hu_dof"] = self.u[1].dof
        argsDict["hv_dof"] = self.u[2].dof
        argsDict["q_cfl"] = self.q[('cfl', 0)]
        argsDict["sdInfo_hu_hu_rowptr"] = self.coefficients.sdInfo[(1, 1)][0]
        argsDict["sdInfo_hu_hu_colind"] = self.coefficients.sdInfo[(1, 1)][1]
        argsDict["sdInfo_hu_hv_rowptr"] = self.coefficients.sdInfo[(1, 2)][0]
        argsDict["sdInfo_hu_hv_colind"] = self.coefficients.sdInfo[(1, 2)][1]
        argsDict["sdInfo_hv_hv_rowptr"] = self.coefficients.sdInfo[(2, 2)][0]
        argsDict["sdInfo_hv_hv_colind"] = self.coefficients.sdInfo[(2, 2)][1]
        argsDict["sdInfo_hv_hu_rowptr"] = self.coefficients.sdInfo[(2, 1)][0]
        argsDict["sdInfo_hv_hu_colind"] = self.coefficients.sdInfo[(2, 1)][1]
        argsDict["csrRowIndeces_h_h"] = self.csrRowIndeces[(0, 0)]
        argsDict["csrColumnOffsets_h_h"] = self.csrColumnOffsets[(0, 0)]
        argsDict["csrRowIndeces_h_hu"] = self.csrRowIndeces[(0, 1)]
        argsDict["csrColumnOffsets_h_hu"] = self.csrColumnOffsets[(0, 1)]
        argsDict["csrRowIndeces_h_hv"] = self.csrRowIndeces[(0, 2)]
        argsDict["csrColumnOffsets_h_hv"] = self.csrColumnOffsets[(0, 2)]
        argsDict["csrRowIndeces_hu_h"] = self.csrRowIndeces[(1, 0)]
        argsDict["csrColumnOffsets_hu_h"] = self.csrColumnOffsets[(1, 0)]
        argsDict["csrRowIndeces_hu_hu"] = self.csrRowIndeces[(1, 1)]
        argsDict["csrColumnOffsets_hu_hu"] = self.csrColumnOffsets[(1, 1)]
        argsDict["csrRowIndeces_hu_hv"] = self.csrRowIndeces[(1, 2)]
        argsDict["csrColumnOffsets_hu_hv"] = self.csrColumnOffsets[(1, 2)]
        argsDict["csrRowIndeces_hv_h"] = self.csrRowIndeces[(2, 0)]
        argsDict["csrColumnOffsets_hv_h"] = self.csrColumnOffsets[(2, 0)]
        argsDict["csrRowIndeces_hv_hu"] = self.csrRowIndeces[(2, 1)]
        argsDict["csrColumnOffsets_hv_hu"] = self.csrColumnOffsets[(2, 1)]
        argsDict["csrRowIndeces_hv_hv"] = self.csrRowIndeces[(2, 2)]
        argsDict["csrColumnOffsets_hv_hv"] = self.csrColumnOffsets[(2, 2)]
        argsDict["globalJacobian"] = jacobian.getCSRrepresentation()[2]
        argsDict["porosity_dof"] = self.coefficients.porosity_dof
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["isDOFBoundary_h"] = self.numericalFlux.isDOFBoundary[0]
        argsDict["isDOFBoundary_hu"] = self.numericalFlux.isDOFBoundary[1]
        argsDict["isDOFBoundary_hv"] = self.numericalFlux.isDOFBoundary[2]
        argsDict["isAdvectiveFluxBoundary_h"] = self.ebqe[('advectiveFlux_bc_flag', 0)]
        argsDict["isAdvectiveFluxBoundary_hu"] = self.ebqe[('advectiveFlux_bc_flag', 1)]
        argsDict["isAdvectiveFluxBoundary_hv"] = self.ebqe[('advectiveFlux_bc_flag', 2)]
        argsDict["isDiffusiveFluxBoundary_hu"] = self.ebqe[('diffusiveFlux_bc_flag', 1, 1)]
        argsDict["isDiffusiveFluxBoundary_hv"] = self.ebqe[('diffusiveFlux_bc_flag', 2, 2)]
        argsDict["ebqe_bc_h_ext"] = self.numericalFlux.ebqe[('u', 0)]
        argsDict["ebqe_bc_flux_mass_ext"] = self.ebqe[('advectiveFlux_bc', 0)]
        argsDict["ebqe_bc_flux_mom_hu_adv_ext"] = self.ebqe[('advectiveFlux_bc', 1)]
        argsDict["ebqe_bc_flux_mom_hv_adv_ext"] = self.ebqe[('advectiveFlux_bc', 2)]
        argsDict["ebqe_bc_hu_ext"] = self.numericalFlux.ebqe[('u', 1)]
        argsDict["ebqe_bc_flux_hu_diff_ext"] = self.ebqe[('diffusiveFlux_bc', 1, 1)]
        argsDict["ebqe_penalty_ext"] = self.ebqe[('penalty')]
        argsDict["ebqe_bc_hv_ext"] = self.numericalFlux.ebqe[('u', 2)]
        argsDict["ebqe_bc_flux_hv_diff_ext"] = self.ebqe[('diffusiveFlux_bc', 2, 2)]
        argsDict["csrColumnOffsets_eb_h_h"] = self.csrColumnOffsets_eb[(0, 0)]
        argsDict["csrColumnOffsets_eb_h_hu"] = self.csrColumnOffsets_eb[(0, 1)]
        argsDict["csrColumnOffsets_eb_h_hv"] = self.csrColumnOffsets_eb[(0, 2)]
        argsDict["csrColumnOffsets_eb_hu_h"] = self.csrColumnOffsets_eb[(1, 0)]
        argsDict["csrColumnOffsets_eb_hu_hu"] = self.csrColumnOffsets_eb[(1, 1)]
        argsDict["csrColumnOffsets_eb_hu_hv"] = self.csrColumnOffsets_eb[(1, 2)]
        argsDict["csrColumnOffsets_eb_hv_h"] = self.csrColumnOffsets_eb[(2, 0)]
        argsDict["csrColumnOffsets_eb_hv_hu"] = self.csrColumnOffsets_eb[(2, 1)]
        argsDict["csrColumnOffsets_eb_hv_hv"] = self.csrColumnOffsets_eb[(2, 2)]
        argsDict["dt"] = self.timeIntegration.dt
        self.calculateJacobian(argsDict)

        # for reflecting conditions on ALL boundaries
        legacy_walls = not self.coefficients.INTEGRATE_BY_PARTS_WALLS
        if legacy_walls and self.reflectingBoundaryConditions and self.boundaryIndex is not None:
            for dummy, index in enumerate(self.boundaryIndex):
                global_dofN = self.offset[1] + self.stride[1] * index
                for i in range(self.rowptr[global_dofN], self.rowptr[global_dofN + 1]):
                    if (self.colind[i] == global_dofN):
                        self.nzval[i] = 1.0
                    else:
                        self.nzval[i] = 0.0
                    #
                #
                global_dofN = self.offset[2] + self.stride[2] * index
                for i in range(self.rowptr[global_dofN], self.rowptr[global_dofN + 1]):
                    if (self.colind[i] == global_dofN):
                        self.nzval[i] = 1.0
                    else:
                        self.nzval[i] = 0.0
                    #
                #
            #
        #

        # for reflecting conditions partial boundaries
        if legacy_walls and not self.reflectingBoundaryConditions and self.reflectingBoundaryIndex is not None:
            for dummy, index in enumerate(self.reflectingBoundaryIndex):
                global_dofN = self.offset[1] + self.stride[1] * index
                for i in range(self.rowptr[global_dofN], self.rowptr[global_dofN + 1]):
                    if (self.colind[i] == global_dofN):
                        self.nzval[i] = 1.0
                    else:
                        self.nzval[i] = 0.0
                    #
                #
                global_dofN = self.offset[2] + self.stride[2] * index
                for i in range(self.rowptr[global_dofN], self.rowptr[global_dofN + 1]):
                    if (self.colind[i] == global_dofN):
                        self.nzval[i] = 1.0
                    else:
                        self.nzval[i] = 0.0
                    #
                #
            #
        #

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
        #
        if self.constrainedDOFsIndices is not None:
            for index in self.constrainedDOFsIndices:
                for cj in range(self.nc):
                        global_dofN = self.offset[cj] + self.stride[cj] * index
                        for i in range(self.rowptr[global_dofN], self.rowptr[global_dofN + 1]):
                            if (self.colind[i] == global_dofN):
                                self.nzval[i] = 1.0
                            else:
                                self.nzval[i] = 0.0
        #
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
        if self.postProcessing:
            self.tmpvt.calculateElementQuadrature()
        self.u[0].femSpace.elementMaps.getValues(self.elementQuadraturePoints, self.q['x'])
        self.u[0].femSpace.elementMaps.getBasisValuesRef(self.elementQuadraturePoints)
        self.u[0].femSpace.elementMaps.getBasisGradientValuesRef(self.elementQuadraturePoints)
        self.u[0].femSpace.getBasisValuesRef(self.elementQuadraturePoints)
        self.u[0].femSpace.getBasisGradientValuesRef(self.elementQuadraturePoints)
        self.u[1].femSpace.getBasisValuesRef(self.elementQuadraturePoints)
        self.u[1].femSpace.getBasisGradientValuesRef(self.elementQuadraturePoints)
        self.coefficients.initializeElementQuadrature(self.timeIntegration.t, self.q)

    def calculateElementBoundaryQuadrature(self):
        """
        Calculate the physical location and weights of the quadrature rules
        and the shape information at the quadrature points on element boundaries.

        This function should be called only when the mesh changes.
        """
        if self.postProcessing:
            self.tmpvt.calculateElementBoundaryQuadrature()
        pass

    def calculateExteriorElementBoundaryQuadrature(self):
        """
        Calculate the physical location and weights of the quadrature rules
        and the shape information at the quadrature points on global element boundaries.

        This function should be called only when the mesh changes.
        """
        if self.postProcessing:
            self.tmpvt.calculateExteriorElementBoundaryQuadrature()
        #
        # get physical locations of element boundary quadrature points
        #
        # assume all components live on the same mesh
        self.u[0].femSpace.elementMaps.getBasisValuesTraceRef(self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.elementMaps.getBasisGradientValuesTraceRef(self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.getBasisValuesTraceRef(self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.getBasisGradientValuesTraceRef(self.elementBoundaryQuadraturePoints)
        self.u[1].femSpace.getBasisValuesTraceRef(self.elementBoundaryQuadraturePoints)
        self.u[1].femSpace.getBasisGradientValuesTraceRef(self.elementBoundaryQuadraturePoints)
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
        OneLevelTransport.calculateAuxiliaryQuantitiesAfterStep(self)

    def getForce(self, cg, forceExtractionFaces, force, moment):
        pass
