import proteus
from proteus import FemTools
from proteus import LinearAlgebraTools as LAT
from proteus.mprans.cGN_SW2DCV import *
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
        argsDict["heta_dof_old"] = self.transport.u[3].dof
        argsDict["csrRowIndeces_DofLoops"] = rowptr_cMatrix
        argsDict["csrColumnOffsets_DofLoops"] = colind_cMatrix
        argsDict["hEps"] = self.transport.hEps
        argsDict["Cx"] = Cx
        argsDict["Cy"] = Cy
        argsDict["CTx"] = CTx
        argsDict["CTy"] = CTy
        argsDict["inverse_mesh"] = self.transport.inverse_mesh
        argsDict["edge_based_cfl"] = self.transport.edge_based_cfl
        if self.transport.wall_dof is None:
            self.transport.setWallDOFs()
        argsDict["wall_dof"] = self.transport.wall_dof
        argsDict["porosity_dof"] = self.transport.coefficients.porosity_dof
        adjusted_maxCFL = self.transport.dsw_2d.calculateEdgeBasedCFL(argsDict)

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
                self.transport.heta_dof_old[:] = self.u_dof_lstage[3]
                self.transport.hw_dof_old[:] = self.u_dof_lstage[4]
                self.transport.hbeta_dof_old[:] = self.u_dof_lstage[5]
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
                self.transport.heta_dof_old[:] = self.u_dof_lstage[3]
                self.transport.hw_dof_old[:] = self.u_dof_lstage[4]
                self.transport.hbeta_dof_old[:] = self.u_dof_lstage[5]
                logEvent("Second stage of SSP33 method finished", level=4)
            else:
                for ci in range(self.nc):
                    self.u_dof_lstage[ci][:] = self.transport.u[ci].dof
                    self.u_dof_lstage[ci][:] *= 2.0/3.0
                    self.u_dof_lstage[ci][:] += 1.0 / 3.0 * self.u_dof_last[ci]
                    # update solution to u[0].dof
                    self.transport.u[ci].dof[:] = self.u_dof_lstage[ci]
                # update u_dof_old
                self.transport.h_dof_old[:] = self.u_dof_lstage[0]
                self.transport.hu_dof_old[:] = self.u_dof_lstage[1]
                self.transport.hv_dof_old[:] = self.u_dof_lstage[2]
                self.transport.heta_dof_old[:] = self.u_dof_lstage[3]
                self.transport.hw_dof_old[:] = self.u_dof_lstage[4]
                self.transport.hbeta_dof_old[:] = self.u_dof_lstage[5]
                # self.transport.h_dof_old[:] = self.u_dof_last[0]
                # self.transport.hu_dof_old[:] = self.u_dof_last[1]
                # self.transport.hv_dof_old[:] = self.u_dof_last[2]
                # self.transport.heta_dof_old[:] = self.u_dof_last[3]
                # self.transport.hw_dof_old[:] = self.u_dof_last[4]
                # self.transport.hbeta_dof_old[:] = self.u_dof_last[5]
        elif self.timeOrder == 2:
            if self.lstage == 1:
                logEvent("First stage of SSP22 method finished", level=4)
                for ci in range(self.nc):
                    self.u_dof_lstage[ci][:] = self.transport.u[ci].dof
                # Update u_dof_old
                self.transport.h_dof_old[:] = self.u_dof_lstage[0]
                self.transport.hu_dof_old[:] = self.u_dof_lstage[1]
                self.transport.hv_dof_old[:] = self.u_dof_lstage[2]
                self.transport.heta_dof_old[:] = self.u_dof_lstage[3]
                self.transport.hw_dof_old[:] = self.u_dof_lstage[4]
                self.transport.hbeta_dof_old[:] = self.u_dof_lstage[5]
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
                self.transport.heta_dof_old[:] = self.u_dof_last[3]
                self.transport.hw_dof_old[:] = self.u_dof_last[4]
                self.transport.hbeta_dof_old[:] = self.u_dof_last[5]
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
    The coefficients for the modified Serre-Green-Naghdi equations (dispersive SWEs)
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
                 LUMPED_MASS_MATRIX=1,
                 LINEAR_FRICTION=0,
                 mannings=0.,
                 forceStrongConditions=True,
                 constrainedDOFs=None,
                 gen_length=0.,
                 gen_start=0.,
                 abs_length=0.,
                 abs_start=0.,
                 waveConditions=None,
                 INTEGRATE_BY_PARTS_WALLS=1,
                 porosity=None,
                 drag_alpha=None,
                 drag_beta=None):
        # Canopy model, as in SW2DCV.Coefficients: porosity phi and
        # Darcy-Forchheimer drag -(alpha + beta |u|) h u per unit plan area.
        self.porosity = porosity
        self.drag_alpha = drag_alpha
        self.drag_beta = drag_beta
        self.porosity_dof = None
        self.drag_alpha_dof = None
        self.drag_beta_dof = None
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
        self.LUMPED_MASS_MATRIX = LUMPED_MASS_MATRIX
        self.LINEAR_FRICTION = LINEAR_FRICTION
        self.mannings = mannings
        self.modelIndex = modelIndex
        self.gen_length = gen_length
        self.gen_start = gen_start
        self.abs_length = abs_length
        self.abs_start = abs_start
        self.waveConditions = waveConditions
        mass = {}
        advection = {}
        diffusion = {}
        potential = {}
        reaction = {}
        hamiltonian = {}
        if nd == 2:
            variableNames = ['h', 'h_u', 'h_v', 'h_eta', 'h_w', 'h_beta']
            mass = {0: {0: 'linear'},
                    1: {0: 'linear', 1: 'linear'},
                    2: {0: 'linear', 2: 'linear'},
                    3: {0: 'linear', 3: 'linear'},
                    4: {0: 'linear', 4: 'linear'},
                    5: {0: 'linear', 5: 'linear'}}
            advection = {0: {0: 'nonlinear',
                             1: 'nonlinear',
                             2: 'nonlinear',
                             3: 'nonlinear',
                             4: 'nonlinear',
                             5: 'nonlinear'},
                         1: {0: 'nonlinear',
                             1: 'nonlinear',
                             2: 'nonlinear',
                             3: 'nonlinear',
                             4: 'nonlinear',
                             5: 'nonlinear'},
                         2: {0: 'nonlinear',
                             1: 'nonlinear',
                             2: 'nonlinear',
                             3: 'nonlinear',
                             4: 'nonlinear',
                             5: 'nonlinear'},
                         3: {0: 'nonlinear',
                             1: 'nonlinear',
                             2: 'nonlinear',
                             3: 'nonlinear',
                             4: 'nonlinear',
                             5: 'nonlinear'},
                         4: {0: 'nonlinear',
                             1: 'nonlinear',
                             2: 'nonlinear',
                             3: 'nonlinear',
                             4: 'nonlinear',
                             5: 'nonlinear'},
                         5: {0: 'nonlinear',
                             1: 'nonlinear',
                             2: 'nonlinear',
                             3: 'nonlinear',
                             4: 'nonlinear',
                             5: 'nonlinear'}}
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
                             6,  # Number of components
                             mass,
                             advection,
                             diffusion,
                             potential,
                             reaction,
                             hamiltonian,
                             variableNames,
                             sparseDiffusionTensors=sdInfo,
                             useSparseDiffusion=sd,
                             movingDomain=movingDomain)
            self.vectorComponents = [1, 2]

    def attachModels(self, modelList):
        self.model = modelList[self.modelIndex]
        self.model.q['velocity_porous'] = self.q_velocity_porous
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
            mesh.nodeArray[:,2] = self.b.dof                #if bathy is a function no need to pass subdomain info here
        
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
        self.q_velocity_porous = np.zeros(cq[('velocity', 0)].shape, 'd')
        # pass

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
            self.model.reflectingBoundaryIndex = np.where(
                np.isin(self.model.mesh.nodeMaterialTypes, 99))[0].tolist()
            self.model.reflectingBoundaryIndex = np.array(
                self.model.reflectingBoundaryIndex)

            # Init zoneElementMaterialType for relaxation zones
            self.model.relaxationZone_Elements = np.where(self.model.mesh.elementMaterialTypes > 1)[0].tolist()
            self.model.relaxationZone_Elements = np.array(self.model.relaxationZone_Elements)

            # get nodes that are in relaxation zones, how to make this faster?
            # if self.model.relaxationZone_Elements.any():
            #     self.model.relaxationZone_nodeIndex = []
            #     for i, eN in enumerate(self.model.relaxationZone_Elements):
            #         # get node array for local element
            #         nodes = self.model.u[0].femSpace.dofMap.l2g[eN]
            #         for node_i, node_value in enumerate(nodes):
            #             if node_value not in self.model.relaxationZone_nodeIndex:
            #                 self.model.relaxationZone_nodeIndex.append(node_value)
            #     self.model.relaxationZone_nodeIndex = np.array(self.model.relaxationZone_nodeIndex)
        #
        self.model.h_dof_old[:] = self.model.u[0].dof
        self.model.hu_dof_old[:] = self.model.u[1].dof
        self.model.hv_dof_old[:] = self.model.u[2].dof
        self.model.heta_dof_old[:] = self.model.u[3].dof
        self.model.hw_dof_old[:] = self.model.u[4].dof
        self.model.hbeta_dof_old[:] = self.model.u[5].dof

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
                 name='GN_SW2DCV',
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
                assert self.u[ci].femSpace.__class__.__name__ == self.u[
                    0].femSpace.__class__.__name__, "to reuse_test_trial_quad all femSpaces must be the same!"
        # Simplicial Mesh
        # assume the same mesh for  all components for now
        self.mesh = self.u[0].femSpace.mesh
        self.testSpace = testSpaceDict
        self.dirichletConditions = dofBoundaryConditionsDict
        # explicit Dirichlet conditions for now, no Dirichlet BC constraints
        self.dirichletNodeSetList = None
        self.coefficients = coefficients
        # cek hack? give coefficients a bathymetriy array
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
        self.nDOF_trial_element = [
            u_j.femSpace.max_nDOF_element for u_j in list(self.u.values())]
        self.nDOF_phi_trial_element = [
            phi_k.femSpace.max_nDOF_element for phi_k in list(self.phi.values())]
        self.n_phi_ip_element = [
            phi_k.femSpace.referenceFiniteElement.interpolationConditions.nQuadraturePoints for phi_k in list(self.phi.values())]
        self.nDOF_test_element = [
            femSpace.max_nDOF_element for femSpace in list(self.testSpace.values())]
        self.nFreeDOF_global = [dc.nFreeDOF_global for dc in list(
            self.dirichletConditions.values())]
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
        self.nQuadraturePoints_global = self.nQuadraturePoints_element * \
            self.mesh.nElements_global
        #
        # Repeat the same thing for the element boundary quadrature
        #
        (self.elementBoundaryQuadraturePoints,
         self.elementBoundaryQuadratureWeights,
         self.elementBoundaryQuadratureRuleIndeces) = Quadrature.buildUnion(elementBoundaryQuadratureDict)
        self.nElementBoundaryQuadraturePoints_elementBoundary = self.elementBoundaryQuadraturePoints.shape[
            0]
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
        # Old DOFs
        # NOTE (Mql): It is important to link h_dof_old by reference with u[0].dof (and so on).
        # This is because  I need the initial condition to be passed to them as well (before calling calculateResidual).
        # During preStep I change this and copy the values instead of keeping the reference.
        self.h_dof_old = None
        self.hu_dof_old = None
        self.hv_dof_old = None
        self.heta_dof_old = None
        self.hw_dof_old = None
        self.hbeta_dof_old = None

        # Vector for mass matrix
        self.check_positivity_water_height = True
        # mesh
        self.q['x'] = np.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element, 3), 'd')
        self.ebqe['x'] = np.zeros((self.mesh.nExteriorElementBoundaries_global,
                                   self.nElementBoundaryQuadraturePoints_elementBoundary, 3), 'd')
        self.ebq_global[('totalFlux', 0)] = np.zeros(
            (self.mesh.nElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebq_global[('velocityAverage', 0)] = np.zeros((self.mesh.nElementBoundaries_global,
                                                            self.nElementBoundaryQuadraturePoints_elementBoundary, self.nSpace_global), 'd')
        self.q[('dV_u', 0)] = (1.0/self.mesh.nElements_global) * \
            np.ones((self.mesh.nElements_global,
                     self.nQuadraturePoints_element), 'd')
        self.q[('dV_u', 1)] = (1.0/self.mesh.nElements_global) * \
            np.ones((self.mesh.nElements_global,
                     self.nQuadraturePoints_element), 'd')
        self.q[('dV_u', 2)] = (1.0/self.mesh.nElements_global) * \
            np.ones((self.mesh.nElements_global,
                     self.nQuadraturePoints_element), 'd')
        self.q['dV'] = self.q[('dV_u', 0)]
        self.q[('u', 0)] = np.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('u', 1)] = np.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('u', 2)] = np.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('u', 3)] = np.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('u', 4)] = np.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('u', 5)] = np.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('m', 0)] = self.q[('u', 0)]
        self.q[('m', 1)] = self.q[('u', 1)]
        self.q[('m', 2)] = self.q[('u', 2)]
        self.q[('m', 3)] = self.q[('u', 3)]
        self.q[('m', 4)] = self.q[('u', 4)]
        self.q[('m', 5)] = self.q[('u', 5)]
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
        self.q[('f', 0)] = np.zeros((self.mesh.nElements_global,
                                     self.nQuadraturePoints_element, self.nSpace_global), 'd')
        self.q[('velocity', 0)] = np.zeros((self.mesh.nElements_global,
                                            self.nQuadraturePoints_element, self.nSpace_global), 'd')
        self.q[('cfl', 0)] = np.zeros(
            (self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.ebqe[('u', 0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,
                                        self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('u', 1)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,
                                        self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('u', 2)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,
                                        self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('u', 3)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,
                                        self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('u', 4)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,
                                        self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('u', 5)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,
                                        self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('advectiveFlux_bc_flag', 0)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'i')
        self.ebqe[('advectiveFlux_bc_flag', 1)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'i')
        self.ebqe[('advectiveFlux_bc_flag', 2)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'i')
        self.ebqe[('advectiveFlux_bc_flag', 3)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'i')
        self.ebqe[('advectiveFlux_bc_flag', 4)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'i')
        self.ebqe[('advectiveFlux_bc_flag', 5)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'i')
        self.ebqe[('diffusiveFlux_bc_flag', 1, 1)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'i')
        self.ebqe[('diffusiveFlux_bc_flag', 2, 2)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'i')
        self.ebqe[('diffusiveFlux_bc_flag', 3, 3)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'i')
        self.ebqe[('diffusiveFlux_bc_flag', 4, 4)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'i')
        self.ebqe[('diffusiveFlux_bc_flag', 5, 5)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'i')
        self.ebqe[('advectiveFlux_bc', 0)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('advectiveFlux_bc', 1)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('advectiveFlux_bc', 2)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('advectiveFlux_bc', 3)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('advectiveFlux_bc', 4)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('advectiveFlux_bc', 5)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('diffusiveFlux_bc', 1, 1)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('penalty')] = np.zeros((self.mesh.nExteriorElementBoundaries_global,
                                           self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('diffusiveFlux_bc', 2, 2)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('diffusiveFlux_bc', 3, 3)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('diffusiveFlux_bc', 4, 4)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('diffusiveFlux_bc', 5, 5)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('velocity', 0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,
                                               self.nElementBoundaryQuadraturePoints_elementBoundary, self.nSpace_global), 'd')
        self.ebqe[('velocity', 1)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,
                                               self.nElementBoundaryQuadraturePoints_elementBoundary, self.nSpace_global), 'd')
        self.ebqe[('velocity', 2)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,
                                               self.nElementBoundaryQuadraturePoints_elementBoundary, self.nSpace_global), 'd')
        self.points_elementBoundaryQuadrature = set()
        self.scalars_elementBoundaryQuadrature = set(
            [('u', ci) for ci in range(self.nc)])
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
        logEvent(
            "Interpolation points for nonlinear diffusion potential (phi_ip)", level=9)
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
            self.inflowBoundaryBC[cj] = np.zeros(
                (self.mesh.nExteriorElementBoundaries_global,), 'i')
            self.inflowBoundaryBC_values[cj] = np.zeros(
                (self.mesh.nExteriorElementBoundaries_global, self.nDOF_trial_element[cj]), 'd')
            self.inflowFlux[cj] = np.zeros(
                (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
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
        logEvent(memory("inflowBC, internalNodes,updateLocal2Global",
                        "OneLevelTransport"), level=4)
        self.timeIntegration = TimeIntegrationClass(self)

        if options is not None:
            self.timeIntegration.setFromOptions(options)
        logEvent(memory("TimeIntegration", "OneLevelTransport"), level=4)
        logEvent("Calculating numerical quadrature formulas", 2)
        self.calculateQuadrature()
        self.setupFieldStrides()

        # initialize some things
        self.eps = None
        self.h0_max = None
        self.hEps = None  # for 1/h regularization
        self.ML = None  # lumped mass matrix
        self.inverse_mesh = None  # used for relaxation
        self.MC_global = None  # consistent mass matrix
        ### Global C Matrices (mql)
        self.cterm_global = None
        self.cterm_transpose_global = None
        ## for EV
        self.dij_small = None
        self.entropy = None
        self.global_entropy_residual = None
        ### For high order/convex limiting
        self.RHS_high_h = None
        self.RHS_high_hu = None
        self.RHS_high_hv = None
        self.RHS_high_heta = None
        self.RHS_high_hw = None
        self.RHS_high_hbeta = None
        self.size_of_domain = None
        self.delta_Sqd_h = None
        self.delta_Sqd_heta = None
        # low-order solution using bar states
        self.hLow = None
        self.huLow = None
        self.hvLow = None
        self.hetaLow = None
        self.hwLow = None
        self.hbetaLow = None
        # for bounds
        self.h_min = None
        self.h_max = None
        self.heta_min = None
        self.heta_max = None
        self.kin_max = None
        self.KE_tiny = None
        # four sources
        self.SourceTerm_h = None
        self.SourceTerm_hu = None
        self.SourceTerm_hv = None
        self.SourceTerm_heta = None
        self.SourceTerm_hw = None
        self.SourceTerm_hbeta = None
        self.extendedSourceTerm_hu = None
        self.extendedSourceTerm_hv = None
        ## NORMALS
        self.COMPUTE_NORMALS = 1
        self.normalx = None
        self.normaly = None
        self.boundaryIndex = None
        self.reflectingBoundaryConditions = False
        self.reflectingBoundaryIndex = None
        self.wall_dof = None
        # for relaxation zones node material, initialize index array with hack
        # self.relaxationZone_nodeIndex = np.array([-1, -1])
        self.h_wave = None
        self.h_u_wave = None
        self.h_v_wave = None
        self.h_eta_wave = None
        self.h_w_wave = None
        self.h_beta_wave = None

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
        self.velocityPostProcessor = PostProcessingTools.VelocityPostProcessingChooser(
            self)
        logEvent(memory("velocity postprocessor", "OneLevelTransport"), level=4)
        # helper for writing out data storage
        from proteus import Archiver
        self.elementQuadratureDictionaryWriter = Archiver.XdmfWriter()
        self.elementBoundaryQuadratureDictionaryWriter = Archiver.XdmfWriter()
        self.exteriorElementBoundaryQuadratureDictionaryWriter = Archiver.XdmfWriter()
        for ci, fbcObject in list(self.fluxBoundaryConditionsObjectsDict.items()):
            self.ebqe[('advectiveFlux_bc_flag', ci)] = np.zeros(
                self.ebqe[('advectiveFlux_bc', ci)].shape, 'i')
            for t, g in list(fbcObject.advectiveFluxBoundaryConditionsDict.items()):
                if ci in self.coefficients.advection:
                    self.ebqe[('advectiveFlux_bc', ci)][t[0], t[1]] = g(
                        self.ebqe[('x')][t[0], t[1]], self.timeIntegration.t)
                    self.ebqe[('advectiveFlux_bc_flag', ci)][t[0], t[1]] = 1
            for ck, diffusiveFluxBoundaryConditionsDict in list(fbcObject.diffusiveFluxBoundaryConditionsDictDict.items()):
                self.ebqe[('diffusiveFlux_bc_flag', ck, ci)] = np.zeros(
                    self.ebqe[('diffusiveFlux_bc', ck, ci)].shape, 'i')
                for t, g in list(diffusiveFluxBoundaryConditionsDict.items()):
                    self.ebqe[('diffusiveFlux_bc', ck, ci)][t[0], t[1]] = g(
                        self.ebqe[('x')][t[0], t[1]], self.timeIntegration.t)
                    self.ebqe[('diffusiveFlux_bc_flag', ck, ci)
                              ][t[0], t[1]] = 1
        # self.numericalFlux.setDirichletValues(self.ebqe)
        if self.movingDomain:
            self.MOVING_DOMAIN = 1.0
        else:
            self.MOVING_DOMAIN = 0.0
        # cek hack
        self.movingDomain = False
        self.MOVING_DOMAIN = 0.0
        if self.mesh.nodeVelocityArray is None:
            self.mesh.nodeVelocityArray = np.zeros(
                self.mesh.nodeArray.shape, 'd')
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
        self.dsw_2d = cGN_SW2DCV_base(self.nSpace_global,
                                      self.nQuadraturePoints_element,
                                      self.u[0].femSpace.elementMaps.localFunctionSpace.dim,
                                      self.u[0].femSpace.referenceFiniteElement.localFunctionSpace.dim,
                                      self.testSpace[0].referenceFiniteElement.localFunctionSpace.dim,
                                      self.nElementBoundaryQuadraturePoints_elementBoundary,
                                      compKernelFlag)

        self.calculateResidual = self.dsw_2d.calculateResidual

        # define function to compute preStep
        self.calculatePreStep = self.dsw_2d.calculatePreStep

        # define function to compute entropy viscosity residual
        self.calculateEV = self.dsw_2d.calculateEV

        # define function to compute high order rhs
        self.calculateBoundsAndHighOrderRHS = self.dsw_2d.calculateBoundsAndHighOrderRHS

        if (self.coefficients.LUMPED_MASS_MATRIX):
            self.calculateJacobian = self.dsw_2d.calculateLumpedMassMatrix
        else:
            self.calculateJacobian = self.dsw_2d.calculateMassMatrix
        #
        self.dofsXCoord = None
        self.dofsYCoord = None
        self.constrainedDOFsIndices = None
        self.dataStructuresInitialized = False

        # parallel vectors that don't change in time #
        self.par_bathymetry = None
        self.par_normalx = None
        self.par_normaly = None
        self.par_ML = None
        self.FCT_all = None  # FCT matrices kept between limiting passes
        # low order solution #
        self.par_hLow = None
        self.par_huLow = None
        self.par_hvLow = None
        self.par_hetaLow = None
        # for bounds #
        self.par_h_min = None
        self.par_h_max = None
        self.par_heta_min = None
        self.par_heta_max = None
        self.par_kin_max = None
        # for parallel entropy residual
        self.par_global_entropy_residual = None

    def FCTStep(self):
        # NOTE: this function is meant to be called within the solver
        comm = Comm.get()

        rowptr, colind, MassMatrix = self.MC_global.getCSRrepresentation()
        # Extract hnp1 from global solution u
        index = list(range(0, len(self.timeIntegration.u)))
        hIndex = index[0::6]
        huIndex = index[1::6]
        hvIndex = index[2::6]
        hetaIndex = index[3::6]
        hwIndex = index[4::6]
        hbetaIndex = index[5::6]
        # create limited solution
        limited_hnp1 = np.zeros(self.h_dof_old.shape)
        limited_hunp1 = np.zeros(self.h_dof_old.shape)
        limited_hvnp1 = np.zeros(self.h_dof_old.shape)
        limited_hetanp1 = np.zeros(self.h_dof_old.shape)
        limited_hwnp1 = np.zeros(self.h_dof_old.shape)
        limited_hbetanp1 = np.zeros(self.h_dof_old.shape)

        self.KE_tiny = self.hEps * comm.globalMax(np.amax(self.kin_max))

        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["numDOFs"] = len(rowptr) - 1
        argsDict["csrRowIndeces_DofLoops"] = rowptr
        argsDict["csrColumnOffsets_DofLoops"] = colind
        argsDict["MassMatrix"] = MassMatrix
        argsDict["lumped_mass_matrix"] = self.ML
        argsDict["dt"] = self.timeIntegration.dt
        argsDict["h_old"] = self.h_dof_old
        argsDict["hu_old"] = self.hu_dof_old
        argsDict["hv_old"] = self.hv_dof_old
        argsDict["heta_old"] = self.heta_dof_old
        argsDict["hw_old"] = self.hw_dof_old
        argsDict["hbeta_old"] = self.hbeta_dof_old
        argsDict["b_dof"] = self.coefficients.b.dof
        ###
        argsDict["limited_hnp1"] = limited_hnp1
        argsDict["limited_hunp1"] = limited_hunp1
        argsDict["limited_hvnp1"] = limited_hvnp1
        argsDict["limited_hetanp1"] = limited_hetanp1
        argsDict["limited_hwnp1"] = limited_hwnp1
        argsDict["limited_hbetanp1"] = limited_hbetanp1
        argsDict["hEps"] = self.hEps
        argsDict["hLow"] = self.hLow
        argsDict["huLow"] = self.huLow
        argsDict["hvLow"] = self.hvLow
        argsDict["hetaLow"] = self.hetaLow
        argsDict["hwLow"] = self.hwLow
        argsDict["hbetaLow"] = self.hbetaLow
        argsDict["h_min"] = self.h_min
        argsDict["h_max"] = self.h_max
        argsDict["heta_min"] = self.heta_min
        argsDict["heta_max"] = self.heta_max
        argsDict["kin_max"] = self.kin_max
        argsDict["KE_tiny"] = self.KE_tiny
        argsDict["SourceTerm_h"] = self.SourceTerm_h
        argsDict["SourceTerm_hu"] = self.SourceTerm_hu
        argsDict["SourceTerm_hv"] = self.SourceTerm_hv
        argsDict["SourceTerm_heta"] = self.SourceTerm_heta
        argsDict["SourceTerm_hw"] = self.SourceTerm_hw
        argsDict["SourceTerm_hbeta"] = self.SourceTerm_hbeta
        argsDict["global_entropy_residual"] = self.global_entropy_residual
        argsDict["Cx"] = self.Cx
        argsDict["Cy"] = self.Cy
        argsDict["CTx"] = self.CTx
        argsDict["CTy"] = self.CTy
        argsDict["RHS_high_h"] = self.RHS_high_h
        argsDict["RHS_high_hu"] = self.RHS_high_hu
        argsDict["RHS_high_hv"] = self.RHS_high_hv
        argsDict["RHS_high_heta"] = self.RHS_high_heta
        argsDict["RHS_high_hw"] = self.RHS_high_hw
        argsDict["RHS_high_hbeta"] = self.RHS_high_hbeta
        argsDict["extendedSourceTerm_hu"] = self.extendedSourceTerm_hu
        argsDict["extendedSourceTerm_hv"] = self.extendedSourceTerm_hv
        argsDict["thetaj_inv"] = self.thetaj_inv
        argsDict["wall_dof"] = self.wall_dof
        argsDict["porosity_dof"] = self.coefficients.porosity_dof
        argsDict["g"] = self.coefficients.g
        argsDict["inverse_mesh"] = self.inverse_mesh
        # Two limiting passes, one per kernel call; the second pass reads the
        # neighbours' low-order states, so exchange them with the ghost DOFs
        # in between (a ghost row's own update uses an incomplete stencil).
        n_limit_iter = 2
        if self.FCT_all is None or self.FCT_all.size != 6 * self.Cx.size:
            self.FCT_all = np.zeros(6 * self.Cx.size, 'd')
        argsDict["FCT_all"] = self.FCT_all
        argsDict["n_limit_iter"] = n_limit_iter
        for limit_iter in range(n_limit_iter):
            argsDict["limit_iter"] = limit_iter
            self.dsw_2d.convexLimiting(argsDict)
            if limit_iter < n_limit_iter - 1:
                self.par_hLow.scatter_forward_insert()
                self.par_huLow.scatter_forward_insert()
                self.par_hvLow.scatter_forward_insert()
                self.par_hetaLow.scatter_forward_insert()

        # Pass the post processed hnp1 solution to global solution u
        self.timeIntegration.u[hIndex] = limited_hnp1
        self.timeIntegration.u[huIndex] = limited_hunp1
        self.timeIntegration.u[hvIndex] = limited_hvnp1
        self.timeIntegration.u[hetaIndex] = limited_hetanp1
        self.timeIntegration.u[hwIndex] = limited_hwnp1
        self.timeIntegration.u[hbetaIndex] = limited_hbetanp1

    def computePreStep(self):
        # Arguments
        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["g"] = self.coefficients.g
        argsDict["h_dof_old"] = self.h_dof_old
        argsDict["hu_dof_old"] = self.hu_dof_old
        argsDict["hv_dof_old"] = self.hv_dof_old
        argsDict["heta_dof_old"] = self.heta_dof_old
        argsDict["hEps"] = self.hEps
        argsDict["numDOFsPerEqn"] = self.numDOFsPerEqn
        argsDict["csrRowIndeces_DofLoops"] = self.rowptr_cMatrix
        argsDict["csrColumnOffsets_DofLoops"] = self.colind_cMatrix
        argsDict["entropy"] = self.entropy
        argsDict["delta_Sqd_h"] = self.delta_Sqd_h
        argsDict["delta_Sqd_heta"] = self.delta_Sqd_heta
        argsDict["thetaj_inv"] = self.thetaj_inv
        argsDict["dij_small"] = 0.
        argsDict["h0_max"] = self.h0_max
        argsDict["Cx"] = self.Cx
        argsDict["Cy"] = self.Cy

        # call PreStep function
        self.dsw_2d.calculatePreStep(argsDict)

        # save things
        self.dij_small = globalMax(argsDict.dscalar["dij_small"])
    #

    def computeEV(self):
        # Arguments
        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["g"] = self.coefficients.g
        argsDict["h_dof_old"] = self.h_dof_old
        argsDict["hu_dof_old"] = self.hu_dof_old
        argsDict["hv_dof_old"] = self.hv_dof_old
        argsDict["Cx"] = self.Cx
        argsDict["Cy"] = self.Cy
        argsDict["CTx"] = self.CTx
        argsDict["CTy"] = self.CTy
        argsDict["numDOFsPerEqn"] = self.numDOFsPerEqn
        argsDict["csrRowIndeces_DofLoops"] = self.rowptr_cMatrix
        argsDict["csrColumnOffsets_DofLoops"] = self.colind_cMatrix
        argsDict["lumped_mass_matrix"] = self.ML
        argsDict["hEps"] = self.hEps
        argsDict["global_entropy_residual"] = self.global_entropy_residual
        argsDict["entropy"] = self.entropy
        argsDict["h0_max"] = self.h0_max

        # compute entropy residual
        self.dsw_2d.calculateEV(argsDict)
    #

    def compute_waves(self):
        x = self.mesh.nodeArray[:, 0]
        y = self.mesh.nodeArray[:, 1]
        X = [x,y]
        t = self.timeIntegration.t
        self.h_wave = self.coefficients.waveConditions['h'](X, t)
        self.h_u_wave = self.coefficients.waveConditions['h_u'](X, t)
        self.h_v_wave = self.coefficients.waveConditions['h_v'](X, t)
        self.h_eta_wave = self.coefficients.waveConditions['h_eta'](X, t)
        self.h_w_wave = self.coefficients.waveConditions['h_w'](X, t)
        self.h_beta_wave = self.coefficients.waveConditions['h_beta'](X, t)


    def computeBoundsAndRhsHigh(self):
        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["g"] = self.coefficients.g
        argsDict["h_dof_old"] = self.h_dof_old
        argsDict["hu_dof_old"] = self.hu_dof_old
        argsDict["hv_dof_old"] = self.hv_dof_old
        argsDict["heta_dof_old"] = self.heta_dof_old
        argsDict["hw_dof_old"] = self.hw_dof_old
        argsDict["hbeta_dof_old"] = self.hbeta_dof_old
        argsDict["b_dof"] = self.coefficients.b.dof
        argsDict["Cx"] = self.Cx
        argsDict["Cy"] = self.Cy
        argsDict["CTx"] = self.CTx
        argsDict["CTy"] = self.CTy
        argsDict["numDOFsPerEqn"] = self.numDOFsPerEqn
        argsDict["csrRowIndeces_DofLoops"] = self.rowptr_cMatrix
        argsDict["csrColumnOffsets_DofLoops"] = self.colind_cMatrix
        argsDict["lumped_mass_matrix"] = self.ML
        argsDict["hEps"] = self.hEps
        argsDict["SourceTerm_h"] = self.SourceTerm_h
        argsDict["SourceTerm_hu"] = self.SourceTerm_hu
        argsDict["SourceTerm_hv"] = self.SourceTerm_hv
        argsDict["SourceTerm_heta"] = self.SourceTerm_heta
        argsDict["SourceTerm_hw"] = self.SourceTerm_hw
        argsDict["SourceTerm_hbeta"] = self.SourceTerm_hbeta
        argsDict["dt"] = self.timeIntegration.dt
        argsDict["mannings"] = float(self.coefficients.mannings)
        argsDict["lstage"] = self.timeIntegration.lstage
        argsDict["global_entropy_residual"] = self.global_entropy_residual
        argsDict["dij_small"] = self.dij_small
        argsDict["hLow"] = self.hLow
        argsDict["huLow"] = self.huLow
        argsDict["hvLow"] = self.hvLow
        argsDict["hetaLow"] = self.hetaLow
        argsDict["hwLow"] = self.hwLow
        argsDict["hbetaLow"] = self.hbetaLow
        argsDict["h_min"] = self.h_min
        argsDict["h_max"] = self.h_max
        argsDict["heta_min"] = self.heta_min
        argsDict["heta_max"] = self.heta_max
        argsDict["kin_max"] = self.kin_max
        argsDict["x_values"] = self.dofsXCoord
        argsDict["inverse_mesh"] = self.inverse_mesh
        argsDict["h0_max"] = self.h0_max
        argsDict["RHS_high_h"] = self.RHS_high_h
        argsDict["RHS_high_hu"] = self.RHS_high_hu
        argsDict["RHS_high_hv"] = self.RHS_high_hv
        argsDict["RHS_high_heta"] = self.RHS_high_heta
        argsDict["RHS_high_hw"] = self.RHS_high_hw
        argsDict["RHS_high_hbeta"] = self.RHS_high_hbeta
        argsDict["extendedSourceTerm_hu"] = self.extendedSourceTerm_hu
        argsDict["extendedSourceTerm_hv"] = self.extendedSourceTerm_hv
        argsDict["size_of_domain"] = self.size_of_domain
        argsDict["delta_Sqd_h"] = self.delta_Sqd_h
        argsDict["delta_Sqd_heta"] = self.delta_Sqd_heta
        argsDict["gen_length"] = float(self.coefficients.gen_length)
        argsDict["gen_start"] = float(self.coefficients.gen_start)
        argsDict["abs_length"] = float(self.coefficients.abs_length)
        argsDict["abs_start"] = float(self.coefficients.abs_start)
        argsDict["h_wave"] = self.h_wave
        argsDict["h_u_wave"] = self.h_u_wave
        argsDict["h_v_wave"] = self.h_v_wave
        argsDict["h_eta_wave"] = self.h_eta_wave
        argsDict["h_w_wave"] = self.h_w_wave
        argsDict["h_beta_wave"] = self.h_beta_wave
        argsDict["wall_dof"] = self.wall_dof
        argsDict["porosity_dof"] = self.coefficients.porosity_dof
        argsDict["drag_alpha_dof"] = self.coefficients.drag_alpha_dof
        argsDict["drag_beta_dof"] = self.coefficients.drag_beta_dof

        # function
        self.dsw_2d.calculateBoundsAndHighOrderRHS(argsDict)

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
        nnz_cMatrix = nnz // 6 // 6  # This is always true for the relaxed dSSV model in 2D
        nzval_cMatrix = np.zeros(nnz_cMatrix)
        rowptr_cMatrix = np.zeros(self.u[0].dof.size + 1, 'i')
        colind_cMatrix = np.zeros(nnz_cMatrix, 'i')
        # fill vector rowptr_cMatrix
        for i in range(1, rowptr_cMatrix.size):
            rowptr_cMatrix[i] = rowptr_cMatrix[i - 1] + \
                (rowptr[6 * (i - 1) + 1] - rowptr[6 * (i - 1)])//6

        # fill vector colind_cMatrix
        i_cMatrix = 0  # ith row of cMatrix
        # 0 to total num of DOFs (i.e. num of rows of jacobian)
        for i in range(rowptr.size - 1):
            if (i % 6 == 0):  # Just consider the rows related to the h variable
                for j, offset in enumerate(range(rowptr[i], rowptr[i + 1])):
                    offset_cMatrix = list(
                        range(rowptr_cMatrix[i_cMatrix], rowptr_cMatrix[i_cMatrix + 1]))
                    if (j % 6 == 0):
                        colind_cMatrix[offset_cMatrix[j//6]] = colind[offset]//6
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
        self.u[0].femSpace.getBasisValues(
            self.elementQuadraturePoints, self.q[('w', 0)])
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
        # assume a linear mass term, weighted by the porosity (storage phi U)
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
                                                                  self.csrRowIndeces[(
                                                                      0, 0)] // 6 // 6,
                                                                  self.csrColumnOffsets[(
                                                                      0, 0)] // 6,
                                                                  elementMassMatrix,
                                                                  self.MC_global)

        diamD2 = np.sum(self.q['abs(det(J))'][:]
                        * self.elementQuadratureWeights[('u', 0)])
        self.ML = np.zeros((self.nFreeDOF_global[0],), 'd')
        self.inverse_mesh = np.zeros((self.nFreeDOF_global[0],), 'd')
        for i in range(self.nFreeDOF_global[0]):
            self.ML[i] = self.MC_a[rowptr_cMatrix[i]
                :rowptr_cMatrix[i + 1]].sum()
            self.inverse_mesh[i] = 1. / np.sqrt(self.ML[i])
        # the mesh size comes from the plan-area lumped mass, not the
        # porosity-weighted one: area_e / 3 per vertex for P1
        area_e = np.sum(self.q['abs(det(J))'] * self.elementQuadratureWeights[('u', 0)], axis=1)
        ML_area = np.zeros(self.mesh.nNodes_global, 'd')
        np.add.at(ML_area, self.mesh.elementNodesArray.ravel(),
                  np.repeat(area_e / 3., self.mesh.elementNodesArray.shape[1]))
        nfree = self.nFreeDOF_global[0]
        self.inverse_mesh[:nfree] = 1. / np.sqrt(ML_area[:nfree])
        # ML_i covers only the part of a full patch the node sees: half at a
        # straight boundary, a quarter at a right-angle corner. The mesh size
        # there is not smaller, so rescale ML_i by 2 pi / (interior angle at
        # node i); interior nodes (angle 2 pi) are unchanged.
        if self.nSpace_global == 2:
            nodes = self.mesh.nodeArray[:, :2]
            angle = np.zeros(self.mesh.nNodes_global, 'd')
            for e in self.mesh.elementNodesArray:
                P = nodes[e]
                for k in range(3):
                    a1 = P[(k + 1) % 3] - P[k]
                    a2 = P[(k + 2) % 3] - P[k]
                    angle[e[k]] += np.arccos(np.clip(
                        np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2)), -1., 1.))
            # only exterior-boundary nodes (not partition boundaries)
            ebN = self.mesh.exteriorElementBoundariesArray
            bnodes = np.unique(self.mesh.elementBoundaryNodesArray[ebN].ravel())
            bnodes = bnodes[(bnodes < self.nFreeDOF_global[0]) & (angle[bnodes] > 0.)]
            self.inverse_mesh[bnodes] = 1. / np.sqrt(
                ML_area[bnodes] * 2. * np.pi / angle[bnodes])
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
                                                                self.q[(
                                                                    'grad(w)*dV_f', 0)],
                                                                self.q[(
                                                                    'w', 0)],
                                                                self.cterm[d])  # int[(di*grad(wj))*wi*dV]
            cfemIntegrals.updateGlobalJacobianFromElementJacobian_CSR(self.l2g[0]['nFreeDOF'],
                                                                      self.l2g[0]['freeLocal'],
                                                                      self.l2g[0]['nFreeDOF'],
                                                                      self.l2g[0]['freeLocal'],
                                                                      self.csrRowIndeces[(
                                                                          0, 0)] // 6 // 6,
                                                                      self.csrColumnOffsets[(
                                                                          0, 0)] // 6,
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
            cfemIntegrals.zeroJacobian_CSR(
                nnz_cMatrix, self.cterm_global_transpose[d])
            di[:] = 0.0
            di[..., d] = -1.0
            cfemIntegrals.updateAdvectionJacobian_weak_lowmem(di,
                                                              self.q[('w', 0)],
                                                              self.q[(
                                                                  'grad(w)*dV_f', 0)],
                                                              self.cterm_transpose[d])  # -int[(-di*grad(wi))*wj*dV]

            cfemIntegrals.updateGlobalJacobianFromElementJacobian_CSR(self.l2g[0]['nFreeDOF'],
                                                                      self.l2g[0]['freeLocal'],
                                                                      self.l2g[0]['nFreeDOF'],
                                                                      self.l2g[0]['freeLocal'],
                                                                      self.csrRowIndeces[(
                                                                          0, 0)] // 6 // 6,
                                                                      self.csrColumnOffsets[(
                                                                          0, 0)] // 6,
                                                                      self.cterm_transpose[d],
                                                                      self.cterm_global_transpose[d])
        #
        self.rowptr_cMatrix, self.colind_cMatrix, self.Cx = self.cterm_global[0].getCSRrepresentation(
        )
        rowptr_cMatrix, colind_cMatrix, self.Cy = self.cterm_global[1].getCSRrepresentation(
        )
        rowptr_cMatrix, colind_cMatrix, self.CTx = self.cterm_global_transpose[0].getCSRrepresentation(
        )
        rowptr_cMatrix, colind_cMatrix, self.CTy = self.cterm_global_transpose[1].getCSRrepresentation(
        )
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
            (h, hu, hv, heta, hw, hbeta) = self.coefficients.constrainedDOFs[1](x, y, self.timeIntegration.t,
                                                                                self.u[0].dof[i],
                                                                                self.u[1].dof[i],
                                                                                self.u[2].dof[i],
                                                                                self.u[3].dof[i],
                                                                                self.u[4].dof[i],
                                                                                self.u[5].dof[i])
            if h is not None:
                self.u[0].dof[i] = h
            if hu is not None:
                self.u[1].dof[i] = hu
            if hv is not None:
                self.u[2].dof[i] = hv
            if heta is not None:
                self.u[3].dof[i] = heta
            if hw is not None:
                self.u[4].dof[i] = hw
            if hbeta is not None:
                self.u[5].dof[i] = hbeta
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
        """1 on the DOFs whose rows treat the boundary as a no-flux wall."""
        self.wall_dof = np.zeros(self.u[0].dof.shape, 'd')
        if not self.coefficients.INTEGRATE_BY_PARTS_WALLS:
            return
        if self.reflectingBoundaryConditions:
            ebN = self.mesh.exteriorElementBoundariesArray
            nodes = np.unique(self.mesh.elementBoundaryNodesArray[ebN].ravel())
            self.wall_dof[nodes] = 1.0
        else:
            self.wall_dof[self.mesh.nodeMaterialTypes == 99] = 1.0
    #

    def initDataStructures(self):
        comm = Comm.get()
        self.setWallDOFs()

        # Plan area of the whole domain (bounds relaxation uses m_i / area).
        # In parallel each subdomain also holds overlap elements, so sum only
        # the owned elements and reduce across ranks.
        area_e = np.sum(self.q['abs(det(J))'] * self.elementQuadratureWeights[('u', 0)], axis=1)
        self.size_of_domain = comm.globalSum(area_e[:self.mesh.nElements_owned].sum())

        # old vectors
        self.h_dof_old = np.copy(self.u[0].dof)
        self.hu_dof_old = np.copy(self.u[1].dof)
        self.hv_dof_old = np.copy(self.u[2].dof)
        self.heta_dof_old = np.copy(self.u[3].dof)
        self.hw_dof_old = np.copy(self.u[4].dof)
        self.hbeta_dof_old = np.copy(self.u[5].dof)

        # hEps
        self.eps = 1E-5
        self.h0_max = comm.globalMax(self.u[0].dof.max())
        self.hEps = self.h0_max * self.eps

        # normal vectors
        self.normalx = np.zeros(self.u[0].dof.shape, 'd')
        self.normaly = np.zeros(self.u[0].dof.shape, 'd')
        # quantDOFs
        self.quantDOFs = np.zeros(self.u[0].dof.shape, 'd')
        # boundary Index: I do this in preStep since I need normalx and normaly to be initialized first
        # get coordinates of DOFs
        self.getDOFsCoord()
        #
        self.RHS_high_h = np.zeros(self.u[0].dof.shape, 'd')
        self.RHS_high_hu = np.zeros(self.u[0].dof.shape, 'd')
        self.RHS_high_hv = np.zeros(self.u[0].dof.shape, 'd')
        self.RHS_high_heta = np.zeros(self.u[0].dof.shape, 'd')
        self.RHS_high_hw = np.zeros(self.u[0].dof.shape, 'd')
        self.RHS_high_hbeta = np.zeros(self.u[0].dof.shape, 'd')
        self.delta_Sqd_h = np.zeros(self.u[0].dof.shape, 'd')
        self.delta_Sqd_heta = np.zeros(self.u[0].dof.shape, 'd')
        # These should be defined using bar states
        self.hLow = np.zeros(self.u[0].dof.shape, 'd')
        self.huLow = np.zeros(self.u[0].dof.shape, 'd')
        self.hvLow = np.zeros(self.u[0].dof.shape, 'd')
        self.hetaLow = np.zeros(self.u[0].dof.shape, 'd')
        self.hwLow = np.zeros(self.u[0].dof.shape, 'd')
        self.hbetaLow = np.zeros(self.u[0].dof.shape, 'd')
        #
        self.h_min = np.zeros(self.u[0].dof.shape, 'd')
        self.h_max = np.zeros(self.u[0].dof.shape, 'd')
        self.heta_min = np.zeros(self.u[0].dof.shape, 'd')
        self.heta_max = np.zeros(self.u[0].dof.shape, 'd')
        self.kin_max = np.zeros(self.u[0].dof.shape, 'd')
        #
        self.SourceTerm_h = np.zeros(self.u[0].dof.shape, 'd')
        self.SourceTerm_hu = np.zeros(self.u[0].dof.shape, 'd')
        self.SourceTerm_hv = np.zeros(self.u[0].dof.shape, 'd')
        self.SourceTerm_heta = np.zeros(self.u[0].dof.shape, 'd')
        self.SourceTerm_hw = np.zeros(self.u[0].dof.shape, 'd')
        self.SourceTerm_hbeta = np.zeros(self.u[0].dof.shape, 'd')
        self.extendedSourceTerm_hu = np.zeros(self.u[0].dof.shape, 'd')
        self.extendedSourceTerm_hv = np.zeros(self.u[0].dof.shape, 'd')
        #
        self.h_wave = np.zeros(self.u[0].dof.shape, 'd')
        self.h_u_wave = np.zeros(self.u[0].dof.shape, 'd')
        self.h_v_wave = np.zeros(self.u[0].dof.shape, 'd')
        self.h_eta_wave = np.zeros(self.u[0].dof.shape, 'd')
        self.h_w_wave = np.zeros(self.u[0].dof.shape, 'd')
        self.h_beta_wave = np.zeros(self.u[0].dof.shape, 'd')
        #
        self.entropy = np.zeros(self.u[0].dof.shape, 'd')
        self.global_entropy_residual = np.zeros(self.u[0].dof.shape, 'd')
        self.dij_small = 0.0
        self.thetaj_inv = np.zeros(self.u[0].dof.shape, 'd')

        # PARALLEL VECTORS #
        n = self.u[0].par_dof.dim_proc
        N = self.u[0].femSpace.dofMap.nDOF_all_processes
        nghosts = self.u[0].par_dof.nghosts
        subdomain2global = self.u[0].femSpace.dofMap.subdomain2global
        #
        self.par_RHS_high_h = LAT.ParVec_petsc4py(self.RHS_high_h,
                                                  bs=1,
                                                  n=n, N=N, nghosts=nghosts,
                                                  subdomain2global=subdomain2global)
        self.par_RHS_high_hu = LAT.ParVec_petsc4py(self.RHS_high_hu,
                                                   bs=1,
                                                   n=n, N=N, nghosts=nghosts,
                                                   subdomain2global=subdomain2global)
        self.par_RHS_high_hv = LAT.ParVec_petsc4py(self.RHS_high_hv,
                                                   bs=1,
                                                   n=n, N=N, nghosts=nghosts,
                                                   subdomain2global=subdomain2global)
        self.par_RHS_high_heta = LAT.ParVec_petsc4py(self.RHS_high_heta,
                                                     bs=1,
                                                     n=n, N=N, nghosts=nghosts,
                                                     subdomain2global=subdomain2global)
        self.par_RHS_high_hw = LAT.ParVec_petsc4py(self.RHS_high_hw,
                                                   bs=1,
                                                   n=n, N=N, nghosts=nghosts,
                                                   subdomain2global=subdomain2global)
        self.par_RHS_high_hbeta = LAT.ParVec_petsc4py(self.RHS_high_hbeta,
                                                      bs=1,
                                                      n=n, N=N, nghosts=nghosts,
                                                      subdomain2global=subdomain2global)
        self.par_delta_Sqd_h = LAT.ParVec_petsc4py(self.delta_Sqd_h,
                                                   bs=1,
                                                   n=n, N=N, nghosts=nghosts,
                                                   subdomain2global=subdomain2global)
        self.par_delta_Sqd_heta = LAT.ParVec_petsc4py(self.delta_Sqd_heta,
                                                      bs=1,
                                                      n=n, N=N, nghosts=nghosts,
                                                      subdomain2global=subdomain2global)
        # for bounds
        self.par_h_min = LAT.ParVec_petsc4py(self.h_min,
                                             bs=1,
                                             n=n, N=N, nghosts=nghosts,
                                             subdomain2global=subdomain2global)
        self.par_h_max = LAT.ParVec_petsc4py(self.h_max,
                                             bs=1,
                                             n=n, N=N, nghosts=nghosts,
                                             subdomain2global=subdomain2global)
        self.par_heta_min = LAT.ParVec_petsc4py(self.heta_min,
                                                bs=1,
                                                n=n, N=N, nghosts=nghosts,
                                                subdomain2global=subdomain2global)
        self.par_heta_max = LAT.ParVec_petsc4py(self.heta_max,
                                                bs=1,
                                                n=n, N=N, nghosts=nghosts,
                                                subdomain2global=subdomain2global)
        self.par_kin_max = LAT.ParVec_petsc4py(self.kin_max,
                                               bs=1,
                                               n=n, N=N, nghosts=nghosts,
                                               subdomain2global=subdomain2global)
        self.par_entropy = LAT.ParVec_petsc4py(self.entropy,
                                               bs=1,
                                               n=n, N=N, nghosts=nghosts,
                                               subdomain2global=subdomain2global)
        self.par_global_entropy_residual = LAT.ParVec_petsc4py(self.global_entropy_residual,
                                                               bs=1,
                                                               n=n, N=N, nghosts=nghosts,
                                                               subdomain2global=subdomain2global)
        self.par_normalx = LAT.ParVec_petsc4py(self.normalx,
                                               bs=1,
                                               n=n, N=N, nghosts=nghosts,
                                               subdomain2global=subdomain2global)
        self.par_normaly = LAT.ParVec_petsc4py(self.normaly,
                                               bs=1,
                                               n=n, N=N, nghosts=nghosts,
                                               subdomain2global=subdomain2global)
        self.par_bathymetry = LAT.ParVec_petsc4py(self.coefficients.b.dof,
                                                  bs=1,
                                                  n=n, N=N, nghosts=nghosts,
                                                  subdomain2global=subdomain2global)
        self.par_bathymetry.scatter_forward_insert()

        self.par_ML = LAT.ParVec_petsc4py(self.ML,
                                          bs=1,
                                          n=n, N=N, nghosts=nghosts,
                                          subdomain2global=subdomain2global)
        self.par_ML.scatter_forward_insert()
        # Wall flags come from the subdomain's exterior boundary, which also
        # contains the partition boundary: owned flags are right, ghost flags
        # are not. Take the ghosts' flags from their owners.
        self.par_wall_dof = LAT.ParVec_petsc4py(self.wall_dof,
                                                bs=1,
                                                n=n, N=N, nghosts=nghosts,
                                                subdomain2global=subdomain2global)
        self.par_wall_dof.scatter_forward_insert()
        self.par_inverse_mesh = LAT.ParVec_petsc4py(self.inverse_mesh,
                                                    bs=1,
                                                    n=n, N=N, nghosts=nghosts,
                                                    subdomain2global=subdomain2global)
        self.par_inverse_mesh.scatter_forward_insert()
        self.par_hLow = LAT.ParVec_petsc4py(self.hLow,
                                            bs=1,
                                            n=n, N=N, nghosts=nghosts,
                                            subdomain2global=subdomain2global)
        self.par_huLow = LAT.ParVec_petsc4py(self.huLow,
                                             bs=1,
                                             n=n, N=N, nghosts=nghosts,
                                             subdomain2global=subdomain2global)
        self.par_hvLow = LAT.ParVec_petsc4py(self.hvLow,
                                             bs=1,
                                             n=n, N=N, nghosts=nghosts,
                                             subdomain2global=subdomain2global)
        self.par_hetaLow = LAT.ParVec_petsc4py(self.hetaLow,
                                               bs=1,
                                               n=n, N=N, nghosts=nghosts,
                                               subdomain2global=subdomain2global)
        self.par_kin_max = LAT.ParVec_petsc4py(self.kin_max,
                                               bs=1,
                                               n=n, N=N, nghosts=nghosts,
                                               subdomain2global=subdomain2global)
        self.par_thetaj_inv = LAT.ParVec_petsc4py(self.thetaj_inv,
                                                  bs=1,
                                                  n=n, N=N, nghosts=nghosts,
                                                  subdomain2global=subdomain2global)
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
            assert self.u[0].dof.min(
            ) >= -self.eps * self.u[0].dof.max(), ("Negative water height: ", self.u[0].dof.min())

        # To compute second difference relaxation quantities and entropies
        self.computePreStep()
        ### Distribute ####
        self.par_entropy.scatter_forward_insert()
        self.par_delta_Sqd_h.scatter_forward_insert()
        self.par_delta_Sqd_heta.scatter_forward_insert()
        self.par_thetaj_inv.scatter_forward_insert()
        ##################

        # lets call calculate EV first and distribute
        self.computeEV()
        self.par_global_entropy_residual.scatter_forward_insert()

        # For waves
        if self.coefficients.waveConditions is not None:
            self.compute_waves()
            # get bounds/higher RHS
        self.computeBoundsAndRhsHigh()

        ##### Then distribute ######################
        self.par_RHS_high_h.scatter_forward_insert()
        self.par_RHS_high_hu.scatter_forward_insert()
        self.par_RHS_high_hv.scatter_forward_insert()
        self.par_RHS_high_heta.scatter_forward_insert()
        self.par_RHS_high_hw.scatter_forward_insert()
        self.par_RHS_high_hbeta.scatter_forward_insert()
        self.par_hLow.scatter_forward_insert()
        self.par_huLow.scatter_forward_insert()
        self.par_hvLow.scatter_forward_insert()
        self.par_hetaLow.scatter_forward_insert()
        self.par_h_min.scatter_forward_insert()
        self.par_h_max.scatter_forward_insert()
        self.par_heta_min.scatter_forward_insert()
        self.par_heta_max.scatter_forward_insert()
        self.par_kin_max.scatter_forward_insert()
        #############################################

        rowptr, colind, MassMatrix = self.MC_global.getCSRrepresentation()
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
        argsDict["heta_dof_old"] = self.heta_dof_old
        argsDict["hw_dof_old"] = self.hw_dof_old
        argsDict["hbeta_dof_old"] = self.hbeta_dof_old
        argsDict["b_dof"] = self.coefficients.b.dof
        argsDict["h_dof"] = self.u[0].dof
        argsDict["hu_dof"] = self.u[1].dof
        argsDict["hv_dof"] = self.u[2].dof
        argsDict["heta_dof"] = self.u[3].dof
        argsDict["hw_dof"] = self.u[4].dof
        argsDict["hbeta_dof"] = self.u[5].dof
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
        argsDict["offset_heta"] = self.offset[3]
        argsDict["offset_hw"] = self.offset[4]
        argsDict["offset_hbeta"] = self.offset[5]
        argsDict["stride_h"] = self.stride[0]
        argsDict["stride_hu"] = self.stride[1]
        argsDict["stride_hv"] = self.stride[2]
        argsDict["stride_heta"] = self.stride[3]
        argsDict["stride_hw"] = self.stride[4]
        argsDict["stride_hbeta"] = self.stride[5]
        argsDict["globalResidual"] = r
        argsDict["porosity_dof"] = self.coefficients.porosity_dof
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["isDOFBoundary_h"] = self.numericalFlux.isDOFBoundary[0]
        argsDict["isDOFBoundary_hu"] = self.numericalFlux.isDOFBoundary[1]
        argsDict["isDOFBoundary_hv"] = self.numericalFlux.isDOFBoundary[2]
        argsDict["isAdvectiveFluxBoundary_h"] = self.ebqe[(
            'advectiveFlux_bc_flag', 0)]
        argsDict["isAdvectiveFluxBoundary_hu"] = self.ebqe[(
            'advectiveFlux_bc_flag', 1)]
        argsDict["isAdvectiveFluxBoundary_hv"] = self.ebqe[(
            'advectiveFlux_bc_flag', 2)]
        argsDict["isDiffusiveFluxBoundary_hu"] = self.ebqe[(
            'diffusiveFlux_bc_flag', 1, 1)]
        argsDict["isDiffusiveFluxBoundary_hv"] = self.ebqe[(
            'diffusiveFlux_bc_flag', 2, 2)]
        argsDict["ebqe_bc_h_ext"] = self.numericalFlux.ebqe[('u', 0)]
        argsDict["ebqe_bc_flux_mass_ext"] = self.ebqe[('advectiveFlux_bc', 0)]
        argsDict["ebqe_bc_flux_mom_hu_adv_ext"] = self.ebqe[(
            'advectiveFlux_bc', 1)]
        argsDict["ebqe_bc_flux_mom_hv_adv_ext"] = self.ebqe[(
            'advectiveFlux_bc', 2)]
        argsDict["ebqe_bc_hu_ext"] = self.numericalFlux.ebqe[('u', 1)]
        argsDict["ebqe_bc_flux_hu_diff_ext"] = self.ebqe[(
            'diffusiveFlux_bc', 1, 1)]
        argsDict["ebqe_penalty_ext"] = self.ebqe[('penalty')]
        argsDict["ebqe_bc_hv_ext"] = self.numericalFlux.ebqe[('u', 2)]
        argsDict["ebqe_bc_flux_hv_diff_ext"] = self.ebqe[(
            'diffusiveFlux_bc', 2, 2)]
        argsDict["q_velocity"] = self.q[('velocity', 0)]
        argsDict["ebqe_velocity"] = self.ebqe[('velocity', 0)]
        argsDict["flux"] = self.ebq_global[('totalFlux', 0)]
        argsDict["elementResidual_h_save"] = self.elementResidual[0]
        argsDict["Cx"] = self.Cx
        argsDict["Cy"] = self.Cy
        argsDict["CTx"] = self.CTx
        argsDict["CTy"] = self.CTy
        argsDict["numDOFsPerEqn"] = self.numDOFsPerEqn
        argsDict["csrRowIndeces_DofLoops"] = self.rowptr_cMatrix
        argsDict["csrColumnOffsets_DofLoops"] = self.colind_cMatrix
        argsDict["lumped_mass_matrix"] = self.ML
        argsDict["cfl_run"] = self.timeIntegration.runCFL
        argsDict["hEps"] = self.hEps
        argsDict["hnp1_at_quad_point"] = self.q[('u', 0)]
        argsDict["hunp1_at_quad_point"] = self.q[('u', 1)]
        argsDict["hvnp1_at_quad_point"] = self.q[('u', 2)]
        argsDict["hetanp1_at_quad_point"] = self.q[('u', 3)]
        argsDict["hwnp1_at_quad_point"] = self.q[('u', 4)]
        argsDict["hbetanp1_at_quad_point"] = self.q[('u', 5)]
        argsDict["LUMPED_MASS_MATRIX"] = self.coefficients.LUMPED_MASS_MATRIX
        argsDict["dt"] = self.timeIntegration.dt
        argsDict["quantDOFs"] = self.quantDOFs
        argsDict["SECOND_CALL_CALCULATE_RESIDUAL"] = self.secondCallCalculateResidual
        argsDict["COMPUTE_NORMALS"] = self.COMPUTE_NORMALS
        argsDict["normalx"] = self.normalx
        argsDict["normaly"] = self.normaly
        argsDict["lstage"] = self.timeIntegration.lstage
        argsDict["MassMatrix"] = MassMatrix
        argsDict["RHS_high_h"] = self.RHS_high_h
        argsDict["RHS_high_hu"] = self.RHS_high_hu
        argsDict["RHS_high_hv"] = self.RHS_high_hv
        argsDict["RHS_high_heta"] = self.RHS_high_heta
        argsDict["RHS_high_hw"] = self.RHS_high_hw
        argsDict["RHS_high_hbeta"] = self.RHS_high_hbeta

        ## call calculate residual
        self.calculateResidual(argsDict)

        if self.COMPUTE_NORMALS == 1:
            self.par_normalx.scatter_forward_insert()
            self.par_normaly.scatter_forward_insert()
            self.COMPUTE_NORMALS = 0
        #

        # for reflecting conditions on all boundaries
        legacy_walls = not self.coefficients.INTEGRATE_BY_PARTS_WALLS
        if legacy_walls and self.reflectingBoundaryConditions and self.boundaryIndex is not None:
            for dummy, index in enumerate(self.boundaryIndex):
                r[self.offset[1]+self.stride[1]*index] = 0.
                r[self.offset[2]+self.stride[2]*index] = 0.
            #
        #

        # for reflecting conditions on partial boundaries
        if legacy_walls and not self.reflectingBoundaryConditions and self.reflectingBoundaryIndex is not None:
            for dummy, index in enumerate(self.reflectingBoundaryIndex):
                r[self.offset[1]+self.stride[1]*index] = 0.
                r[self.offset[2]+self.stride[2]*index] = 0.
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
        logEvent("Global residual hyperbolic SGN: ", level=9, data=r)
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
        argsDict["heta_dof"] = self.u[3].dof
        argsDict["hw_dof"] = self.u[4].dof
        argsDict["hbeta_dof"] = self.u[5].dof
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
        argsDict["csrRowIndeces_h_heta"] = self.csrRowIndeces[(0, 3)]
        argsDict["csrColumnOffsets_h_heta"] = self.csrColumnOffsets[(0, 3)]
        argsDict["csrRowIndeces_h_hw"] = self.csrRowIndeces[(0, 4)]
        argsDict["csrColumnOffsets_h_hw"] = self.csrColumnOffsets[(0, 4)]
        argsDict["csrRowIndeces_h_hbeta"] = self.csrRowIndeces[(0, 5)]
        argsDict["csrColumnOffsets_h_hbeta"] = self.csrColumnOffsets[(0, 5)]
        argsDict["csrRowIndeces_hu_h"] = self.csrRowIndeces[(1, 0)]
        argsDict["csrColumnOffsets_hu_h"] = self.csrColumnOffsets[(1, 0)]
        argsDict["csrRowIndeces_hu_hu"] = self.csrRowIndeces[(1, 1)]
        argsDict["csrColumnOffsets_hu_hu"] = self.csrColumnOffsets[(1, 1)]
        argsDict["csrRowIndeces_hu_hv"] = self.csrRowIndeces[(1, 2)]
        argsDict["csrColumnOffsets_hu_hv"] = self.csrColumnOffsets[(1, 2)]
        argsDict["csrRowIndeces_hu_heta"] = self.csrRowIndeces[(1, 3)]
        argsDict["csrColumnOffsets_hu_heta"] = self.csrColumnOffsets[(1, 3)]
        argsDict["csrRowIndeces_hu_hw"] = self.csrRowIndeces[(1, 4)]
        argsDict["csrColumnOffsets_hu_hw"] = self.csrColumnOffsets[(1, 4)]
        argsDict["csrRowIndeces_hu_hbeta"] = self.csrRowIndeces[(1, 5)]
        argsDict["csrColumnOffsets_hu_hbeta"] = self.csrColumnOffsets[(1, 5)]
        argsDict["csrRowIndeces_hv_h"] = self.csrRowIndeces[(2, 0)]
        argsDict["csrColumnOffsets_hv_h"] = self.csrColumnOffsets[(2, 0)]
        argsDict["csrRowIndeces_hv_hu"] = self.csrRowIndeces[(2, 1)]
        argsDict["csrColumnOffsets_hv_hu"] = self.csrColumnOffsets[(2, 1)]
        argsDict["csrRowIndeces_hv_hv"] = self.csrRowIndeces[(2, 2)]
        argsDict["csrColumnOffsets_hv_hv"] = self.csrColumnOffsets[(2, 2)]
        argsDict["csrRowIndeces_hv_heta"] = self.csrRowIndeces[(2, 3)]
        argsDict["csrColumnOffsets_hv_heta"] = self.csrColumnOffsets[(2, 3)]
        argsDict["csrRowIndeces_hv_hw"] = self.csrRowIndeces[(2, 4)]
        argsDict["csrColumnOffsets_hv_hw"] = self.csrColumnOffsets[(2, 4)]
        argsDict["csrRowIndeces_hv_hbeta"] = self.csrRowIndeces[(2, 5)]
        argsDict["csrColumnOffsets_hv_hbeta"] = self.csrColumnOffsets[(2, 5)]
        argsDict["csrRowIndeces_heta_h"] = self.csrRowIndeces[(3, 0)]
        argsDict["csrColumnOffsets_heta_h"] = self.csrColumnOffsets[(3, 0)]
        argsDict["csrRowIndeces_heta_hu"] = self.csrRowIndeces[(3, 1)]
        argsDict["csrColumnOffsets_heta_hu"] = self.csrColumnOffsets[(3, 1)]
        argsDict["csrRowIndeces_heta_hv"] = self.csrRowIndeces[(3, 2)]
        argsDict["csrColumnOffsets_heta_hv"] = self.csrColumnOffsets[(3, 2)]
        argsDict["csrRowIndeces_heta_heta"] = self.csrRowIndeces[(3, 3)]
        argsDict["csrColumnOffsets_heta_heta"] = self.csrColumnOffsets[(3, 3)]
        argsDict["csrRowIndeces_heta_hw"] = self.csrRowIndeces[(3, 4)]
        argsDict["csrColumnOffsets_heta_hw"] = self.csrColumnOffsets[(3, 4)]
        argsDict["csrRowIndeces_heta_hbeta"] = self.csrRowIndeces[(3, 5)]
        argsDict["csrColumnOffsets_heta_hbeta"] = self.csrColumnOffsets[(3, 5)]
        argsDict["csrRowIndeces_hw_h"] = self.csrRowIndeces[(4, 0)]
        argsDict["csrColumnOffsets_hw_h"] = self.csrColumnOffsets[(4, 0)]
        argsDict["csrRowIndeces_hw_hu"] = self.csrRowIndeces[(4, 1)]
        argsDict["csrColumnOffsets_hw_hu"] = self.csrColumnOffsets[(4, 1)]
        argsDict["csrRowIndeces_hw_hv"] = self.csrRowIndeces[(4, 2)]
        argsDict["csrColumnOffsets_hw_hv"] = self.csrColumnOffsets[(4, 2)]
        argsDict["csrRowIndeces_hw_heta"] = self.csrRowIndeces[(4, 3)]
        argsDict["csrColumnOffsets_hw_heta"] = self.csrColumnOffsets[(4, 3)]
        argsDict["csrRowIndeces_hw_hw"] = self.csrRowIndeces[(4, 4)]
        argsDict["csrColumnOffsets_hw_hw"] = self.csrColumnOffsets[(4, 4)]
        argsDict["csrRowIndeces_hw_hbeta"] = self.csrRowIndeces[(4, 5)]
        argsDict["csrColumnOffsets_hw_hbeta"] = self.csrColumnOffsets[(4, 5)]
        argsDict["csrRowIndeces_hbeta_h"] = self.csrRowIndeces[(5, 0)]
        argsDict["csrColumnOffsets_hbeta_h"] = self.csrColumnOffsets[(5, 0)]
        argsDict["csrRowIndeces_hbeta_hu"] = self.csrRowIndeces[(5, 1)]
        argsDict["csrColumnOffsets_hbeta_hu"] = self.csrColumnOffsets[(5, 1)]
        argsDict["csrRowIndeces_hbeta_hv"] = self.csrRowIndeces[(5, 2)]
        argsDict["csrColumnOffsets_hbeta_hv"] = self.csrColumnOffsets[(5, 2)]
        argsDict["csrRowIndeces_hbeta_heta"] = self.csrRowIndeces[(5, 3)]
        argsDict["csrColumnOffsets_hbeta_heta"] = self.csrColumnOffsets[(5, 3)]
        argsDict["csrRowIndeces_hbeta_hw"] = self.csrRowIndeces[(5, 4)]
        argsDict["csrColumnOffsets_hbeta_hw"] = self.csrColumnOffsets[(5, 4)]
        argsDict["csrRowIndeces_hbeta_hbeta"] = self.csrRowIndeces[(5, 5)]
        argsDict["csrColumnOffsets_hbeta_hbeta"] = self.csrColumnOffsets[(
            5, 5)]
        argsDict["globalJacobian"] = jacobian.getCSRrepresentation()[2]
        argsDict["porosity_dof"] = self.coefficients.porosity_dof
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["isDOFBoundary_h"] = self.numericalFlux.isDOFBoundary[0]
        argsDict["isDOFBoundary_hu"] = self.numericalFlux.isDOFBoundary[1]
        argsDict["isDOFBoundary_hv"] = self.numericalFlux.isDOFBoundary[2]
        argsDict["isAdvectiveFluxBoundary_h"] = self.ebqe[(
            'advectiveFlux_bc_flag', 0)]
        argsDict["isAdvectiveFluxBoundary_hu"] = self.ebqe[(
            'advectiveFlux_bc_flag', 1)]
        argsDict["isAdvectiveFluxBoundary_hv"] = self.ebqe[(
            'advectiveFlux_bc_flag', 2)]
        argsDict["isDiffusiveFluxBoundary_hu"] = self.ebqe[(
            'diffusiveFlux_bc_flag', 1, 1)]
        argsDict["isDiffusiveFluxBoundary_hv"] = self.ebqe[(
            'diffusiveFlux_bc_flag', 2, 2)]
        argsDict["ebqe_bc_h_ext"] = self.numericalFlux.ebqe[('u', 0)]
        argsDict["ebqe_bc_flux_mass_ext"] = self.ebqe[('advectiveFlux_bc', 0)]
        argsDict["ebqe_bc_flux_mom_hu_adv_ext"] = self.ebqe[(
            'advectiveFlux_bc', 1)]
        argsDict["ebqe_bc_flux_mom_hv_adv_ext"] = self.ebqe[(
            'advectiveFlux_bc', 2)]
        argsDict["ebqe_bc_hu_ext"] = self.numericalFlux.ebqe[('u', 1)]
        argsDict["ebqe_bc_flux_hu_diff_ext"] = self.ebqe[(
            'diffusiveFlux_bc', 1, 1)]
        argsDict["ebqe_penalty_ext"] = self.ebqe[('penalty')]
        argsDict["ebqe_bc_hv_ext"] = self.numericalFlux.ebqe[('u', 2)]
        argsDict["ebqe_bc_flux_hv_diff_ext"] = self.ebqe[(
            'diffusiveFlux_bc', 2, 2)]
        argsDict["csrColumnOffsets_eb_h_h"] = self.csrColumnOffsets_eb[(0, 0)]
        argsDict["csrColumnOffsets_eb_h_hu"] = self.csrColumnOffsets_eb[(0, 1)]
        argsDict["csrColumnOffsets_eb_h_hv"] = self.csrColumnOffsets_eb[(0, 2)]
        argsDict["csrColumnOffsets_eb_hu_h"] = self.csrColumnOffsets_eb[(1, 0)]
        argsDict["csrColumnOffsets_eb_hu_hu"] = self.csrColumnOffsets_eb[(
            1, 1)]
        argsDict["csrColumnOffsets_eb_hu_hv"] = self.csrColumnOffsets_eb[(
            1, 2)]
        argsDict["csrColumnOffsets_eb_hv_h"] = self.csrColumnOffsets_eb[(2, 0)]
        argsDict["csrColumnOffsets_eb_hv_hu"] = self.csrColumnOffsets_eb[(
            2, 1)]
        argsDict["csrColumnOffsets_eb_hv_hv"] = self.csrColumnOffsets_eb[(
            2, 2)]
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
        self.u[0].femSpace.elementMaps.getValues(
            self.elementQuadraturePoints, self.q['x'])
        self.u[0].femSpace.elementMaps.getBasisValuesRef(
            self.elementQuadraturePoints)
        self.u[0].femSpace.elementMaps.getBasisGradientValuesRef(
            self.elementQuadraturePoints)
        self.u[0].femSpace.getBasisValuesRef(self.elementQuadraturePoints)
        self.u[0].femSpace.getBasisGradientValuesRef(
            self.elementQuadraturePoints)
        self.u[1].femSpace.getBasisValuesRef(self.elementQuadraturePoints)
        self.u[1].femSpace.getBasisGradientValuesRef(
            self.elementQuadraturePoints)
        self.coefficients.initializeElementQuadrature(
            self.timeIntegration.t, self.q)

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
        self.u[0].femSpace.elementMaps.getValuesGlobalExteriorTrace(self.elementBoundaryQuadraturePoints,
                                                                    self.ebqe['x'])
        self.fluxBoundaryConditionsObjectsDict = dict([(cj, FluxBoundaryConditions(self.mesh,
                                                                                   self.nElementBoundaryQuadraturePoints_elementBoundary,
                                                                                   self.ebqe[(
                                                                                       'x')],
                                                                                   self.advectiveFluxBoundaryConditionsSetterDict[
                                                                                       cj],
                                                                                   self.diffusiveFluxBoundaryConditionsSetterDictDict[cj]))
                                                       for cj in list(self.advectiveFluxBoundaryConditionsSetterDict.keys())])
        self.coefficients.initializeGlobalExteriorElementBoundaryQuadrature(
            self.timeIntegration.t, self.ebqe)

    def estimate_mt(self):
        pass

    def calculateSolutionAtQuadrature(self):
        pass

    def calculateAuxiliaryQuantitiesAfterStep(self):
        OneLevelTransport.calculateAuxiliaryQuantitiesAfterStep(self)

    def getForce(self, cg, forceExtractionFaces, force, moment):
        pass