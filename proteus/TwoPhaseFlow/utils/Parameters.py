import numpy as np
from petsc4py import PETSc
from proteus.Profiling import logEvent
from proteus.MeshTools import MeshOptions
from proteus.defaults import (Physics_base,
                              Numerics_base,
                              System_base)
from proteus import Comm
comm=Comm.get()
if comm.size() > 1:
    parallel=True
else:
    parallel=False

# models
from proteus.mprans import (RANS2P,
                            RANS3PF,
                            VOF,
                            RDLS,
                            NCLS,
                            MCorr,
                            CLSVOF,
                            AddedMass,
                            MoveMesh,
                            MoveMeshMonitor,
                            Pres,
                            PresInit,
                            PresInc,
                            Kappa,
                            Dissipation)
# numerical options
from proteus import (StepControl,
                     TimeIntegration,
                     NonlinearSolvers,
                     LinearSolvers,
                     LinearAlgebraTools,
                     NumericalFlux)

# default values for several models
epsFact = 1.5
sc_uref = 1.
sc_beta = 1.5
shockCapturingFactor = 0.5
minTol = 1e-8
default_kappa_turbulence = 1e-3
default_dissipation_turbulence = 1e-3


class ParametersHolder:
    """
    """
    def __init__(self, ProblemInstance=None):
        # gain access to problem class if necessary
        self._Problem = ProblemInstance
        # default options
        self.model_list = []
        self.physical = self._Problem.SystemPhysics

    def initializeParameters(self):
        logEvent('----------')
        logEvent('Mesh Options')
        for key, value in self.mesh.__dict__.items():
            if key[0] != '_':  # do not print hidden attributes
                logEvent('{key}: {value}'. format(key=key, value=value))
        logEvent('----------')
        logEvent('Physical Parameters')
        for key, value in self.physical.__dict__.items():
            if key[0] != '_':  # do not print hidden attributes
                logEvent('{key}: {value}'. format(key=key, value=value))
        logEvent('----------')
        n_base = Numerics_base()
        p_base = Physics_base()
        for (idx,model) in enumerate(self.model_list):
            model.initializePhysics()
            model.initializeNumerics()
            model.initializePETScOptions()
            logEvent('TwoPhaseFlow parameters for model: {name}'.format(name=model['name']))
            logEvent('-----')
            logEvent('{name} PHYSICS'.format(name=model.name))
            logEvent('-----')
            logEvent('COEFFICIENTS OPTIONS')
            for key, value in sorted(model.p.coefficients.__dict__.items()):
                if key[0] != '_':  # do not print hidden attributes
                    logEvent('{key}: {value}'. format(key=key, value=value))
            logEvent('END OF COEFFICIENTS OPTIONS')
            for key, value in sorted(model.p.__dict__.items()):
                if key[0] != '_':  # do not print hidden attributes
                    if key in p_base.__dict__.keys():
                        if value != p_base.__dict__[key]:
                            logEvent('(!) {key}: {value}'. format(key=key, value=value))
                        else:
                            logEvent('{key}: {value}'. format(key=key, value=value))
                    else:
                        logEvent('{key}: {value}'. format(key=key, value=value))
            logEvent('-----')
            logEvent('{name} NUMERICS'.format(name=model.name))
            logEvent('-----')
            for key, value in sorted(model.n.__dict__.items()):
                if key[0] != '_':  # do not print hidden attributes
                    if key in n_base.__dict__.keys():
                        if value != n_base.__dict__[key]:
                            logEvent('(!) {key}: {value}'. format(key=key, value=value))
                        else:
                            logEvent('{key}: {value}'. format(key=key, value=value))
                    else:
                        logEvent('{key}: {value}'. format(key=key, value=value))
            logEvent('----------')

        logEvent('-----')
        logEvent('PETSc OPTIONS')
        petsc_info = PETSc.Options().getAll()
        for key, val in sorted(petsc_info.items()):
            logEvent(str(key)+': '+str(val))
        logEvent('-----')

class FreezableClass(object):
    """Base class for all parameters class, enforces attribute freezing
    """
    __frozen = False

    def __init__(self, name=None):
        self.name = name

    def __getitem__(self, key):
        if key not in self.__dict__:
            raise AttributeError("{key} is not an option for class {name}".format(key=key, name=self.__class__.__name__))

        return self.__dict__[key]

    def __setitem__(self, key, val):
        self.__setattr__(key, val)

    def __setattr__(self, key, val):
        if self.__frozen and not hasattr(self, key):
            raise AttributeError("{key} is not an option for class {name}".format(key=key, name=self.__class__.__name__))
        object.__setattr__(self, key, val)

    def _freeze(self):
        self.__frozen = True

    def addOption(self, name, value):
        self.__frozen = False
        self.__setattr__(name, value)
        self._freeze()

def petsc_has_hypre():
    """True when the PETSc in use was built with hypre (rans2p.preconditioner = 'schur_boomeramg' needs it)."""
    return bool(PETSc.Sys.hasExternalPackage('hypre'))


class ParametersModelBase(FreezableClass):
    """
    """
    def __init__(self,
                 name=None,Problem=None):
        super(ParametersModelBase, self).__init__(name=name)
        self.index = None
        self.auxiliaryVariables = []
        self._Problem = Problem
        self.OptDB = PETSc.Options()
        self.p = Physics_base()
        self.p.name = name
        # self.p.CoefficientsOptions = FreezableClass()
        self.p._freeze()
        self.n = Numerics_base()
        self.n.name = name
        self.n.ShockCapturingOptions = FreezableClass()
        self.n.SubgridErrorOptions = FreezableClass()
        self.n._freeze()
        # NON LINEAR SOLVERS
        # self.n.fullNewtonFlag = True
        # self.n.multilevelNonlinearSolver = NonlinearSolvers.Newton
        # self.n.levelNonlinearSolver = NonlinearSolvers.Newton
        # self.n.nonlinearSmoother = None
        # self.n.levelNonlinearSolverConvergenceTest = 'r'
        # self.n.nonlinearSolverConvergenceTest = 'r'
        # LINEAR ALGEBRA
        # self.n.matrix = LinearAlgebraTools.SparseMatrix
        # self.n.linearSmoother = None
        # NUMERICAL FLUX
        # self.n.massLumping = False
        self.n.conservativeFlux = None
        # TOLERANCES
        self.n.nl_atol_res = None
        self.n.l_atol_res = None
        # self.n.linTolFac = 0.001
        # self.n.useEisenstatWalker = False
        # self.n.tolFac = 0.
        # self.n.maxNonlinearIts = 50
        # self.n.maxLineSearches = 0

    def initializePhysics(self):
        self.p.domain = self._Problem.domain
        self.p.nd = self._Problem.domain.nd
        self.p.movingDomain = self._Problem.SystemPhysics.movingDomain
        self.p.genMesh = self._Problem.domain.MeshOptions.genMesh
        # initialize extra parameters
        self._initializePhysics()
        self.p._unfreeze()

    def _initializePhysics(self):
        # to overwrite for each models
        pass
    def _setPhysicsValues(self):
        coeffs = self.p.coefficients
        coeffs.movingDomain = self.p.movingDomain
        pparams = self._Problem.SystemPhysics
        coeffs.sigma = pparams.surf_tension_coeff
        coeffs.rho_0 = pparams.rho_0
        coeffs.rho_1 = pparams.rho_1
        coeffs.nu_0 = pparams.nu_0
        coeffs.nu_1 = pparams.nu_1
        coeffs.g = np.array(pparams.gravity)
       

    def initializeNumerics(self):
        self.n.runCFL = self._Problem.SystemNumerics.cfl
        # MESH
        meshOptions = self._Problem.domain.MeshOptions
        self.n.triangleFlag = meshOptions.triangleFlag
        self.n.nnx = meshOptions.nnx
        self.n.nny = meshOptions.nny
        self.n.nnz = meshOptions.nnz
        self.n.triangleOptions = meshOptions.triangleOptions
        self.n.parallelPartitioningType = meshOptions.parallelPartitioningType
        self.n.nLayersOfOverlapForParallel = meshOptions.nLayersOfOverlapForParallel
        self.n.restrictFineSolutionToAllMeshes = meshOptions.restrictFineSolutionToAllMeshes
        # TIME INTEGRATION
        self.n.runCFL = self._Problem.SystemNumerics.cfl
        # FINITE ELEMENT SPACES
        FESpace = self._Problem.FESpace
        self.n.elementQuadrature = FESpace['elementQuadrature']
        self.n.elementBoundaryQuadrature = FESpace['elementBoundaryQuadrature']
        # SUPERLU
        if self._Problem.SystemNumerics.useSuperlu and not parallel:
            self.n.multilevelLinearSolver = LinearSolvers.LU
            self.n.levelLinearSolver = LinearSolvers.LU
        # AUXILIARY VARIABLES
        self.n.auxiliaryVariables = self.auxiliaryVariables
        # initialize extra parameters
        self._initializeNumerics()
        self.n._unfreeze()

    def _initializeNumerics(self):
        # to overwrite for each models
        pass

    def initializePETScOptions(self):
        if not self._Problem.SystemNumerics.usePETScOptionsFileExternal:
            # use default options if no file
            self._initializePETScOptions()
        else:
            # check if file contains options for model
            # use default options for model if no options in file
            prefix = self.n.linear_solver_options_prefix
            petsc_options = PETSc.Options().getAll()
            initialize = True
            i = 0
            for key in petsc_options.keys():
                i += 1
                if prefix == key[:len(prefix)]:
                    initialize = False
                    break
            if initialize:
                self._initializePETScOptions()

    def _initializePETScOptions(self):
        pass
    
    def fetchIndex(self,idxDict,name):
        try:
            return idxDict[name]
        except:
            return None

    def setInitialConditionStructure(self):
        self.p.initialConditions = FreezableClass()
        for name in self.p.coefficients.variableNames:
            setattr(self.p.initialConditions,name,None)
        self.p.initialConditions._freeze()
        self.p.LevelModelType.var2idxDict = {self.p.coefficients.variableNames[i]: i for i,_ in enumerate(self.p.coefficients.variableNames) } 

class ParametersModelRANS2P(ParametersModelBase):
    """
    """
    def __init__(self,ProblemInstance):
        super(ParametersModelRANS2P, self).__init__(name='rans2p',Problem=ProblemInstance)

        self.timeDiscretization = 'be'
        self.p.coefficients = RANS2P.Coefficients(
            nd = self._Problem.domain.nd,
            initialize=False,
            useMetrics=1.,
            epsFact=epsFact,
            eb_penalty_constant=100.,
            particle_epsFact = 3.,
            useExact=ProblemInstance.SystemNumerics.useExact
        )
        scopts = self.n.ShockCapturingOptions
        scopts.shockCapturingFactor = shockCapturingFactor
        scopts.lag = True
        scopts._freeze()
        seopts = self.n.SubgridErrorOptions
        seopts.lag = True
        seopts._freeze()
        # LEVEL MODEL
        self.p.LevelModelType = RANS2P.LevelModel
        self.n.timeOrder = 1
        # NON LINEAR SOLVER
        self.n.multilevelNonlinearSolver = NonlinearSolvers.Newton
        # NUMERICAL FLUX
        self.n.numericalFluxType = RANS2P.NumericalFlux
        # LINEAR ALGEBRA
        self.n.multilevelLinearSolver = LinearSolvers.KSP_petsc4py
        self.n.levelLinearSolver = LinearSolvers.KSP_petsc4py
        self.n.linear_solver_options_prefix = 'rans2p_'
        self.n.linearSolverConvergenceTest = 'r-true'
        self.n.linearSmoother = LinearSolvers.SimpleNavierStokes3D
        self.n.conservativeFlux= {0:'pwl-bdm-opt'}
        # TOLERANCES
        self.n.linTolFac = 0.01
        self.n.tolFac = 0.
        self.n.maxNonlinearIts = 100
        self.n.maxLineSearches = 0
        # The preconditioner of the iterative RANS2P solve (useSuperlu off, the default linear smoother):
        #   'asm'             GMRES + ASM with an LU per subdomain (the default);
        #   'schur_boomeramg' GMRES + an upper Schur complement fieldsplit with BoomerAMG blocks (opt-in;
        #                     needs a PETSc built with hypre). See _initializePETScOptions for the numbers.
        self.preconditioner = 'asm'
        self._freeze()

        #Initial Conditions
        self.setInitialConditionStructure() 

    def _initializePhysics(self):
        pparams = self._Problem.SystemPhysics # physical parameters
        domain = self._Problem.domain
        nd = domain.nd
        # MODEL INDEX
        idxDict = self._Problem.SystemPhysics._modelIdxDict
        ME_model = self.fetchIndex(idxDict,self.name)
        assert ME_model is not None, 'rans2p model index was not set!'
        CLSVOF_model = self.fetchIndex(idxDict, 'clsvof')
        VF_model = self.fetchIndex(idxDict, 'vof')
        LS_model = self.fetchIndex(idxDict, 'ncls')
        K_model = self.fetchIndex(idxDict, 'kappa')
        DISS_model = self.fetchIndex(idxDict, 'dissipation')
        # POROSITY / RELAXATION
        if hasattr(domain, 'porosityTypes'):
            porosityTypes = domain.porosityTypes
            dragAlphaTypes = domain.dragAlphaTypes
            dragBetaTypes = domain.dragBetaTypes
            epsFact_porous = domain.epsFact_porous
        else:
            porosityTypes = None
            dragAlphaTypes = None
            dragBetaTypes = None
            epsFact_solid = None
            epsFact_porous = None
        # COEFFICIENTS
        coeffs = self.p.coefficients
        self._setPhysicsValues()
        coeffs.nd = nd
        coeffs.ME_model = ME_model
        coeffs.CLSVOF_model = CLSVOF_model
        coeffs.VF_model = VF_model
        coeffs.LS_model = LS_model
        coeffs.Closure_0_model = K_model
        coeffs.Closure_1_model = DISS_model
        coeffs.porosityTypes = porosityTypes
        coeffs.dragAlphaTypes = dragAlphaTypes
        coeffs.dragBetaTypes = dragBetaTypes
        if coeffs.barycenters is None:
            coeffs.barycenters = domain.barycenters
        coeffs.initialize()

        # BOUNDARY CONDITIONS
        boundaryConditions = self._Problem.SystemPhysics.boundaryConditions
        if domain.useSpatialTools is False or self._Problem.SystemPhysics.useBoundaryConditionsModule is False:
            self.p.dirichletConditions = {0: boundaryConditions['pressure_DBC'],
                                          1: boundaryConditions['vel_u_DBC'],
                                          2: boundaryConditions['vel_v_DBC']}
            self.p.advectiveFluxBoundaryConditions = {0: boundaryConditions['pressure_AFBC'],
                                                      1: boundaryConditions['vel_u_AFBC'],
                                                      2: boundaryConditions['vel_v_AFBC']}
            self.p.diffusiveFluxBoundaryConditions = {0: {},
                                                      1: {1: boundaryConditions['vel_u_DFBC']},
                                                      2: {2: boundaryConditions['vel_v_DFBC']}}
            if nd == 3:
                self.p.dirichletConditions[3] = boundaryConditions['vel_w_DBC']
                self.p.advectiveFluxBoundaryConditions[3] = boundaryConditions['vel_w_AFBC']
                self.p.diffusiveFluxBoundaryConditions[3] = {3: boundaryConditions['vel_w_DFBC']}
        else:
            self.p.dirichletConditions = {0: lambda x, flag: domain.BCbyFlag[flag].p_dirichlet.uOfXT,
                                          1: lambda x, flag: domain.BCbyFlag[flag].u_dirichlet.uOfXT,
                                          2: lambda x, flag: domain.BCbyFlag[flag].v_dirichlet.uOfXT}
            self.p.advectiveFluxBoundaryConditions = {0: lambda x, flag: domain.BCbyFlag[flag].p_advective.uOfXT,
                                                      1: lambda x, flag: domain.BCbyFlag[flag].u_advective.uOfXT,
                                                      2: lambda x, flag: domain.BCbyFlag[flag].v_advective.uOfXT}
            self.p.diffusiveFluxBoundaryConditions = {0: {},
                                                      1: {1:lambda x, flag: domain.BCbyFlag[flag].u_diffusive.uOfXT},
                                                      2: {2:lambda x, flag: domain.BCbyFlag[flag].v_diffusive.uOfXT}}
            if nd == 3:
                self.p.dirichletConditions[3] = lambda x, flag: domain.BCbyFlag[flag].w_dirichlet.uOfXT
                self.p.advectiveFluxBoundaryConditions[3] = lambda x, flag: domain.BCbyFlag[flag].w_advective.uOfXT
                self.p.diffusiveFluxBoundaryConditions[3] = {3: lambda x, flag: domain.BCbyFlag[flag].w_diffusive.uOfXT}

    def _initializeNumerics(self):
        nd = self._Problem.domain.nd
        # TIME
        if self.timeDiscretization=='vbdf':
            self.n.timeIntegration = TimeIntegration.VBDF
        elif self.timeDiscretization=='be': #backward euler
            self.n.timeIntegration = TimeIntegration.BackwardEuler_cfl
        else:
            raise ValueError("{scheme} scheme is not valid. Accepted schemes values are 'be' and 'vbdf'".format(scheme=self.timeDiscretization))

        self.n.stepController = StepControl.Min_dt_cfl_controller
        # FINITE ELEMENT SPACES
        FESpace = self._Problem.FESpace
        self.n.femSpaces = {0: FESpace['pBasis'],
                            1: FESpace['velBasis'],
                            2: FESpace['velBasis']}
        if nd == 3:
            self.n.femSpaces[3] = FESpace['velBasis']
        # NUMERICAL FLUX
        seopts = self.n.SubgridErrorOptions
        self.n.subgridError = RANS2P.SubgridError(coefficients=self.p.coefficients,
                                                  nd=nd,
                                                  lag=seopts.lag,
                                                  hFactor=FESpace['hFactor'])
        scopts = self.n.ShockCapturingOptions
        self.n.shockCapturing = RANS2P.ShockCapturing(coefficients=self.p.coefficients,
                                                      nd=nd,
                                                      shockCapturingFactor=scopts.shockCapturingFactor,
                                                      lag=scopts.lag)
        # TOLERANCES
        meshOptions = self._Problem.domain.MeshOptions
        if self.n.nl_atol_res is None:
            self.n.nl_atol_res = max(minTol, 0.001*meshOptions.he**2)
        if self.n.l_atol_res is None:
            self.n.l_atol_res = 0.01*self.n.nl_atol_res

    def _initializePETScOptions(self):
        prefix = self.n.linear_solver_options_prefix
        if self._Problem.SystemNumerics.useSuperlu:
            self.OptDB.setValue(prefix+'ksp_type', 'preonly')
            self.OptDB.setValue(prefix+'pc_type', 'lu')
            self.OptDB.setValue(prefix+'pc_factor_mat_solver_type', 'superlu_dist')
        elif self.n.linearSmoother == LinearSolvers.SimpleNavierStokes3D:
            # Measured on damBreakLevelSet (C19 round 2, cekees/proteus#198; ON_NODE_PERFORMANCE_BASELINE §9.1).
            # Both choices: GMRES, right-preconditioned with PETSc's convergence test on the true residual (it
            # replaced a Python callback that rebuilt the solution every iteration), knoll initial guess, and one
            # preconditioner setup per Newton solve (pc_lag_newton).
            if self.preconditioner not in ('asm', 'schur_boomeramg'):
                raise ValueError("rans2p.preconditioner must be 'asm' or 'schur_boomeramg', not %r"
                                 % (self.preconditioner,))
            if self.preconditioner == 'schur_boomeramg' and not petsc_has_hypre():
                raise ValueError("rans2p.preconditioner = 'schur_boomeramg' needs a PETSc built with hypre "
                                 "(BoomerAMG); this PETSc has no hypre. Use 'asm' or rebuild PETSc with hypre.")
            self.OptDB.setValue(prefix+'ksp_type', 'gmres')
            self.OptDB.setValue(prefix+'ksp_max_it', 2000)
            self.OptDB.setValue(prefix+'ksp_gmres_restart', 300)
            self.OptDB.setValue(prefix+'ksp_pc_side', 'right')
            self.OptDB.setValue(prefix+'ksp_converged_use_initial_residual_norm', True)
            self.OptDB.setValue(prefix+'ksp_knoll', 1)
            self.OptDB.setValue(prefix+'pc_lag_newton', True)
            if self.preconditioner == 'asm':
                # The default. ASM with an LU per subdomain: PETSc's own LU with nested-dissection ordering
                # (its symbolic factorization is kept across setups; serial SuperLU's was redone each time),
                # classical Gram-Schmidt (one reduction per iteration, not one per basis vector). Against the
                # options before 2.0, one step is 2.1x faster at 24 ranks and 2.4x at 96 (ceds-model, 1.15 M
                # triangles); KSP iterations per Newton iteration 29 at 24 ranks, 41 at 96.
                self.OptDB.setValue(prefix+'pc_type', 'asm')
                self.OptDB.setValue(prefix+'pc_asm_type', 'basic')
                self.OptDB.setValue(prefix+'sub_ksp_type', 'preonly')
                self.OptDB.setValue(prefix+'sub_pc_type', 'lu')
                self.OptDB.setValue(prefix+'sub_pc_factor_mat_solver_type', 'petsc')
                self.OptDB.setValue(prefix+'sub_pc_factor_mat_ordering_type', 'nd')
            else:
                # Opt-in. An upper block-triangular Schur complement preconditioner on the (velocity,
                # pressure) split SimpleNavierStokes3D defines: one BoomerAMG V-cycle on the velocity block
                # and one on the selfp approximation A11 - A10 diag(A00)^-1 A01. KSP iterations stay at 3-4
                # from 4 to 96 ranks; one step is 7.9x the pre-2.0 options at 24 ranks and 9.3x at 96
                # (3.8x and 3.9x the 'asm' default). Not the default until it is checked on a 3D RANS2P case
                # and every PETSc proteus is built with (conda-forge, the HPC modules, the PETSc fork's
                # --download-proteus-support) is known to have hypre.
                self.OptDB.setValue(prefix+'pc_type', 'fieldsplit')
                self.OptDB.setValue(prefix+'pc_fieldsplit_type', 'schur')
                self.OptDB.setValue(prefix+'pc_fieldsplit_schur_fact_type', 'upper')
                self.OptDB.setValue(prefix+'pc_fieldsplit_schur_precondition', 'selfp')
                for block in ('velocity', 'pressure'):
                    self.OptDB.setValue(prefix+'fieldsplit_'+block+'_ksp_type', 'preonly')
                    self.OptDB.setValue(prefix+'fieldsplit_'+block+'_pc_type', 'hypre')
                    self.OptDB.setValue(prefix+'fieldsplit_'+block+'_pc_hypre_type', 'boomeramg')
            if self.n.linearSolverConvergenceTest == 'r-true':
                self.n.linearSolverConvergenceTest = 'r'
        elif self.n.linearSmoother == LinearSolvers.NavierStokes_TwoPhasePCD:
            # Options for PCD
            # Global KSP options
            self.OptDB.setValue(prefix+'ksp_type', 'fgmres')
            self.OptDB.setValue(prefix+'ksp_gmres_restart', 300)
            self.OptDB.setValue(prefix+'ksp_gmres_modifiedgramschmidt', 1)
            self.OptDB.setValue(prefix+'ksp_pc_side','right')
            self.OptDB.setValue(prefix+'pc_fieldsplit_type', 'schur')
            self.OptDB.setValue(prefix+'pc_fieldsplit_schur_fact_type', 'upper')
            self.OptDB.setValue(prefix+'pc_fieldsplit_schur_precondition', 'user')
            # Velocity block options
            self.OptDB.setValue(prefix+'fieldsplit_velocity_ksp_type', 'gmres')
            self.OptDB.setValue(prefix+'fieldsplit_velocity_ksp_gmres_modifiedgramschmidt', 1)
            self.OptDB.setValue(prefix+'fieldsplit_velocity_ksp_atol', 1e-5)
            self.OptDB.setValue(prefix+'fieldsplit_velocity_ksp_rtol', 1e-5)
            self.OptDB.setValue(prefix+'fieldsplit_velocity_ksp_pc_side', 'right')
            self.OptDB.setValue(prefix+'fieldsplit_velocity_fieldsplit_u_ksp_type', 'preonly')
            self.OptDB.setValue(prefix+'fieldsplit_velocity_fieldsplit_u_pc_type', 'hypre')
            self.OptDB.setValue(prefix+'fieldsplit_velocity_fieldsplit_u_pc_hypre_type', 'boomeramg')
            self.OptDB.setValue(prefix+'fieldsplit_velocity_fieldsplit_u_pc_hypre_boomeramg_coarsen_type', 'HMIS')
            self.OptDB.setValue(prefix+'fieldsplit_velocity_fieldsplit_v_ksp_type', 'preonly')
            self.OptDB.setValue(prefix+'fieldsplit_velocity_fieldsplit_v_pc_type', 'hypre')
            self.OptDB.setValue(prefix+'fieldsplit_velocity_fieldsplit_v_pc_hypre_type', 'boomeramg')
            self.OptDB.setValue(prefix+'fieldsplit_velocity_fieldsplit_v_pc_hypre_boomeramg_coarsen_type', 'HMIS')
            self.OptDB.setValue(prefix+'fieldsplit_velocity_fieldsplit_w_ksp_type', 'preonly')
            self.OptDB.setValue(prefix+'fieldsplit_velocity_fieldsplit_w_pc_type', 'hypre')
            self.OptDB.setValue(prefix+'fieldsplit_velocity_fieldsplit_w_pc_hypre_type', 'boomeramg')
            self.OptDB.setValue(prefix+'fieldsplit_velocity_fieldsplit_w_pc_hypre_boomeramg_coarsen_type', 'HMIS')
            #PCD Schur Complement options
            self.OptDB.setValue(prefix+'fieldsplit_pressure_ksp_type', 'preonly')
            self.OptDB.setValue('innerTPPCDsolver_Qp_visc_ksp_type', 'preonly')
            self.OptDB.setValue('innerTPPCDsolver_Qp_visc_pc_type', 'lu')
            self.OptDB.setValue('innerTPPCDsolver_Qp_visc_pc_factor_mat_solver_type', 'superlu_dist')
            self.OptDB.setValue('innerTPPCDsolver_Qp_dens_ksp_type', 'preonly')
            self.OptDB.setValue('innerTPPCDsolver_Qp_dens_pc_type', 'lu')
            self.OptDB.setValue('innerTPPCDsolver_Qp_dens_pc_factor_mat_solver_type', 'superlu_dist')
            self.OptDB.setValue('innerTPPCDsolver_Ap_rho_ksp_type', 'richardson')
            self.OptDB.setValue('innerTPPCDsolver_Ap_rho_ksp_max_it', 1)
            #self.OptDB.setValue('innerTPPCDsolver_Ap_rho_ksp_constant_null_space',1)
            self.OptDB.setValue('innerTPPCDsolver_Ap_rho_pc_type', 'hypre')
            self.OptDB.setValue('innerTPPCDsolver_Ap_rho_pc_hypre_type', 'boomeramg')
            self.OptDB.setValue('innerTPPCDsolver_Ap_rho_pc_hypre_boomeramg_strong_threshold', 0.5)
            self.OptDB.setValue('innerTPPCDsolver_Ap_rho_pc_hypre_boomeramg_interp_type', 'ext+i-cc')
            self.OptDB.setValue('innerTPPCDsolver_Ap_rho_pc_hypre_boomeramg_coarsen_type', 'HMIS')
            self.OptDB.setValue('innerTPPCDsolver_Ap_rho_pc_hypre_boomeramg_agg_nl', 2)

class ParametersModelRANS3PF(ParametersModelBase):
    """
    """
    def __init__(self,ProblemInstance):
        super(ParametersModelRANS3PF, self).__init__(name='rans3p',Problem=ProblemInstance)
        self.p.coefficients = RANS3PF.Coefficients(
            nd = self._Problem.domain.nd,
            initialize=False,
            useMetrics=1.,
            epsFact_density=epsFact,
            particle_epsFact=3.,
            eb_penalty_constant = 100.0,
            ARTIFICIAL_VISCOSITY=3,
            INT_BY_PARTS_PRESSURE=1,
            USE_SUPG=0,
        )

        scopts = self.n.ShockCapturingOptions
        scopts.shockCapturingFactor = shockCapturingFactor
        scopts.lag = True
        scopts._freeze()
        seopts = self.n.SubgridErrorOptions
        seopts.lag = True
        seopts._freeze()
        # LEVEL MODEL
        self.p.LevelModelType = RANS3PF.LevelModel
        # TIME DISCRETIZATION
        self.timeDiscretization = 'vbdf'
        # NUMERICAL FLUX
        self.n.numericalFluxType = RANS3PF.NumericalFlux
        # LINEAR ALGEBRA
        self.n.multilevelLinearSolver = LinearSolvers.KSP_petsc4py
        self.n.levelLinearSolver = LinearSolvers.KSP_petsc4py
        self.n.linear_solver_options_prefix = 'rans3p_'
        self.n.linearSolverConvergenceTest = 'r-true'
        # NON LINEAR SOLVER
        self.n.multilevelNonlinearSolver = NonlinearSolvers.Newton
        self.n.nonlinearSolverConvergenceTest = 'rits'
        self.n.levelNonlinearSolverConvergenceTest = 'rits'
        # TOLERANCES
        self.n.linTolFac = 0.
        self.n.tolFac = 0.
        self.n.maxNonlinearIts = 1 # This is a linear problem
        self.n.maxLineSearches = 0
        # OTHER
        self.n.addOption('forceTerms', None)
        self.p.addOption('forceTerms', None)
        # freeze attributes
        self._freeze()

        #Initial Conditions
        self.setInitialConditionStructure()

    def _initializePhysics(self):
        pparams = self._Problem.SystemPhysics # physical parameters
        domain = self._Problem.domain
        nd = domain.nd
        # MODEL INDEX

        idxDict = self._Problem.SystemPhysics._modelIdxDict
        nModelId = self.fetchIndex(idxDict, 'ncls')

        VOF_model=self.fetchIndex(idxDict,'vof')
        LS_model=self.fetchIndex(idxDict,'ncls')
        RD_model=self.fetchIndex(idxDict,'rdls')
        MCORR_model=self.fetchIndex(idxDict,'mcorr')

        SED_model=None
        VOS_model=None
        CLSVOF_model = self.fetchIndex(idxDict,'clsvof')
        V_model = self.fetchIndex(idxDict,self.name)
        PINC_model = self.fetchIndex(idxDict,'pressureIncrement')
        PRESSURE_model = self.fetchIndex(idxDict,'pressure')
        K_model = self.fetchIndex(idxDict,'kappa')
        DISS_model = self.fetchIndex(idxDict,'dissipation')

        # COEFFICIENTS
        coeffs = self.p.coefficients
        if coeffs.forceTerms is not None:
            self.p.forceTerms = coeffs.forceTerms
            coeffs.MULTIPLY_EXTERNAL_FORCE_BY_DENSITY = 1
        coeffs.nd = nd
        self._setPhysicsValues()
        coeffs.nd = nd
        coeffs.ME_model = V_model
        coeffs.VOF_model = VOF_model
        coeffs.CLSVOF_model = CLSVOF_model
        coeffs.LS_model = LS_model
        coeffs.SED_model = SED_model
        coeffs.VOS_model = VOS_model
        coeffs.PRESSURE_model = PRESSURE_model
        coeffs.Closure_0_model = K_model
        coeffs.Closure_1_model = DISS_model
        coeffs.initialize()

        # BOUNDARY CONDITIONS
        boundaryConditions = self._Problem.SystemPhysics.boundaryConditions
        if domain.useSpatialTools is False or self._Problem.SystemPhysics.useBoundaryConditionsModule is False:
            self.p.dirichletConditions = {0: boundaryConditions['vel_u_DBC'],
                                          1: boundaryConditions['vel_v_DBC']}
            self.p.advectiveFluxBoundaryConditions = {0: boundaryConditions['vel_u_AFBC'],
                                                      1: boundaryConditions['vel_v_AFBC']}
            self.p.diffusiveFluxBoundaryConditions = {0: {0: boundaryConditions['vel_u_DFBC']},
                                                      1: {1: boundaryConditions['vel_v_DFBC']}}
            if nd == 3:
                self.p.dirichletConditions[2] = boundaryConditions['vel_w_DBC']
                self.p.advectiveFluxBoundaryConditions[2] = boundaryConditions['vel_w_AFBC']
                self.p.diffusiveFluxBoundaryConditions[2] = {2: boundaryConditions['vel_w_DFBC']}
        else:
            self.p.dirichletConditions = {0: lambda x, flag: domain.BCbyFlag[flag].u_dirichlet.uOfXT,
                                          1: lambda x, flag: domain.BCbyFlag[flag].v_dirichlet.uOfXT}
            self.p.advectiveFluxBoundaryConditions = {0: lambda x, flag: domain.BCbyFlag[flag].u_advective.uOfXT,
                                                      1: lambda x, flag: domain.BCbyFlag[flag].v_advective.uOfXT}
            self.p.diffusiveFluxBoundaryConditions = {0: {0:lambda x, flag: domain.BCbyFlag[flag].u_diffusive.uOfXT},
                                                      1: {1:lambda x, flag: domain.BCbyFlag[flag].v_diffusive.uOfXT}}
            if nd == 3:
                self.p.dirichletConditions[2] = lambda x, flag: domain.BCbyFlag[flag].w_dirichlet.uOfXT
                self.p.advectiveFluxBoundaryConditions[2] = lambda x, flag: domain.BCbyFlag[flag].w_advective.uOfXT
                self.p.diffusiveFluxBoundaryConditions[2] = {2: lambda x, flag: domain.BCbyFlag[flag].w_diffusive.uOfXT}

    def _initializeNumerics(self):
        self.n.forceTerms = self.p.forceTerms
        nd = self._Problem.domain.nd
        # TIME
        if self.timeDiscretization=='vbdf':
            self.n.timeIntegration = TimeIntegration.VBDF
            self.n.timeOrder = 2
        else: #backward euler
            self.n.timeIntegration = TimeIntegration.BackwardEuler_cfl
        self.n.stepController = StepControl.Min_dt_cfl_controller
        # FINITE ELEMENT SPACES
        FESpace = self._Problem.FESpace
        self.n.femSpaces = {0: FESpace['velBasis'],
                            1: FESpace['velBasis']}
        if nd == 3:
            self.n.femSpaces[2] = FESpace['velBasis']
        # NUMERICAL FLUX
        seopts = self.n.SubgridErrorOptions
        self.n.subgridError = RANS3PF.SubgridError(coefficients=self.p.coefficients,
                                                  nd=nd,
                                                  lag=seopts.lag,
                                                  hFactor=FESpace['hFactor'])
        scopts = self.n.ShockCapturingOptions
        self.n.shockCapturing = RANS3PF.ShockCapturing(coefficients=self.p.coefficients,
                                                      nd=nd,
                                                      shockCapturingFactor=scopts.shockCapturingFactor,
                                                      lag=scopts.lag)
        # TOLERANCES
        meshOptions = self._Problem.domain.MeshOptions
        if self.n.nl_atol_res is None:
            self.n.nl_atol_res = max(minTol, 0.01*meshOptions.he**2)
        if self.n.l_atol_res is None:
            self.n.l_atol_res = 0.1*self.n.nl_atol_res


class ParametersModelPressure(ParametersModelBase):
    """
    """
    def __init__(self,ProblemInstance):
        super(ParametersModelPressure, self).__init__(name='pressure',Problem=ProblemInstance)
        self.p.coefficients = Pres.Coefficients(
            initialize=False,
        )
        # LEVEL MODEL
        self.p.LevelModelType = Pres.LevelModel
        # NON LINEAR SOLVER
        self.n.multilevelNonlinearSolver = NonlinearSolvers.Newton
        # LINEAR ALGEBRA
        self.n.multilevelLinearSolver = LinearSolvers.KSP_petsc4py
        self.n.levelLinearSolver = LinearSolvers.KSP_petsc4py
        self.n.linear_solver_options_prefix = 'pressure_'
        self.n.linearSolverConvergenceTest = 'r-true'
        # NUMERICAL FLUX
        self.n.numericalFluxType = NumericalFlux.ConstantAdvection_exterior
        # TOLERANCES
        self.n.tolFac = 0.
        self.n.linTolFac = 0.
        self.n.maxLineSearches = 0
        # freeze attributes
        self._freeze()
        #Initial Conditions
        self.setInitialConditionStructure()

    def _initializePhysics(self):
        domain = self._Problem.domain
        nd = domain.nd
        # MODEL INDEXING
        idxDict = self._Problem.SystemPhysics._modelIdxDict
        PRESSURE_model = self.fetchIndex(idxDict,'pressure')
        V_model = self.fetchIndex(idxDict,'rans3p')
        PINC_model = self.fetchIndex(idxDict,'pressureIncrement')
        # COEFFICIENTS
        coeffs = self.p.coefficients
        coeffs.modelIndex = PRESSURE_model
        coeffs.fluidModelIndex = V_model
        coeffs.pressureIncrementModelIndex = PINC_model
        coeffs.initialize()

        # BOUNDARY CONDITIONS
        BC = self._Problem.SystemPhysics.boundaryConditions
        if domain.useSpatialTools is False or self._Problem.SystemPhysics.useBoundaryConditionsModule is False:
            self.p.dirichletConditions = {0: BC['pressure_DBC']}
            self.p.advectiveFluxBoundaryConditions = {0: BC['pressure_AFBC']}
        else:
            self.p.dirichletConditions = {0: lambda x, flag: domain.BCbyFlag[flag].p_dirichlet.uOfXT}
            self.p.advectiveFluxBoundaryConditions = {0: lambda x, flag: domain.BCbyFlag[flag].p_advective.uOfXT}

    def _initializeNumerics(self):
        domain = self._Problem.domain
        nd = domain.nd
        # TIME
        self.n.stepController = StepControl.FixedStep
        # FINITE ELEMENT SPACES
        FESpace = self._Problem.FESpace
        self.n.femSpaces = {0: FESpace['pBasis']}
        # TOLERANCE
        meshOptions = self._Problem.domain.MeshOptions
        if self.n.nl_atol_res is None:
            self.n.nl_atol_res = max(minTol, 0.01*meshOptions.he**2)
        if self.n.l_atol_res is None:
            self.n.l_atol_res = 0.1*self.n.nl_atol_res

class ParametersModelPressureInitial(ParametersModelBase):
    """
    """
    def __init__(self,ProblemInstance):
        super(ParametersModelPressureInitial, self).__init__(name='pressureInitial',Problem=ProblemInstance)
        self.p.coefficients = PresInit.Coefficients(
            initialize=False,
        )

        # NUMERICAL FLUX
        self.n.numericalFluxType = NumericalFlux.ConstantAdvection_exterior
        # NON LINEAR SOLVER
        self.n.multilevelNonlinearSolver = NonlinearSolvers.Newton
        # LINEAR ALGEBRA
        self.n.multilevelLinearSolver = LinearSolvers.KSP_petsc4py
        self.n.levelLinearSolver = LinearSolvers.KSP_petsc4py
        self.n.linearSmoother = LinearSolvers.NavierStokesPressureCorrection # pure neumann laplacian solver
        self.n.linear_solver_options_prefix = 'pinit_'
        self.n.linearSolverConvergenceTest = 'r-true'
        # TOLERANCES
        self.n.tolFac = 0.
        self.n.linTolFac = 0.
        self.n.maxLineSearches = 0
        # freeze attributes
        self._freeze()

        #Initial Conditions
        self.setInitialConditionStructure()

    def _initializePhysics(self):
        domain = self._Problem.domain
        nd = domain.nd
        # MODEL INDEXING
        idxDict = self._Problem.SystemPhysics._modelIdxDict
        PRESSURE_model = self.fetchIndex(idxDict,'pressure')
        V_model = self.fetchIndex(idxDict,'rans3p')
        PINIT_model = self.fetchIndex(idxDict,'pressureInitial')
        # COEFFICIENTS
        coeffs = self.p.coefficients
        coeffs.nd = nd
        coeffs.modelIndex = PINIT_model
        coeffs.fluidModelIndex = V_model
        coeffs.pressureModelIndex = PRESSURE_model
        coeffs.initialize()

        # BOUNDARY CONDITIONS
        BC = self._Problem.SystemPhysics.boundaryConditions
        if domain.useSpatialTools is False or self._Problem.SystemPhysics.useBoundaryConditionsModule is False:
            self.p.dirichletConditions = {0: BC['pressure_DBC']}
            self.p.advectiveFluxBoundaryConditions = {0: BC['pressure_AFBC']}
            self.p.diffusiveFluxBoundaryConditions = {0:{0: BC['pressure_increment_DFBC']}}
        else:
            self.p.dirichletConditions = {0: lambda x, flag: domain.BCbyFlag[flag].p_dirichlet.uOfXT}
            self.p.advectiveFluxBoundaryConditions = {0: lambda x, flag: domain.BCbyFlag[flag].p_advective.uOfXT}
            self.p.diffusiveFluxBoundaryConditions = {0: {0: lambda x, flag: domain.BCbyFlag[flag].pInc_diffusive.uOfXT}}
        # freeze attributes
        self._freeze()

    def _initializeNumerics(self):
        domain = self._Problem.domain
        nd = domain.nd
        # TIME
        self.n.stepController = StepControl.FixedStep
        # FINITE ELEMENT SPACES
        FESpace = self._Problem.FESpace
        self.n.femSpaces = {0: FESpace['pBasis']}
        # LINEAR ALGEBRA
        if self._Problem.SystemNumerics.useSuperlu:
            self.n.linearSmoother = None
        # TOLERANCE
        meshOptions = self._Problem.domain.MeshOptions
        if self.n.nl_atol_res is None:
            self.n.nl_atol_res = max(minTol, 0.01*meshOptions.he**2)
        if self.n.l_atol_res is None:
            self.n.l_atol_res = 0.1*self.n.nl_atol_res

class ParametersModelPressureIncrement(ParametersModelBase):
    """
    """
    def __init__(self,ProblemInstance):
        super(ParametersModelPressureIncrement, self).__init__(name='pressureIncrement',Problem=ProblemInstance)
        self.p.coefficients = PresInc.Coefficients(
            initialize=False,
        )

        # LEVEL MODEL
        self.p.LevelModelType = PresInc.LevelModel
        # NUMERICAL FLUX
        self.n.numericalFluxType = PresInc.NumericalFlux
        # NON LINEAR SOLVER
        self.n.multilevelNonlinearSolver = NonlinearSolvers.Newton
        self.n.multilevelNonlinearSolver = NonlinearSolvers.Newton
        # LINEAR ALGEBRA
        self.n.multilevelLinearSolver = LinearSolvers.KSP_petsc4py
        self.n.levelLinearSolver = LinearSolvers.KSP_petsc4py
        self.n.linear_solver_options_prefix = 'phi_'
        self.n.linearSolverConvergenceTest = 'r-true'
        # TOLERANCES
        self.n.tolFac = 0.
        self.n.linTolFac = 0.
        self.n.maxNonlinearIts = 50
        self.n.maxLineSearches = 0
        # freeze attributes
        self._freeze()

        #Initial Conditions
        self.setInitialConditionStructure()

    def _initializePhysics(self):
        domain = self._Problem.domain
        nd = domain.nd
        pparams = self._Problem.SystemPhysics
        # MODEL INDEXING
        idxDict = self._Problem.SystemPhysics._modelIdxDict
        V_model = self.fetchIndex(idxDict,'rans3p')
        PINC_model = self.fetchIndex(idxDict,'pressureIncrement')
        # COEFFICIENTS
        coeffs = self.p.coefficients
        self._setPhysicsValues()
        coeffs.rho_f_min = (1.0-1.0e-8)*coeffs.rho_1
        coeffs.rho_s_min = (1.0-1.0e-8)*coeffs.rho_0
        coeffs.nd = nd
        coeffs.modelIndex = PINC_model
        coeffs.fluidModelIndex = V_model
        coeffs.initialize()

        # BOUNDARY CONDITIONS
        BC = self._Problem.SystemPhysics.boundaryConditions
        if domain.useSpatialTools is False or self._Problem.SystemPhysics.useBoundaryConditionsModule is False:
            self.p.dirichletConditions = {0: BC['pressure_increment_DBC']}
            self.p.advectiveFluxBoundaryConditions = {0: BC['pressure_increment_AFBC']}
            self.p.diffusiveFluxBoundaryConditions = {0:{0: BC['pressure_increment_DFBC']}}
        else:
            self.p.dirichletConditions = {0: lambda x, flag: domain.BCbyFlag[flag].pInc_dirichlet.uOfXT}
            self.p.advectiveFluxBoundaryConditions = {0: lambda x, flag: domain.BCbyFlag[flag].pInc_advective.uOfXT}
            self.p.diffusiveFluxBoundaryConditions = {0: {0: lambda x, flag: domain.BCbyFlag[flag].pInc_diffusive.uOfXT}}
        # freeze attributes
        self._freeze()

    def _initializeNumerics(self):
        domain = self._Problem.domain
        nd = domain.nd
        # TIME
        self.n.stepController = StepControl.FixedStep
        # FINITE ELEMENT SPACES
        FESpace = self._Problem.FESpace
        self.n.femSpaces = {0: FESpace['pBasis']}
        # LINEAR ALGEBRA
        if self._Problem.SystemNumerics.useSuperlu:
            self.n.linearSmoother = None
        # TOLERANCE
        meshOptions = self._Problem.domain.MeshOptions
        if self.n.nl_atol_res is None:
            self.n.nl_atol_res = max(minTol, 0.01*meshOptions.he**2)
        if self.n.l_atol_res is None:
            self.n.l_atol_res = 0.01*self.n.nl_atol_res

class ParametersModelKappa(ParametersModelBase):
    """
    """
    def __init__(self,ProblemInstance):
        super(ParametersModelKappa, self).__init__(name='kappa',Problem=ProblemInstance)

        self.timeOrder = 1
        self.timeDiscretization = 'be'
        self.p.coefficients = Kappa.Coefficients(
            initialize=False,
            useMetrics=1.,
            epsFact=epsFact,
            sc_uref=sc_uref,
            sc_beta=sc_beta,
        )
        scopts = self.n.ShockCapturingOptions
        scopts.shockCapturingFactor = shockCapturingFactor
        scopts.lag = True
        scopts._freeze()
        seopts = self.n.SubgridErrorOptions
        seopts.lag = True
        seopts._freeze()

        # LEVEL MODEL
        self.p.LevelModelType = Kappa.LevelModel
        # NUMERICAL FLUX
        self.n.numericalFluxType = Kappa.NumericalFlux
        self.n.conservativeFlux = None
       # LINEAR ALGEBRA
        self.n.multilevelLinearSolver = LinearSolvers.KSP_petsc4py
        self.n.levelLinearSolver = LinearSolvers.KSP_petsc4py
        self.n.linear_solver_options_prefix = 'kappa_'
        self.n.linearSolverConvergenceTest = 'r-true'
        # NON LINEAR SOLVER
        self.n.multilevelNonlinearSolver = NonlinearSolvers.Newton
        self.n.nonlinearSolverConvergenceTest = 'rits'
        self.n.levelNonlinearSolverConvergenceTest = 'rits'
        # TOLERANCES
        self.n.linTolFac = 0.
        self.n.tolFac = 0.
        self.n.maxNonlinearIts = 50
        self.n.maxLineSearches = 0
        # freeze attributes
        self._freeze()

        #Initial Conditions
        self.setInitialConditionStructure()

    def _initializePhysics(self):
        idxDict = self._Problem.SystemPhysics._modelIdxDict
        pparams = self._Problem.SystemPhysics # physical parameters
        domain = self._Problem.domain
        nd = domain.nd
        # MODEL INDEX
        VOF_model=self.fetchIndex(idxDict,'vof')
        LS_model=self.fetchIndex(idxDict,'ncls')
        RD_model=self.fetchIndex(idxDict,'rdls')
        MCORR_model=self.fetchIndex(idxDict,'mcorr')
        SED_model=None
        VOS_model=None
        CLSVOF_model = self.fetchIndex(idxDict,'clsvof')
        if(self.fetchIndex(idxDict,'rans3p') is not None):
            V_model = self.fetchIndex(idxDict,'rans3p')
        elif(self.fetchIndex(idxDict,'rans2p') is not None):
            V_model = self.fetchIndex(idxDict,'rans2p')
        else:
            raise ValueError("Kappa model: RANS2P or RANS3P model has not been defined. Please define either one (but not both)")
        PINC_model = self.fetchIndex(idxDict,'pressureIncrement')
        PRESSURE_model = self.fetchIndex(idxDict,'pressure')
        K_model = self.fetchIndex(idxDict,'kappa')
        DISS_model = self.fetchIndex(idxDict,'dissipation')
        # COEFFICIENTS
        coeffs = self.p.coefficients
        coeffs.VOS_model = VOS_model
        coeffs.flowModelIndex = V_model
        coeffs.LS_modelIndex = LS_model
        coeffs.RD_modelIndex = RD_model
        coeffs.dissipation_modelIndex = DISS_model
        coeffs.modelIndex = K_model
        coeffs.SED_modelIndex = SED_model
        coeffs.dissipation_model_flag = pparams.useRANS
        #coeffs.c_mu = pparams.c_mu
        #coeffs.sigma_k = pparams.sigma_k
        self._setPhysicsValues()
        coeffs.nd = nd
        coeffs.initialize()

        # BOUNDARY CONDITIONS
        boundaryConditions = self._Problem.SystemPhysics.boundaryConditions
        if domain.useSpatialTools is False or self._Problem.SystemPhysics.useBoundaryConditionsModule is False:
            self.p.dirichletConditions = {0: boundaryConditions['k_DBC']}

            self.p.advectiveFluxBoundaryConditions = {0: boundaryConditions['k_AFBC']}

            self.p.diffusiveFluxBoundaryConditions = {0: {0: boundaryConditions['k_DFBC']}}
        else:
            self.p.dirichletConditions = {0: lambda x, flag: domain.bc[flag].k_dirichlet.init_cython()}

            self.p.advectiveFluxBoundaryConditions = {0: lambda x, flag: domain.bc[flag].k_advective.init_cython()}

            self.p.diffusiveFluxBoundaryConditions = {0: {0:lambda x, flag: domain.bc[flag].k_diffusive.init_cython()}}


    def _initializeNumerics(self):
        nd = self._Problem.domain.nd
        # TIME
        self.n.timeOrder = 1
        self.n.timeIntegration = TimeIntegration.VBDF#BackwardEuler
        self.n.stepController = StepControl.Min_dt_controller
        # FINITE ELEMENT SPACES
        FESpace = self._Problem.FESpace
        self.n.femSpaces = {0: FESpace['lsBasis']}
        # NUMERICAL FLUX
        seopts=self.n.SubgridErrorOptions
        self.n.subgridError = Kappa.SubgridError(coefficients=self.p.coefficients,
                                                  nd=nd)
        scopts = self.n.ShockCapturingOptions
        self.n.shockCapturing = Kappa.ShockCapturing(coefficients=self.p.coefficients,
                                                      nd=nd,
                                                      shockCapturingFactor=scopts.shockCapturingFactor,
                                                      lag=scopts.lag)
        # TOLERANCES
        meshOptions = self._Problem.domain.MeshOptions
        if self.n.nl_atol_res is None:
            self.n.nl_atol_res = max(minTol, 0.01*meshOptions.he**2)
        if self.n.l_atol_res is None:
            self.n.l_atol_res = 0.1*self.n.nl_atol_res


class ParametersModelDissipation(ParametersModelBase):
    """
    """

    def __init__(self,ProblemInstance):
        super(ParametersModelDissipation, self).__init__(name='dissipation',Problem=ProblemInstance)

        self.p.coefficients = Dissipation.Coefficients(
            initialize=False,
            useMetrics=1.,
            epsFact=epsFact,
            sc_uref=sc_uref,
            sc_beta=sc_beta,
        )
        scopts = self.n.ShockCapturingOptions
        scopts.shockCapturingFactor = shockCapturingFactor
        scopts.lag = True
        scopts._freeze()
        seopts = self.n.SubgridErrorOptions
        seopts.lag = True
        seopts._freeze()

        # LEVEL MODEL
        self.p.LevelModelType = Dissipation.LevelModel
        # TIME DISCRETIZATION
        # self.n.timeOrder = 2
        # self.n.timeDiscretization = 'be'
        # NUMERICAL FLUX
        self.n.numericalFluxType = Dissipation.NumericalFlux
        self.n.conservativeFlux = None
        # LINEAR ALGEBRA
        self.n.multilevelLinearSolver = LinearSolvers.KSP_petsc4py
        self.n.levelLinearSolver = LinearSolvers.KSP_petsc4py
        self.n.linear_solver_options_prefix = 'dissipation_'
        self.n.linearSolverConvergenceTest = 'r-true'
        # NON LINEAR SOLVER
        self.n.multilevelNonlinearSolver = NonlinearSolvers.Newton
        self.n.nonlinearSolverConvergenceTest = 'rits'
        self.n.levelNonlinearSolverConvergenceTest = 'rits'
        # TOLERANCES
        self.n.linTolFac = 0.
        self.n.tolFac = 0.
        self.n.maxNonlinearIts = 50
        self.n.maxLineSearches = 0
        # freeze attributes
        self._freeze()

        #Initial Conditions
        self.setInitialConditionStructure()

    def _initializePhysics(self):
        idxDict = self._Problem.SystemPhysics._modelIdxDict
        pparams = self._Problem.Parameters.physical  # physical parameters
        domain = self._Problem.domain
        nd = domain.nd
        # MODEL INDEX
        VOF_model = self.fetchIndex(idxDict,'vof')
        LS_model = self.fetchIndex(idxDict,'ncls')
        RD_model = self.fetchIndex(idxDict,'rdls')
        MCORR_model = self.fetchIndex(idxDict,'mcorr')
        SED_model = None
        VOS_model = None
        CLSVOF_model = self.fetchIndex(idxDict,'clsvof')
        if(self.fetchIndex(idxDict,'rans3p') is not None):
            V_model = self.fetchIndex(idxDict,'rans3p')
        elif(self.fetchIndex(idxDict,'rans2p') is not None):
            V_model = self.fetchIndex(idxDict,'rans2p')
        else:
            raise ValueError("Dissipation model: RANS2P or RANS3P model has not been defined. Please define either one (but not both)")
        PINC_model = self.fetchIndex(idxDict,'pressureIncrement')
        PRESSURE_model = self.fetchIndex(idxDict,'pressure')
        K_model = self.fetchIndex(idxDict,'kappa')
        DISS_model = self.fetchIndex(idxDict,'dissipation')
        # COEFFICIENTS
        coeffs = self.p.coefficients
        coeffs.VOS_modelIndex = VOS_model
        coeffs.flowModelIndex = V_model
        coeffs.LS_modelIndex = LS_model
        coeffs.RD_modelIndex = RD_model
        coeffs.kappa_modelIndex = K_model
        coeffs.modelIndex = DISS_model
        coeffs.SED_modelIndex = SED_model
        #coeffs.c_mu = pparams.c_mu
        #coeffs.c_1 = pparams.c_1
        #coeffs.c_2 = pparams.c_2
        #coeffs.c_e = pparams.c_e
        #coeffs.sigma_e = pparams.sigma_e
        self._setPhysicsValues()
        coeffs.nd = nd
        # default K-Epsilon, 2 --> K-Omega, 1998, 3 --> K-Omega 1988
        coeffs.dissipation_model_flag = pparams.useRANS
        coeffs.initialize()

        # BOUNDARY CONDITIONS
        boundaryConditions = self._Problem.SystemPhysics.boundaryConditions
        if domain.useSpatialTools is False or self._Problem.SystemPhysics.useBoundaryConditionsModule is False:
            self.p.dirichletConditions = {0: boundaryConditions['dissipation_DBC']}

            self.p.advectiveFluxBoundaryConditions = {0: boundaryConditions['dissipation_AFBC']}

            self.p.diffusiveFluxBoundaryConditions = {0: {0: boundaryConditions['dissipation_DFBC']}}
        else:
            self.p.dirichletConditions = {0: lambda x, flag: domain.bc[flag].dissipation_dirichlet.init_cython()}

            self.p.advectiveFluxBoundaryConditions = {0: lambda x, flag: domain.bc[flag].dissipation_advective.init_cython()}
            self.p.diffusiveFluxBoundaryConditions = {0: {0:lambda x, flag: domain.bc[flag].dissipation_diffusive.init_cython()}}


    def _initializeNumerics(self):
        nd = self._Problem.domain.nd
        # TIME
        self.n.timeOrder = 1
        self.n.timeIntegration = TimeIntegration.BackwardEuler
        self.n.stepController = StepControl.Min_dt_controller
        # FINITE ELEMENT SPACES
        FESpace = self._Problem.FESpace
        self.n.femSpaces = {0: FESpace['lsBasis']}
        # NUMERICAL FLUX
        seopts = self.n.SubgridErrorOptions
        self.n.subgridError = Dissipation.SubgridError(coefficients=self.p.coefficients,
                                                 nd=nd)
        scopts = self.n.ShockCapturingOptions
        self.n.shockCapturing = Dissipation.ShockCapturing(coefficients=self.p.coefficients,
                                                     nd=nd,
                                                     shockCapturingFactor=scopts.shockCapturingFactor,
                                                     lag=scopts.lag)
        # TOLERANCES
        meshOptions = self._Problem.domain.MeshOptions
        if self.n.nl_atol_res is None:
            self.n.nl_atol_res = max(minTol, 0.01 * meshOptions.he ** 2)
        if self.n.l_atol_res is None:
            self.n.l_atol_res = 0.1 * self.n.nl_atol_res


class ParametersModelCLSVOF(ParametersModelBase):
    def __init__(self,ProblemInstance):
        super(ParametersModelCLSVOF, self).__init__(name='clsvof',Problem=ProblemInstance)
        self.p.coefficients = CLSVOF.Coefficients(
            initialize=False,
            useMetrics=1,
            epsFactHeaviside=epsFact,
            epsFactDirac=epsFact,
            epsFactRedist=0.33,
            lambdaFact=10.,
            computeMetrics=1,
        )

        # LEVEL MODEL
        self.p.LevelModelType = CLSVOF.LevelModel
        # NUMERICAL FLUX
        self.n.numericalFluxType = CLSVOF.NumericalFlux
        # NON LINEAR SOLVER
        self.n.levelNonlinearSolver = NonlinearSolvers.CLSVOFNewton
        self.n.multilevelNonlinearSolver = NonlinearSolvers.Newton
        self.n.addOption('updateJacobian', True)
        # LINEAR ALGEBRA
        self.n.multilevelLinearSolver = LinearSolvers.KSP_petsc4py
        self.n.levelLinearSolver = LinearSolvers.KSP_petsc4py
        self.n.linear_solver_options_prefix = 'clsvof_'
        self.n.linearSolverConvergenceTest = 'r-true'
        # TOLERANCES
        self.n.tolFac = 0.
        self.n.maxNonlinearIts = 50
        # freeze attributes
        self._freeze()

        #Initial Conditions
        self.setInitialConditionStructure()

    def _initializePhysics(self):
        domain = self._Problem.domain
        nd = domain.nd
        # MODEL INDEXING

        idxDict = self._Problem.SystemPhysics._modelIdxDict
        CLSVOF_model = self.fetchIndex(idxDict,'clsvof')
        V_model = self.fetchIndex(idxDict,'rans2p')

        if V_model is None:
            V_model = self.fetchIndex(idxDict,'rans3p')
        # COEFFICIENTS
        coeffs = self.p.coefficients
        coeffs.flowModelIndex = V_model
        coeffs.modelIndex = CLSVOF_model
        coeffs.movingDomain = self.p.movingDomain
        coeffs.variableNames = ['phi']
        coeffs.initialize()

        # BOUNDARY CONDITIONS
        BC = self._Problem.SystemPhysics.boundaryConditions
        if domain.useSpatialTools is False or self._Problem.SystemPhysics.useBoundaryConditionsModule is False:
            self.p.dirichletConditions = {0: BC['clsvof_DBC']}
            self.p.advectiveFluxBoundaryConditions = {0: BC['clsvof_AFBC']}
            self.p.diffusiveFluxBoundaryConditions = {0:{0: BC['clsvof_DFBC']}}
        else:
            self.p.dirichletConditions = {0: lambda x, flag: domain.BCbyFlag[flag].vof_dirichlet.uOfXT}
            self.p.advectiveFluxBoundaryConditions = {0: lambda x, flag: domain.BCbyFlag[flag].vof_advective.uOfXT}
            self.p.diffusiveFluxBoundaryConditions = {0: {0: lambda x, flag: domain.BCbyFlag[flag].clsvof_diffusive.uOfXT}}

    def _initializeNumerics(self):
        domain = self._Problem.domain
        nd = domain.nd
        # TIME
        self.n.timeIntegration = TimeIntegration.BackwardEuler_cfl
        self.n.stepController = StepControl.Min_dt_controller
        # FINITE ELEMENT SPACES
        FESpace = self._Problem.FESpace
        self.n.femSpaces = {0: FESpace['lsBasis']}
        # TOLERANCE
        meshOptions = self._Problem.domain.MeshOptions
        if self.n.nl_atol_res is None:
            self.n.nl_atol_res = max(minTol, 0.01*meshOptions.he**2)
        if self.n.l_atol_res is None:
            self.n.l_atol_res = 0.1*self.n.nl_atol_res


class ParametersModelVOF(ParametersModelBase):
    """
    """
    def __init__(self,ProblemInstance):
        super(ParametersModelVOF, self).__init__(name='vof',Problem=ProblemInstance)
        self.p.coefficients = VOF.Coefficients(
            initialize=False,
            useMetrics=1.,
            epsFact=epsFact,
            sc_uref=sc_uref,
            sc_beta=sc_beta,
        )
        
        # LEVEL MODEL
        self.p.LevelModelType = VOF.LevelModel
        # TIME INTEGRATION
        self.n.timeIntegration = TimeIntegration.BackwardEuler_cfl
        self.n.stepController  = StepControl.Min_dt_cfl_controller
        # NUMERICS
        self.n.ShockCapturingOptions.shockCapturingFactor = shockCapturingFactor
        self.n.ShockCapturingOptions.lag = True
        # NUMERICAL FLUX
        self.n.numericalFluxType = VOF.NumericalFlux
        # NON LINEAR SOLVER
        self.n.multilevelNonlinearSolver = NonlinearSolvers.Newton
        # LINEAR ALGEBRA
        self.n.multilevelLinearSolver = LinearSolvers.KSP_petsc4py
        self.n.levelLinearSolver = LinearSolvers.KSP_petsc4py
        self.n.linear_solver_options_prefix = 'vof_'
        self.n.linearSolverConvergenceTest = 'r-true'
        # TOLERANCES
        self.n.maxNonlinearIts = 50
        self.n.maxLineSearches = 0
        self.n.tolFac = 0.
        # freeze attributes
        self._freeze()

        #Initial Conditions
        self.setInitialConditionStructure()

    def _initializePhysics(self):
        domain = self._Problem.domain
        nd = domain.nd
        # MODEL INDEX
        idxDict = self._Problem.SystemPhysics._modelIdxDict
        ME_model = self.fetchIndex(idxDict,self.name)
        assert ME_model is not None, 'vof model index was not set!'
        if('rans2p' in idxDict):
            V_model = self.fetchIndex(idxDict, 'rans2p')
        elif ('rans3p' in idxDict):
            V_model = self.fetchIndex(idxDict, 'rans3p')
        else:
            assert False, 'RANS2P or RANS3PF must be used with VOF'
        RD_model = self.fetchIndex(idxDict,'rdls')

        # COEFFICIENTS
        coeffs = self.p.coefficients
        coeffs.V_model = V_model
        coeffs.RD_modelIndex = RD_model
        coeffs.modelIndex = ME_model
        coeffs.movingDomain = self.p.movingDomain
        coeffs.initialize()

        # BOUNDARY CONDITIONS
        BC = self._Problem.SystemPhysics.boundaryConditions
        if domain.useSpatialTools is False or self._Problem.SystemPhysics.useBoundaryConditionsModule is False:
            self.p.dirichletConditions = {0: BC['vof_DBC']}
            self.p.advectiveFluxBoundaryConditions = {0: BC['vof_AFBC']}
        else:
            self.p.dirichletConditions = {0: lambda x, flag: domain.BCbyFlag[flag].vof_dirichlet.uOfXT}
            self.p.advectiveFluxBoundaryConditions = {0: lambda x, flag: domain.BCbyFlag[flag].vof_advective.uOfXT}
        self.p.diffusiveFluxBoundaryConditions = {0: {}}

    def _initializeNumerics(self):
        domain = self._Problem.domain
        nd = domain.nd
        # TIME
        self.n.timeIntegration = TimeIntegration.BackwardEuler_cfl
        self.n.stepController = StepControl.Min_dt_cfl_controller
        # FINITE ELEMENT SPACES
        FESpace = self._Problem.FESpace
        self.n.femSpaces = {0: FESpace['lsBasis']}
        # NUMERICAL FLUX
        self.n.subgridError = VOF.SubgridError(coefficients=self.p.coefficients,
                                               nd=nd)
        scopts = self.n.ShockCapturingOptions
        self.n.shockCapturing = VOF.ShockCapturing(coefficients=self.p.coefficients,
                                                   nd=nd,
                                                   shockCapturingFactor=scopts.shockCapturingFactor,
                                                   lag=scopts.lag)
        # TOLERANCE
        meshOptions = self._Problem.domain.MeshOptions
        if self.n.nl_atol_res is None:
            self.n.nl_atol_res = max(minTol, 0.001*meshOptions.he**2)
        if self.n.l_atol_res is None:
            self.n.l_atol_res = 0.001*self.n.nl_atol_res

    def _initializePETScOptions(self):
        prefix = self.n.linear_solver_options_prefix
        if self._Problem.SystemNumerics.useSuperlu:
            self.OptDB.setValue(prefix+'ksp_type', 'preonly')
            self.OptDB.setValue(prefix+'pc_type', 'lu')
            self.OptDB.setValue(prefix+'pc_factor_mat_solver_type', 'superlu_dist')
        else:
            self.OptDB.setValue(prefix+'ksp_type', 'gmres')
            self.OptDB.setValue(prefix+'pc_type', 'hypre')
            self.OptDB.setValue(prefix+'pc_pc_hypre_type', 'boomeramg')
            self.OptDB.setValue(prefix+'ksp_gmres_restart', 300)
            self.OptDB.setValue(prefix+'ksp_knoll', 1)
            self.OptDB.setValue(prefix+'ksp_max_it', 2000)


class ParametersModelNCLS(ParametersModelBase):
    """
    """
    def __init__(self,ProblemInstance):
        super(ParametersModelNCLS, self).__init__(name='ncls',Problem=ProblemInstance)
        # PHYSICS
        self.p.coefficients = NCLS.Coefficients(
            initialize=False,
            useMetrics=1.,
            checkMass=False,
            sc_uref=sc_uref,
            sc_beta=sc_beta,
            epsFact=epsFact,
        )
        # LEVEL MODEL
        self.p.LevelModelType = NCLS.LevelModel
        # TIME INTEGRATION
        self.n.timeIntegration = TimeIntegration.BackwardEuler_cfl
        self.n.stepController  = StepControl.Min_dt_cfl_controller
        # NUMERICAL FLUX
        self.n.numericalFluxType = NCLS.NumericalFlux
        self.n.ShockCapturingOptions.shockCapturingFactor = shockCapturingFactor
        self.n.ShockCapturingOptions.lag = True
        # NUMERICAL FLUX
        self.n.numericalFluxType = NCLS.NumericalFlux
        # NON LINEAR SOLVER
        self.n.multilevelNonlinearSolver = NonlinearSolvers.Newton
        # LINEAR ALGEBRA
        self.n.multilevelLinearSolver = LinearSolvers.KSP_petsc4py
        self.n.levelLinearSolver = LinearSolvers.KSP_petsc4py
        self.n.linear_solver_options_prefix = 'ncls_'
        self.n.linearSolverConvergenceTest = 'r-true'
        # TOLERANCES
        self.n.maxNonlinearIts = 50
        self.n.maxLineSearches = 0
        self.n.tolFac = 0.
        # freeze attributes
        self._freeze()
        self.setInitialConditionStructure()

    def _initializePhysics(self):
        domain = self._Problem.domain
        # MODEL INDEXING
        idxDict = self._Problem.SystemPhysics._modelIdxDict
        ME_model = self.fetchIndex(idxDict, self.name)
        assert ME_model is not None, 'ls model index was not set!'
        if('rans2p' in idxDict):
            V_model = self.fetchIndex(idxDict, 'rans2p')
        elif ('rans3p' in idxDict):
            V_model = self.fetchIndex(idxDict, 'rans3p')
        else:
            assert mparams.rans2p.index is not None or mparams.rans3p.index is not None, 'RANS2P or RANS3PF must be used with VOF'
        #RD_model = mparams.rdls.index
        RD_model=self.fetchIndex(idxDict,'rdls')
        coeffs = self.p.coefficients
        coeffs.flowModelIndex = V_model
        coeffs.RD_modelIndex = RD_model
        coeffs.modelIndex = ME_model
        coeffs.movingDomain = self.p.movingDomain
        coeffs.initialize()

        # BOUNDARY CONDITIONS
        BC = self._Problem.SystemPhysics.boundaryConditions
        if self.p.dirichletConditions is None or len(self.p.dirichletConditions) == 0:
            if domain.useSpatialTools is False or self._Problem.SystemPhysics.useBoundaryConditionsModule is False:
                if 'ncls_DBC' in BC:
                    self.p.dirichletConditions = {0: BC['ncls_DBC']}
                else:
                    self.p.dirichletCondtions = {0: lambda x,t: None}
            else:
                self.p.dirichletConditions = {0: lambda x, flag: domain.BCbyFlag[flag].phi_dirichlet.uOfXT}
            self.p.diffusiveFluxBoundaryConditions = {0: {}}

    def _initializeNumerics(self):
        domain = self._Problem.domain
        nd = domain.nd
        # TIME
        self.n.timeIntegration = TimeIntegration.BackwardEuler_cfl
        self.n.stepController = StepControl.Min_dt_cfl_controller
        # FINITE ELEMENT SPACES
        FESpace = self._Problem.FESpace
        self.n.femSpaces = {0: FESpace['lsBasis']}
        # NUMERICAL FLUX
        self.n.subgridError = NCLS.SubgridError(coefficients=self.p.coefficients,
                                                nd=nd)
        scopts = self.n.ShockCapturingOptions
        self.n.shockCapturing = NCLS.ShockCapturing(coefficients=self.p.coefficients,
                                                    nd=nd,
                                                    shockCapturingFactor=scopts.shockCapturingFactor,
                                                    lag=scopts.lag)
        # TOLERANCES
        meshOptions = self._Problem.domain.MeshOptions
        if self.n.nl_atol_res is None:
            self.n.nl_atol_res = max(minTol, 0.001*meshOptions.he**2)
        if self.n.l_atol_res is None:
            self.n.l_atol_res = 0.001*self.n.nl_atol_res

    def _initializePETScOptions(self):
        prefix = self.n.linear_solver_options_prefix
        if self._Problem.SystemNumerics.useSuperlu:
            self.OptDB.setValue(prefix+'ksp_type', 'preonly')
            self.OptDB.setValue(prefix+'pc_type', 'lu')
            self.OptDB.setValue(prefix+'pc_factor_mat_solver_type', 'superlu_dist')
        else:
            self.OptDB.setValue(prefix+'ksp_type', 'gmres')
            self.OptDB.setValue(prefix+'pc_type', 'hypre')
            self.OptDB.setValue(prefix+'pc_pc_hypre_type', 'boomeramg')
            self.OptDB.setValue(prefix+'ksp_gmres_restart', 300)
            self.OptDB.setValue(prefix+'ksp_knoll', 1)
            self.OptDB.setValue(prefix+'ksp_max_it', 2000)

class ParametersModelRDLS(ParametersModelBase):
    """
    """
    def __init__(self,ProblemInstance):
        super(ParametersModelRDLS, self).__init__(name='rdls',Problem=ProblemInstance)
        self.p.coefficients = RDLS.Coefficients(
            initialize=False,
            useMetrics=1.,
            epsFact=0.75,
        )

        scopts = self.n.ShockCapturingOptions
        scopts.shockCapturingFactor = 0.9
        scopts.lag = False
        scopts._freeze()
        # LEVEL MODEL
        self.p.LevelModelType = RDLS.LevelModel
        # TIME INTEGRATION
        self.n.timeIntegration = TimeIntegration.NoIntegration
        self.n.stepController = StepControl.Newton_controller
        # NONLINEAR SOLVERS
        self.n.multilevelNonlinearSolver = NonlinearSolvers.Newton
        self.n.nonlinearSmoother = NonlinearSolvers.NLGaussSeidel
        # NUMERICAL FLUX
        self.n.numericalFluxType = NumericalFlux.DoNothing
        # LINEAR ALGEBRA
        self.n.multilevelLinearSolver = LinearSolvers.KSP_petsc4py
        self.n.levelLinearSolver = LinearSolvers.KSP_petsc4py
        self.n.linear_solver_options_prefix = 'rdls_'
        self.n.linearSolverConvergenceTest = 'r-true'
        #self.n.nonlinearSolverConvergenceTest = 'rits'
        #self.n.levelNonlinearSolverConvergenceTest = 'rits'
        # TOLERANCES
        self.n.tolFac = 0.
        #self.n.maxNonlinearIts = 1
        self.n.maxNonlinearIts = 50
        self.n.maxLineSearches = 0
        # freeze attributes
        self._freeze()
        self.setInitialConditionStructure() 

    def _initializePhysics(self):
        # MODEL INDEXING
        idxDict = self._Problem.SystemPhysics._modelIdxDict
        nModelId = self.fetchIndex(idxDict, 'ncls')
        assert nModelId is not None, 'ncls model index was not set!'
        rdModelId = self.fetchIndex(idxDict, self.name)

        # COEFFICIENTS
        coeffs = self.p.coefficients
        coeffs.nModelId = nModelId
        coeffs.rdModelId = rdModelId
        coeffs.initialize()
        
        # BOUNDARY CONDITIONS
        self.p.dirichletConditions = {0: lambda x, flag: None}
        self.p.weakDirichletConditions = {0: RDLS.setZeroLSweakDirichletBCsSimple}
        self.p.advectiveFluxBoundaryConditions = {}
        self.p.diffusiveFluxBoundaryConditions = {0: {}}

    def _initializeNumerics(self):
        domain = self._Problem.domain
        nd = domain.nd
        # FINITE ELEMENT SPACES
        FESpace = self._Problem.FESpace
        self.n.femSpaces = {0: FESpace['lsBasis']}
        # NON LINEAR SOLVER
        self.n.nonlinearSmoother = NonlinearSolvers.NLGaussSeidel
        # NUMERICAL FLUX
        self.n.subgridError = RDLS.SubgridError(coefficients=self.p.coefficients,
                                                nd=nd)
        scopts = self.n.ShockCapturingOptions
        self.n.shockCapturing = RDLS.ShockCapturing(coefficients=self.p.coefficients,
                                                    nd=nd,
                                                    shockCapturingFactor=scopts.shockCapturingFactor,
                                                    lag=scopts.lag)
        # TOLERANCES
        meshOptions = self._Problem.domain.MeshOptions
        if self.n.nl_atol_res is None:
            self.n.nl_atol_res = max(minTol, 0.1*meshOptions.he)
        if self.n.l_atol_res is None:
            self.n.l_atol_res = 0.001*self.n.nl_atol_res

    def _initializePETScOptions(self):
        prefix = self.n.linear_solver_options_prefix
        if self._Problem.SystemNumerics.useSuperlu:
            self.OptDB.setValue(prefix+'ksp_type', 'preonly')
            self.OptDB.setValue(prefix+'pc_type', 'lu')
            self.OptDB.setValue(prefix+'pc_factor_mat_solver_type', 'superlu_dist')
        else:
            self.OptDB.setValue(prefix+'ksp_type', 'gmres')
            self.OptDB.setValue(prefix+'pc_type', 'asm')
            self.OptDB.setValue(prefix+'pc_pc_asm_type', 'basic')
            self.OptDB.setValue(prefix+'ksp_gmres_modifiedgramschmidt', 1)
            self.OptDB.setValue(prefix+'ksp_gmres_restart', 300)
            self.OptDB.setValue(prefix+'ksp_knoll', 1)
            self.OptDB.setValue(prefix+'sub_ksp_type', 'preonly')
            self.OptDB.setValue(prefix+'sub_pc_factor_mat_solver_type', 'superlu')
            self.OptDB.setValue(prefix+'sub_pc_type', 'lu')
            self.OptDB.setValue(prefix+'max_it', 2000)

class ParametersModelMCorr(ParametersModelBase):
    """
    """
    def __init__(self,ProblemInstance):
        super(ParametersModelMCorr, self).__init__(name='mcorr',Problem=ProblemInstance)
        self.p.coefficients = MCorr.Coefficients(
            initialize=False,
            useMetrics=1.,
            checkMass=False,
            epsFactHeaviside=epsFact,
            epsFactDirac=epsFact,
            epsFactDiffusion=10.,
        )

        # LEVEL MODEL
        self.p.LevelModelType = MCorr.LevelModel
        # TIME
        self.n.timeIntegration = TimeIntegration.BackwardEuler_cfl
        self.n.stepController  = StepControl.Min_dt_cfl_controller
        # NUMERICAL FLUX
        self.n.numericalFluxType = NumericalFlux.DoNothing
        # NON LINEAR SOLVER
        self.n.multilevelNonlinearSolver = NonlinearSolvers.Newton
        # LINEAR ALGEBRA
        self.n.multilevelLinearSolver = LinearSolvers.KSP_petsc4py
        self.n.levelLinearSolver = LinearSolvers.KSP_petsc4py
        self.n.linear_solver_options_prefix = 'mcorr_'
        self.n.linearSolverConvergenceTest = 'r-true'
        #self.n.nonlinearSolverConvergenceTest = 'rits'
        #self.n.levelNonlinearSolverConvergenceTest = 'rits'
        # TOLERANCES
        self.n.linTolFac = 0.
        self.n.tolFac = 0.
        self.n.maxNonlinearIts = 50
        self.n.maxLineSearches = 0
        self.n.useEisenstatWalker = True
        # freeze attributes
        self._freeze()
        self.setInitialConditionStructure()

    def _initializePhysics(self):
        domain = self._Problem.domain
        nd = domain.nd
        # MODEL INDEXING
        idxDict = self._Problem.SystemPhysics._modelIdxDict
        ME_model = self.fetchIndex(idxDict, self.name)
        assert ME_model is not None, 'mcorr model index was not set!'
        LS_model = self.fetchIndex(idxDict, 'ncls')
        VOF_model = self.fetchIndex(idxDict, 'vof')

        if('rans2p' in idxDict):
            V_model = self.fetchIndex(idxDict, 'rans2p')
        elif ('rans3p' in idxDict):
            V_model = self.fetchIndex(idxDict, 'rans3p')
        else:
            assert False, 'RANS2P or RANS3PF must be used with VOF'

        # COEFFICIENTS
        coeffs = self.p.coefficients
        coeffs.flowModelIndex = V_model
        coeffs.me_model = ME_model
        coeffs.VOFModelIndex = VOF_model
        coeffs.levelSetModelIndex = LS_model
        coeffs.nd = nd
        coeffs.initialize()

        # BOUNDARY CONDITIONS
        # N/A

    def _initializeNumerics(self):
        # TIME
        self.n.timeIntegration = TimeIntegration.BackwardEuler_cfl
        self.n.stepController = StepControl.Min_dt_cfl_controller
        # FINITE ELEMENT SPACES
        FESpace = self._Problem.FESpace
        self.n.femSpaces = {0: FESpace['lsBasis']}
        # TOLERANCES
        meshOptions = self._Problem.domain.MeshOptions
        if self.n.nl_atol_res is None:
            self.n.nl_atol_res = max(minTol, 0.0001*meshOptions.he**2)
        if self.n.l_atol_res is None:
            self.n.l_atol_res = 0.001*self.n.nl_atol_res
        
    def _initializePETScOptions(self):
        prefix = self.n.linear_solver_options_prefix
        if self._Problem.SystemNumerics.useSuperlu:
            self.OptDB.setValue(prefix+'ksp_type', 'preonly')
            self.OptDB.setValue(prefix+'pc_type', 'lu')
            self.OptDB.setValue(prefix+'pc_factor_mat_solver_type', 'superlu_dist')
        else:
            self.OptDB.setValue(prefix+'ksp_type', 'fgmres')
            self.OptDB.setValue(prefix+'pc_type', 'hypre')
            self.OptDB.setValue(prefix+'pc_pc_hypre_type', 'boomeramg')
            self.OptDB.setValue(prefix+'ksp_max_it', 2000)

class ParametersModelAddedMass(ParametersModelBase):
    """
    """
    def __init__(self, ProblemInstance):
        super(ParametersModelAddedMass, self).__init__(name='addedMass', Problem=ProblemInstance)

        self.p.coefficients = AddedMass.Coefficients(
            initialize=False,
        )

        # LEVEL MODEL
        self.p.LevelModelType = AddedMass.LevelModel
        # TIME
        self.n.timeIntegration = TimeIntegration.BackwardEuler_cfl
        self.n.stepController = StepControl.Min_dt_cfl_controller
        # NONLINEAR SOLVER
        self.n.multilevelNonlinearSolver = NonlinearSolvers.AddedMassNewton
        self.n.levelNonlinearSolver = NonlinearSolvers.AddedMassNewton
        self.n.nonlinearSmoother = NonlinearSolvers.AddedMassNewton
        # NUMERICAL FLUX
        self.n.numericalFluxType = AddedMass.NumericalFlux
        # LINEAR ALGEBRA
        self.n.linearSmoother = LinearSolvers.NavierStokesPressureCorrection
        self.n.multilevelLinearSolver = LinearSolvers.KSP_petsc4py
        self.n.levelLinearSolver = LinearSolvers.KSP_petsc4py
        self.n.linear_solver_options_prefix = 'am_'
        self.n.linearSolverConvergenceTest = 'r-true'
        #self.n.nonlinearSolverConvergenceTest = 'rits'
        #self.n.levelNonlinearSolverConvergenceTest = 'rits'
        # TOLERANCES
        self.n.linTolFac = 0.
        self.n.tolFac = 0.
        self.n.maxNonlinearIts = 1
        self.n.maxLineSearches = 0
        # freeze attributes
        self._freeze()

        #Initial Conditions
        self.setInitialConditionStructure()

    def _initializePhysics(self):
        domain = self._Problem.domain
        nd = domain.nd
        # MODEL INDEXING
        idxDict = self._Problem.SystemPhysics._modelIdxDict
        if self.fetchIndex(idxDict,'rans2p') is not None:
            V_model = self.fetchIndex(idxDict,'rans2p')
        elif self.fetchIndex(idxDict,'rans3p') is not None:
            V_model = self.fetchIndex(idxDict,'rans3p')
        else:
            assert self.fetchIndex(idxDict,'rans2p') is not None or self.fetchIndex(idxDict,'rans3p') is not None, 'RANS2P or RANS3PF must be used with addedMass'
        # COEFFICIENTS
        coeffs = self.p.coefficients
        coeffs.flowModelIndex = V_model
        coeffs.barycenters = domain.barycenters
        coeffs.nd = nd
        coeffs.initialize()

        # BOUNDARY CONDITIONS
        BC = self._Problem.SystemPhysics.boundaryConditions
        self.p.dirichletConditions = {0: lambda x, flag: domain.BCbyFlag[flag].pAddedMass_dirichlet.uOfXT}
        self.p.advectiveFluxBoundaryConditions = {}
        def getFlux_am(x, flag):
            #the unit rigid motions will applied internally
            #leave this set to zero
            return lambda x,t: 0.0
        self.p.diffusiveFluxBoundaryConditions = {0: {0: getFlux_am}}

    def _initializeNumerics(self):
        # FINITE ELEMENT SPACES
        FESpace = self._Problem.FESpace
        self.n.femSpaces = {0: FESpace['lsBasis']}
        # TOLERANCES
        meshOptions = self._Problem.domain.MeshOptions
        if self.n.nl_atol_res is None:
            self.n.nl_atol_res = max(minTol, 0.0001*meshOptions.he**2)
        if self.n.l_atol_res is None:
            self.n.l_atol_res = 0.001*self.n.nl_atol_res
        #override LU selection, even in serial
        self.n.multilevelLinearSolver = LinearSolvers.KSP_petsc4py
        self.n.levelLinearSolver = LinearSolvers.KSP_petsc4py

    def _initializePETScOptions(self):
        prefix = self.n.linear_solver_options_prefix
        self.OptDB.setValue(prefix+'ksp_type', 'cg')
        self.OptDB.setValue(prefix+'pc_type', 'gamg')
        self.OptDB.setValue(prefix+'ksp_max_it', 2000)

class ParametersModelMoveMeshMonitor(ParametersModelBase):
    """
    """
    def __init__(self, ProblemInstance):
        super(ParametersModelMoveMeshMonitor, self).__init__(name='moveMeshMonitor',                                                             Problem=ProblemInstance)
        self.p.coefficients = MoveMeshMonitor.Coefficients(
            initialize=False,
            ME_MODEL=None,
            func=lambda x, t: 1000.,
            he_min=0.,
            he_max=1000.,
            epsFact_density=epsFact,
            epsTimeStep=0.1,
            grading_type=2,
            useLS=True,
        )

        # TIME INTEGRATION
        self.n.timeIntegration = TimeIntegration.NoIntegration
        # NONLINEAR SOLVER
        self.n.multilevelNonlinearSolver = NonlinearSolvers.MoveMeshMonitorNewton
        self.n.levelNonlinearSolver = NonlinearSolvers.MoveMeshMonitorNewton
        self.n.nonlinearSmoother = NonlinearSolvers.MoveMeshMonitorNewton
        # NUMERICAL FLUX
        self.n.numericalFluxType = NumericalFlux.Diffusion_SIPG_exterior
        # LINEAR ALGEBRA
        self.n.linearSmoother = LinearSolvers.NavierStokesPressureCorrection
        self.n.multilevelLinearSolver = LinearSolvers.KSP_petsc4py
        self.n.levelLinearSolver = LinearSolvers.KSP_petsc4py
        self.n.linear_solver_options_prefix = 'mesh2_'
        self.n.linearSolverConvergenceTest = 'r-true'
        # TOLERANCES
        self.n.linTolFac = 0.
        self.n.tolFac = 0.
        self.n.maxNonlinearIts = 1
        self.n.maxLineSearches = 0
        # freeze attributes
        self._freeze()

        #Initial Conditions
        self.setInitialConditionStructure()

    def _initializePhysics(self):
        domain = self._Problem.domain
        nd = domain.nd
        # MODEL INDEXING
        idxDict = self._Problem.SystemPhysics._modelIdxDict
        ME_MODEL = self.index
        assert ME_MODEL is not None, 'moveMeshMonitor model index was not set!'
        if self.p.coefficients.useLS is True:
            LS_MODEL = mparams.ncls.index
        else:
            LS_MODEL = None
        # COEFFICIENTS
        coeffs = self.p.coefficients
        coeffs.LS_MODEL = LS_MODEL
        coeffs.ME_MODEL = ME_MODEL
        coeffs.nd = nd
        coeffs.initialize()

        # BOUNDARY CONDITIONS
        BC = self._Problem.SystemPhysics.boundaryConditions
        self.p.dirichletConditions = {0: lambda x, flag: None}
        # self.p.advectiveFluxBoundaryConditions = {}
        self.p.diffusiveFluxBoundaryConditions = {0: {0: lambda x, flag: lambda x, t: 0.}}

    def _initializeNumerics(self):
        # FINITE ELEMENT SPACES
        FESpace = self._Problem.FESpace
        self.n.femSpaces = {0: FESpace['lsBasis']}
        # TOLERANCES
        meshOptions = self._Problem.domain.MeshOptions
        if self.n.nl_atol_res is None:
            self.n.nl_atol_res = max(minTol, 0.0001*meshOptions.he**2)
        if self.n.l_atol_res is None:
            self.n.l_atol_res = 0.001*self.n.nl_atol_res

    def _initializePETScOptions(self):
        prefix = self.n.linear_solver_options_prefix
        self.OptDB.setValue(prefix+'ksp_type', 'cg')
        self.OptDB.setValue(prefix+'pc_type', 'hypre')
        self.OptDB.setValue(prefix+'pc_hypre_type', 'boomeramg')
        # self.OptDB.setValue(prefix+'ksp_constant_null_space', 1)
        # self.OptDB.setValue(prefix+'pc_factor_shift_type', 'NONZERO')
        # self.OptDB.setValue(prefix+'pc_factor_shift_amount', 1e-10)
        # self.OptDB.setValue(prefix+'ksp_max_it', 2000)


class ParametersModelMoveMeshElastic(ParametersModelBase):
    """
    """
    def __init__(self, ProblemInstance):
        super(ParametersModelMoveMeshElastic, self).__init__(name='moveMeshElastic',                                                              Problem=ProblemInstance)
        self.p.coefficients = MoveMesh.Coefficients(
            nd = self._Problem.domain.nd,
            initialize=False,
            modelType_block=None,
            modelParams_block=None,
        )

        # LEVEL MODEL
        self.p.LevelModelType = MoveMesh.LevelModel
        # TIME INTEGRATION
        self.n.timeIntegration = TimeIntegration.NoIntegration
        # NONLINEAR SOLVER
        self.n.multilevelNonlinearSolver = NonlinearSolvers.Newton
        # NUMERICAL FLUX
        self.n.numericalFluxType = NumericalFlux.Stress_IIPG_exterior
        # LINEAR ALGEBRA
        self.n.multilevelLinearSolver = LinearSolvers.KSP_petsc4py
        self.n.levelLinearSolver = LinearSolvers.KSP_petsc4py
        self.n.linear_solver_options_prefix = 'mesh_'
        self.n.linearSolverConvergenceTest = 'r-true'
        # TOLERANCES
        self.n.tolFac = 0.
        self.n.linTolFac = 0.
        self.n.maxNonlinearIts = 4
        self.n.maxLineSearches = 0
        # freeze attributes
        self._freeze()

        #Initial Conditions
        self.setInitialConditionStructure()

    def _initializePhysics(self):
        domain = self._Problem.domain
        nd = domain.nd
        # NUM PARAMS
        nMediaTypes = len(domain.regionFlags)  # (!) should be region flags
        smTypes = np.zeros((nMediaTypes+1, 2), 'd')
        smFlags = np.zeros((nMediaTypes+1,), 'i')
        smTypes[:, 0] = 1.
        smTypes[:, 1] = 0.3
        # MODEL INDEXING
        idxDict = self._Problem.SystemPhysics._modelIdxDict
        ME_model = self.fetchIndex(idxDict,'moveMeshElastic')
        assert ME_model is not None, 'vof model index was not set!'
        if self.fetchIndex(idxDict,'rans2p') is not None:
            V_model = self.fetchIndex(idxDict,'rans2p')
        elif self.fetchIndex(idxDict,'rans3p') is not None:
            V_model = self.fetchIndex(idxDict,'rans3p')
        else:
            assert self.fetchIndex(idxDict,'rans2p') is not None or self.fetchIndex(idxDict,'rans3p') is not None, 'RANS2P or RANS3PF must be used with VOF'
        # COEFFICIENTS
        coeffs = self.p.coefficients
        coeffs.flowModelIndex = V_model
        coeffs.meIndex = ME_model
        coeffs.modelType_block = smFlags
        coeffs.modelParams_block = smTypes
        coeffs.nd = nd
        coeffs.initialize()

        # BOUNDARY CONDITIONS
        BC = self._Problem.SystemPhysics.boundaryConditions
        if domain.useSpatialTools is False or self._Problem.SystemPhysics.useBoundaryConditionsModule is False:
            self.p.dirichletConditions = {0: BC['hx'],
                                          1: BC['hy']}
            self.p.stressFluxBoundaryConditions = {0: BC['u_stress'],
                                                   1: BC['v_stress']}
            if nd == 3:
                self.p.dirichletConditions[2] = BC['hz']
                self.p.stressFluxBoundaryConditions[2] = BC['w_stress']

        else:
            self.p.dirichletConditions = {0: lambda x, flag: domain.BCbyFlag[flag].hx_dirichlet.uOfXT,
                                          1: lambda x, flag: domain.BCbyFlag[flag].hy_dirichlet.uOfXT}
            self.p.stressFluxBoundaryConditions = {0: lambda x, flag: domain.BCbyFlag[flag].u_stress.uOfXT,
                                                   1: lambda x, flag: domain.BCbyFlag[flag].v_stress.uOfXT}
            if nd == 3:
                self.p.dirichletConditions[2] = lambda x, flag: domain.BCbyFlag[flag].hz_dirichlet.uOfXT
                self.p.stressFluxBoundaryConditions[2] = lambda x, flag: domain.BCbyFlag[flag].w_stress.uOfXT
        self.p.fluxBoundaryConditions = {0: 'noFlow',
                                         1: 'noFlow'}
        self.p.advectiveFluxBoundaryConditions = {}

        self.p.diffusiveFluxBoundaryConditions = {0: {},
                                                  1: {}}
        if nd == 3:
            self.p.fluxBoundaryConditions[2] = 'noFlow'
            self.p.diffusiveFluxBoundaryConditions[2] = {}

    def _initializeNumerics(self):
        domain = self._Problem.domain
        nd = domain.nd
        # FINITE ELEMENT SPACES
        FESpace = self._Problem.FESpace
        self.n.femSpaces = {0: FESpace['velBasis'],
                            1: FESpace['velBasis']}
        if nd == 3:
            self.n.femSpaces[2] = FESpace['velBasis']
        # TOLERANCES
        meshOptions = self._Problem.domain.MeshOptions
        if self.n.nl_atol_res is None:
            self.n.nl_atol_res = max(minTol, 0.0001*meshOptions.he**2)
        if self.n.l_atol_res is None:
            self.n.l_atol_res = 0.1*self.n.nl_atol_res

    def _initializePETScOptions(self):
        prefix = self.n.linear_solver_options_prefix
        if self._Problem.SystemNumerics.useSuperlu:
            self.OptDB.setValue(prefix+'ksp_type', 'preonly')
            self.OptDB.setValue(prefix+'pc_type', 'lu')
            self.OptDB.setValue(prefix+'pc_factor_mat_solver_type', 'superlu_dist')
        else:
            self.OptDB.setValue(prefix+'ksp_type', 'cg')
            self.OptDB.setValue(prefix+'pc_type', 'asm')
            self.OptDB.setValue(prefix+'pc_asm_type', 'basic')
            self.OptDB.setValue(prefix+'ksp_max_it', 2000)
            self.OptDB.setValue(prefix+'sub_ksp_type', 'preonly')
            self.OptDB.setValue(prefix+'sub_pc_factor_mat_solver_type', 'superlu')
            self.OptDB.setValue(prefix+'ksp_knoll', 1)
            self.OptDB.setValue(prefix+'sub_pc_type', 'lu')
