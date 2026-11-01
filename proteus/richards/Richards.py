from __future__ import division
from builtins import range
import proteus
from .cRichards import *
import numpy as np
from proteus.Transport import OneLevelTransport
from proteus.Transport import TC_base, NonlinearEquation, logEvent, memory
from proteus.Transport import FluxBoundaryConditions, Comm, cfemIntegrals
from proteus.Transport import DOFBoundaryConditions, Quadrature
from proteus.mprans import cArgumentsDict
from proteus.LinearAlgebraTools import SparseMat
from proteus import TimeIntegration
from proteus.NonlinearSolvers import ExplicitLumpedMassMatrixForRichards
#from proteus.NonlinearSolvers import Newton


class ThetaScheme(TimeIntegration.BackwardEuler):
    """Two-level theta method for the Richards LevelModel.

    The storage term is always (m(u^{n+1}) - m(u^n))/dt; the Galerkin flux terms
    are weighted theta*F(u^{n+1}) + (1-theta)*F(u^n). theta = 1 (default) is
    backward Euler and theta = 1/2 is Crank-Nicolson. Set it in the numerics
    file with ``timeIntegrationTheta = 0.5``.

    With STABILIZATION_TYPE=0 the theta weighting applies to the Galerkin
    element integrals (exterior boundary fluxes stay implicit). With the
    edge-based schemes the low-order operator stays backward Euler and theta
    enters only the high-order flux used to build the FCT correction.
    """
    def __init__(self,transport,integrateInterpolationPoints=False,theta=1.0):
        self.transport=transport
        TimeIntegration.BackwardEuler.__init__(self,transport, integrateInterpolationPoints)
        self.theta = theta
    def setFromOptions(self,nOptions):
        TimeIntegration.BackwardEuler.setFromOptions(self,nOptions)
        if 'timeIntegrationTheta' in dir(nOptions):
            self.theta = float(nOptions.timeIntegrationTheta)
        assert 0.0 <= self.theta <= 1.0, "timeIntegrationTheta must be in [0,1]"
    def updateTimeHistory(self,resetFromDOF=False):
        TimeIntegration.BackwardEuler.updateTimeHistory(self,resetFromDOF)
        self.transport.u_dof_old[:] = self.u
class RKEV(TimeIntegration.SSP):
    from proteus import TimeIntegration
    """
    Wrapper for SSPRK time integration using EV

    ... more to come ...
    """

    def __init__(self, transport, timeOrder=1, runCFL=0.1, integrateInterpolationPoints=False):
        TimeIntegration.SSP.__init__(self, transport, integrateInterpolationPoints=integrateInterpolationPoints)
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
                    #print()
        

    # def set_dt(self, DTSET):
    #    self.dt = DTSET #  don't update t
    def choose_dt(self):
        comm = Comm.get()
        maxCFL = 1.0e-6
        maxCFL = max(maxCFL, comm.globalMax(self.cfl.max()))
        self.dt = self.runCFL/ maxCFL
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
                    self.u_dof_stage[ci][self.lstage] *= 1./ 4.
                    self.u_dof_stage[ci][self.lstage] += 3. / 4. * self.u_dof_last[ci]
                    # Update u_dof_old
                    self.transport.u_dof_old[:] = self.u_dof_stage[ci][self.lstage]
            elif self.lstage == 3:
                logEvent("Third stage of SSP33 method", level=4)
                for ci in range(self.nc):
                    self.u_dof_stage[ci][self.lstage][:] = self.transport.u[ci].dof
                    self.u_dof_stage[ci][self.lstage] *= 2.0/ 3.0
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
                    self.u_dof_stage[ci][self.lstage][:] *= 1. / 2.
                    self.u_dof_stage[ci][self.lstage][:] += 1. / 2. * self.u_dof_last[ci]
                    # update u_dof_old
                    self.transport.u_dof_old[:] = self.u_dof_last[ci]
                    # update solution to u[0].dof
                    self.transport.u[ci].dof[:] = self.u_dof_stage[ci][self.lstage]
        else:
            assert self.timeOrder == 1
            for ci in range(self.nc):
                self.u_dof_stage[ci][self.lstage][:] = self.transport.u[ci].dof[:]
                self.transport.u_dof_old[:] = self.transport.u[ci].dof
                

    def initializeTimeHistory(self, resetFromDOF=True):
        """
        Push necessary information into time history arrays
        """
        for ci in range(self.nc):
            self.u_dof_last[ci][:] = self.transport.u[ci].dof[:]
            for k in range(self.nStages):
                self.u_dof_stage[ci][k][:] = self.transport.u[ci].dof[:]

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
    """
    version of Re where element material type id's used in evals
    """
    from proteus.ctransportCoefficients import conservativeHeadRichardsMualemVanGenuchtenHetEvaluateV2
    from proteus.ctransportCoefficients import conservativeHeadRichardsMualemVanGenuchten_sd_het
    def __init__(self,
                 nd,
                 Ksw_types,
                 vgm_n_types,
                 vgm_alpha_types,
                 thetaR_types,
                 thetaSR_types,
                 gravity,
                 density,
                 beta,
                 # PORE SIZE DISTRIBUTION (RETENTION / RELPERM) MODEL
                 PSK_type='VG',
                 diagonal_conductivity=True,
                 getSeepageFace=None,
                 # WEAK (IIPG) DIRICHLET PENALTY
                 # Overrides NumericalFlux.Richards_IIPG_exterior's class default
                 # for this model only; None leaves the numerical flux alone.
                 penalty_constant=None,
                 penalty_power=None,
                 density_model=None,
                 DENSITY_MODEL=None,
                # FOR EDGE BASED EV
                 STABILIZATION_TYPE='Implicit_FCT',
                 ENTROPY_TYPE=2,  # logarithmic
                 LUMPED_MASS_MATRIX=False,
                 MONOLITHIC=False,
                 VMS=0.0,
                 SC=0.0,
                 FCT=True,
                 num_fct_iter=1,
                 FCT_SATURATED_LOW_ORDER=True,
                 FCT_TYPE='semi-implicit',
                 FCT_PRELIMITER_POWER=3.0,
                 FCT_PRELIMITER_VARIABLE='pressure',
                 LSCHEME_L=0.0,
                 NONLINEAR_SOLVER=None,
                 LSCHEME_RETRY_ITS=1,
                 LSCHEME_TRIGGER='residual',
                 LSCHEME_THETA_MAX=0.5,
                 FCT_PREDICTOR_FALLBACK=True,
                 # FOR ENTROPY VISCOSITY
                 cE=1.0,
                 uL=0.0,
                 uR=1.0,
                 # FOR ARTIFICIAL COMPRESSION
                 cK=1.0,
                 # OUTPUT quantDOFs
                 outputQuantDOFs=False,
                 forceStrongConditions = False):
        self.VMS=VMS
        if density_model is None:
            density_model = DENSITY_MODEL
        self.density_model = density_model
        self.modelIndex=1
        self.SC=SC
        self.anb_seepage_flux= 0.00
        #self.anb_seepage_flux_n =0.0
        variableNames=['pressure_head']
        nc=1
        mass={0:{0:'nonlinear'}}
        advection={0:{0:'nonlinear'}}
        diffusion={0:{0:{0:'nonlinear'}}}
        potential={0:{0:'u'}}
        reaction={0:{0:'linear'}}
        hamiltonian={}
        self.getSeepageFace=getSeepageFace
        # Read back in LevelModel.__init__ right after the numerical flux is
        # built, before ebq_global['penalty']/ebqe['penalty'] are filled.
        self.penalty_constant=penalty_constant
        self.penalty_power=penalty_power
        self.gravity=gravity
        self.rho = density
        self.beta=beta
        self.vgm_n_types = vgm_n_types
        self.vgm_alpha_types = vgm_alpha_types
        self.thetaR_types    = thetaR_types
        self.thetaSR_types   = thetaSR_types
        # Pore size distribution model used for the retention curve theta(psi)
        # and the relative permeability kr(theta).  Default 'VG' reproduces the
        # existing behaviour exactly: van Genuchten retention with the Mualem
        # kr.  The integer code is handed to the kernel as "PSK_TYPE" and picks
        # the closure in proteus/pskRelations.h.  Under 'BC' the second
        # numeric parameter (vgm_n_types) is read as the pore-size index lambda
        # and vgm_alpha_types as the inverse entry-pressure head 1/p_d.
        #
        # The two 'BC' codes share one retention curve and differ only in the
        # exponent of kr = Se**eta, which lambda does not fix: Burdine gives
        # (2+3*lambda)/lambda, Mualem 2.5+2/lambda.  At lambda=0.592 that is
        # 6.378 against 5.878, a factor ~4 in kr by psiC = 7.5 m, so the choice
        # has to follow whichever parameter set is being reproduced.  'BC' keeps
        # Burdine, the historical behaviour.
        #
        # 'Gardner' is the quasi-linear closure Tracy's analytical solutions are
        # derived from: theta = thetaR + thetaSR*exp(alpha*psi) with
        # kr = exp(alpha*psi), i.e. K = Ks*exp(alpha*psi).  It is one model, not
        # a retention/kr pair that can be mixed -- kr linear in Se is what makes
        # the Kirchhoff transform hbar = exp(alpha*psi) linearise Richards and
        # the closed forms exist.  alpha is Gardner's exponent [1/m] and
        # vgm_n_types is unused.
        #
        # 'HornungMessing' is the manufactured law s = thetaR + thetaSR*(1 -
        # (4/pi^2) atan(psi)^2), kr = 1/(1+psi^2) for psi < 0 (saturated above).
        # Use thetaR = 0, thetaSR = pi^2/2, Ks = 2 for the published test;
        # alpha and vgm_n_types are unused.
        psk_types = {"VG": 0,          # van Genuchten retention + Mualem  kr
                     "BC": 1,          # Brooks-Corey  retention + Burdine kr
                     "BC_MUALEM": 2,   # Brooks-Corey  retention + Mualem  kr
                     "Gardner": 3,     # Gardner exponential retention + kr=Se
                     "HornungMessing": 4}
        try:
            if isinstance(PSK_type, int):
                PSK_type = [key for key, value in psk_types.items() if value == PSK_type][0]
            self.PSK_type = psk_types[PSK_type]
        except:
            raise ValueError("PSK_type must be one of "+str(list(psk_types.keys()))+" not "+str(PSK_type))
        self.elementMaterialTypes = None
        self.exteriorElementBoundaryTypes  = None
        self.materialTypes_q    = None
        self.materialTypes_ebq  = None
        self.materialTypes_ebqe  = None
        self.nd = nd
        self.nMaterialTypes = len(thetaR_types)
        self.q = {}; self.ebqe = {}; self.ebq = {}; self.ebq_global={}
        #try to allow some flexibility in input of permeability/conductivity tensor
        self.diagonal_conductivity = diagonal_conductivity
        self.Ksw_types_in = Ksw_types
        if self.diagonal_conductivity:
            sparseDiffusionTensors = {(0,0):(np.arange(self.nd+1,dtype='i'),
                                             np.arange(self.nd,dtype='i'))}

            assert len(Ksw_types.shape) in [1,2], "if diagonal conductivity true then Ksw_types scalar or vector of diagonal entries"
            #allow scalar input Ks
            if len(Ksw_types.shape)==1:
                self.Ksw_types = np.zeros((self.nMaterialTypes,self.nd),'d')
                for I in range(self.nd):
                    self.Ksw_types[:,I] = Ksw_types
            else:
                self.Ksw_types = Ksw_types
        else: #full
            sparseDiffusionTensors = {(0,0):(np.arange(self.nd**2+1,step=self.nd,dtype='i'),
                                             np.array([list(range(self.nd)) for row in range(self.nd)],dtype='i'))}
            assert len(Ksw_types.shape) in [1,2], "if full tensor conductivity true then Ksw_types scalar or 'flattened' row-major representation of entries"
            if len(Ksw_types.shape)==1:
                self.Ksw_types = np.zeros((self.nMaterialTypes,self.nd**2),'d')
                for I in range(self.nd):
                    self.Ksw_types[:,I*self.nd+I] = Ksw_types
            else:
                assert Ksw_types.shape[1] == self.nd**2
                self.Ksw_types = Ksw_types

        stabilization_types = {"Galerkin":0, 
                               "EV_Stab":1, 
                               "EntropyViscosity":2, 
                               "Implicit_FCT":3}
        try:
            if isinstance(STABILIZATION_TYPE, int):
                STABILIZATION_TYPE = [key for key, value in stabilization_types.items() if value == STABILIZATION_TYPE][0]
            
            self.STABILIZATION_TYPE = stabilization_types[STABILIZATION_TYPE]
        except:
            raise ValueError("STABILIZATION_TYPE must be one of "+str(stabilization_types.keys())+" not "+STABILIZATION_TYPE)
        
        # EDGE BASED (AND ENTROPY) VISCOSITY
        self.LUMPED_MASS_MATRIX = LUMPED_MASS_MATRIX
        self.MONOLITHIC = MONOLITHIC
        #self.STABILIZATION_TYPE = STABILIZATION_TYPE
        self.ENTROPY_TYPE = ENTROPY_TYPE
        self.FCT = FCT
        self.num_fct_iter=num_fct_iter
        # When True (default), the FCT correction is switched off at DOFs whose low-order
        # predictor is saturated (u >= 0) for retention laws with a saturated
        # plateau (VG, BC, Hornung-Messing; not Gardner). There theta is pinned at
        # theta_S, so a limited mass below it cannot be inverted back to the
        # positive head and the theta -> psi recovery collapses to psi ~ 0-.
        self.FCT_SATURATED_LOW_ORDER = FCT_SATURATED_LOW_ORDER
        # FCT_TYPE (with FCT=True and an edge-based STABILIZATION_TYPE):
        #  'semi-implicit' -- Barua, Kees & Kuzmin (WRR): low-order solve, then an explicit
        #                     correction of theta and a theta -> psi inversion (FCTStep).
        #  'implicit'      -- Dielmann, Kuzmin & Kees (JNMA): Galerkin theta-scheme predictor p^H,
        #                     fluxes at p^H, nodal-variation prelimiter (exponent
        #                     FCT_PRELIMITER_POWER; <= 0 turns it off), Zalesak with the global bounds
        #                     [theta_R, theta_S] relative to theta^n, then an implicit low-order solve
        #                     from the corrected datum theta^{n,FCT}. No inversion.
        if FCT_TYPE not in ('semi-implicit', 'implicit'):
            raise ValueError("FCT_TYPE must be 'semi-implicit' or 'implicit', not %r" % (FCT_TYPE,))
        self.FCT_TYPE = FCT_TYPE
        self.FCT_PRELIMITER_POWER = FCT_PRELIMITER_POWER
        # Variable the implicit FCT nodal-variation detector acts on: 'pressure' (p^H, as in the
        # JNMA MFEM implementation) or 'hydraulic_head' (u^H = p^H + z, as first written in the
        # JNMA draft). They coincide without gravity.
        prelimiter_variables = {'pressure': 0, 'hydraulic_head': 1}
        if FCT_PRELIMITER_VARIABLE not in prelimiter_variables:
            raise ValueError("FCT_PRELIMITER_VARIABLE must be one of %s, not %r"
                             % (list(prelimiter_variables), FCT_PRELIMITER_VARIABLE))
        self.FCT_PRELIMITER_VARIABLE = FCT_PRELIMITER_VARIABLE
        self.FCT_PRELIMITER_HEAD = prelimiter_variables[FCT_PRELIMITER_VARIABLE]
        # Nonlinear iteration matrix. LSCHEME_L = L >= sup dm/dpsi defines the L-scheme matrix
        # J = (L/dt) M_L + A~(p^k) (low order, upwind k_r frozen), used for every residual (Galerkin,
        # low order, implicit-FCT final) as in the JNMA defect correction p^{k+1} = p^k - J^{-1} R(p^k).
        # NONLINEAR_SOLVER:
        #   'newton'  -- exact Jacobians (default when LSCHEME_L = 0)
        #   'lscheme' -- L-scheme matrix in every iteration (default when LSCHEME_L > 0); linear
        #                convergence, so raise maxNonlinearIts and set maxLineSearches = 0
        #   'hybrid'  -- Newton, but a step that fails the trigger is undone and replaced by
        #                LSCHEME_RETRY_ITS L-scheme iterations (instead of a line search), then Newton
        #                resumes. LSCHEME_TRIGGER: 'residual' (the Newton step did not reduce ||r||) or
        #                'theta' (also when Deuflhard's contraction estimate
        #                Theta_k = ||J_k^{-1} r(u_{k+1})|| / ||du_k|| exceeds LSCHEME_THETA_MAX;
        #                Theta ~ h_k/2 of the Kantorovich ball test, one extra back-solve per iteration).
        self.LSCHEME_L = LSCHEME_L
        if NONLINEAR_SOLVER is None:
            NONLINEAR_SOLVER = 'lscheme' if LSCHEME_L > 0.0 else 'newton'
        if NONLINEAR_SOLVER not in ('newton', 'lscheme', 'hybrid'):
            raise ValueError("NONLINEAR_SOLVER must be 'newton', 'lscheme' or 'hybrid'")
        if NONLINEAR_SOLVER != 'newton' and not LSCHEME_L > 0.0:
            raise ValueError("NONLINEAR_SOLVER=%r needs LSCHEME_L > 0" % (NONLINEAR_SOLVER,))
        if LSCHEME_TRIGGER not in ('residual', 'theta'):
            raise ValueError("LSCHEME_TRIGGER must be 'residual' or 'theta'")
        self.NONLINEAR_SOLVER = NONLINEAR_SOLVER
        self.LSCHEME_RETRY_ITS = int(LSCHEME_RETRY_ITS)
        self.LSCHEME_TRIGGER = LSCHEME_TRIGGER
        self.LSCHEME_THETA_MAX = float(LSCHEME_THETA_MAX)
        # FCT_TYPE='implicit': if the Galerkin predictor's nonlinear solve fails, redo the step's
        # predictor with the low-order operator (fluxes then evaluated at p^L, as in the WRR scheme)
        # instead of failing the step. Typical on dry-soil fronts (Szymkiewicz).
        self.FCT_PREDICTOR_FALLBACK = FCT_PREDICTOR_FALLBACK
        self.uL = uL
        self.uR = uR
        self.cK = cK
        # Strong Dirichlet: Dirichlet DOFs are set from the BC each residual and
        # masked out of the residual and the FCT correction (bc_mask = 0). With
        # the default False they are enforced weakly through the boundary flux
        # and remain free DOFs, so the FCT correction also moves them.
        self.forceStrongConditions = forceStrongConditions
        self.cE = cE
        self.outputQuantDOFs = outputQuantDOFs
        #For seepage anb
        self.model = None 

        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         variableNames,
                         sparseDiffusionTensors = sparseDiffusionTensors,
                         useSparseDiffusion = True)

    def attachModels(self, modelList):
        # NOTE: self.model is already set to this Richards LevelModel by
        # OneLevelTransport.__init__ (`self.coefficients.model = self`).
        # Do NOT overwrite it from self.modelIndex — that hardcoded index (=1)
        # points to TADR in the standard pnList, which corrupts Richards'
        # self.model and silently breaks density coupling.
        if self.density_model is None:
            return
        self.densityModel = modelList[self.density_model]

    def preStep(self, t, firstStep=False):
        # every step of the implicit FCT starts with the Galerkin predictor
        if hasattr(self, 'model') and hasattr(self.model, 'fct_phase'):
            self.model.fct_phase = 'predictor'
        # Refresh coupled density every step from the transport (TADR) model,
        # mirroring how TADR refreshes velocity / aliases moisture content.
        if self.density_model is None or not hasattr(self, 'densityModel'):
            return {}
        coeffs = getattr(self.densityModel, 'coefficients', None)
        if coeffs is None:
            return {}
        q_rho = getattr(coeffs, 'q_rho', None)
        ebqe_rho = getattr(coeffs, 'ebqe_rho', None)
        if q_rho is not None and hasattr(self.model, 'q') and 'rho' in self.model.q:
            self.model.q['rho'][:] = q_rho
        if ebqe_rho is not None and hasattr(self.model, 'ebqe') and 'rho' in self.model.ebqe:
            self.model.ebqe['rho'][:] = ebqe_rho

        # ---- coupling diagnostic: MPI-reduced, print on rank 0 ----
        # from mpi4py import MPI
        # comm = MPI.COMM_WORLD
        #
        # def _global_stats(local):
        #     a = np.asarray(local)
        #     lo = comm.allreduce(float(a.min()) if a.size else float('inf'), op=MPI.MIN)
        #     hi = comm.allreduce(float(a.max()) if a.size else float('-inf'), op=MPI.MAX)
        #     ssum = comm.allreduce(float(a.sum()), op=MPI.SUM)
        #     n = comm.allreduce(int(a.size), op=MPI.SUM)
        #     return lo, hi, (ssum / n if n > 0 else float('nan'))
        #
        # if q_rho is not None and 'rho' in self.model.q:
        #     src = np.asarray(q_rho)
        #     dst = np.asarray(self.model.q['rho'])
        #     local_diff = float(np.max(np.abs(src - dst))) if src.shape == dst.shape else float('nan')
        #     diff = comm.allreduce(local_diff, op=MPI.MAX)
        #     s_lo, s_hi, s_mn = _global_stats(src)
        #     d_lo, d_hi, d_mn = _global_stats(dst)
        #     if comm.Get_rank() == 0:
        #         logEvent(
        #             "[Coupling rho q] Richards.preStep t={:.6e} firstStep={} "
        #             "TADR.q_rho (min,max,mean)=({:.6e},{:.6e},{:.6e}) "
        #             "Richards.q['rho'] (min,max,mean)=({:.6e},{:.6e},{:.6e}) "
        #             "max|src-dst|={:.3e}".format(
        #                 float(t), firstStep, s_lo, s_hi, s_mn,
        #                 d_lo, d_hi, d_mn, diff),
        #             level=2)
        # if ebqe_rho is not None and 'rho' in self.model.ebqe:
        #     src_b = np.asarray(ebqe_rho)
        #     dst_b = np.asarray(self.model.ebqe['rho'])
        #     local_diff_b = float(np.max(np.abs(src_b - dst_b))) if src_b.shape == dst_b.shape else float('nan')
        #     diff_b = comm.allreduce(local_diff_b, op=MPI.MAX)
        #     s_lo, s_hi, s_mn = _global_stats(src_b)
        #     d_lo, d_hi, d_mn = _global_stats(dst_b)
        #     if comm.Get_rank() == 0:
        #         logEvent(
        #             "[Coupling rho ebqe] Richards.preStep t={:.6e} "
        #             "TADR.ebqe_rho (min,max,mean)=({:.6e},{:.6e},{:.6e}) "
        #             "Richards.ebqe['rho'] (min,max,mean)=({:.6e},{:.6e},{:.6e}) "
        #             "max|src-dst|={:.3e}".format(
        #                 float(t), s_lo, s_hi, s_mn, d_lo, d_hi, d_mn, diff_b),
        #             level=2)
        return {}


    def initializeMesh(self,mesh):
        from proteus.SubsurfaceTransportCoefficients import BlockHeterogeneousCoefficients
        self.elementMaterialTypes,self.exteriorElementBoundaryTypes,self.elementBoundaryTypes = BlockHeterogeneousCoefficients(mesh).initializeMaterialTypes()
        #want element boundary material types for evaluating heterogeneity
        #not boundary conditions
        self.isSeepageFace = np.zeros((mesh.nExteriorElementBoundaries_global),'i')
        if self.getSeepageFace != None:
            for ebNE in range(mesh.nExteriorElementBoundaries_global):
                #mwf missing ebNE-->ebN?
                ebN = mesh.exteriorElementBoundariesArray[ebNE]
                #print "eb flag",mesh.elementBoundaryMaterialTypes[ebN]
            
                #print self.getSeepageFace(mesh.elementBoundaryMaterialTypes[ebN])
                self.isSeepageFace[ebNE] = self.getSeepageFace(mesh.elementBoundaryMaterialTypes[ebN])
        #print (self.isSeepageFace)
    def initializeElementQuadrature(self,t,cq):
        self.materialTypes_q = self.elementMaterialTypes
        self.q_shape = cq[('u',0)].shape
        #self.anb_seepage_flux= anb_seepage_flux
        #print("The seepage is ", anb_seepage_flux)
#        cq['Ks'] = np.zeros(self.q_shape,'d')
#        for k in range(self.q_shape[1]):
#            cq['Ks'][:,k] = self.Ksw_types[self.elementMaterialTypes,0]
        self.q[('vol_frac',0)] = np.zeros(self.q_shape,'d')
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        self.materialTypes_ebq = np.zeros(cebq[('u',0)].shape[0:2],'i')
        self.ebq_shape = cebq[('u',0)].shape
        for ebN_local in range(self.ebq_shape[1]):
            self.materialTypes_ebq[:,ebN_local] = self.elementMaterialTypes
        self.ebq[('vol_frac',0)] = np.zeros(self.ebq_shape,'d')

    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        self.materialTypes_ebqe = self.exteriorElementBoundaryTypes
        self.ebqe_shape = cebqe[('u',0)].shape
        self.ebqe[('vol_frac',0)] = np.zeros(self.ebqe_shape,'d')
        #
    

    def evaluate(self,t,c):
        if c[('u',0)].shape == self.q_shape:
            materialTypes = self.materialTypes_q
            vol_frac = self.q[('vol_frac',0)]
        elif c[('u',0)].shape == self.ebqe_shape:
            materialTypes = self.materialTypes_ebqe
            vol_frac = self.ebqe[('vol_frac',0)]
        elif c[('u',0)].shape == self.ebq_shape:
            materialTypes = self.materialTypes_ebq
            vol_frac = self.ebq[('vol_frac',0)]
        else:
            assert False, "no materialType found to match c[('u',0)].shape= %s " % c[('u',0)].shape
        self.conservativeHeadRichardsMualemVanGenuchten_sd_het(self.sdInfo[(0,0)][0],
                                                               self.sdInfo[(0,0)][1],
                                                               materialTypes,
                                                               self.rho,
                                                               self.beta,
                                                               self.gravity,
                                                               self.vgm_alpha_types,
                                                               self.vgm_n_types,
                                                               self.thetaR_types,
                                                               self.thetaSR_types,
                                                               self.Ksw_types,
                                                               c[('u',0)],
                                                               c[('m',0)],
                                                               c[('dm',0,0)],
                                                               c[('f',0)],
                                                               c[('df',0,0)],
                                                               c[('a',0,0)],
                                                               c[('da',0,0,0)],
                                                               vol_frac)
         # Log grad(u) for debugging
        # if ('grad(u)', 0) in c:
        #     logEvent(f"Richards grad(u): mean={c[('grad(u)', 0)].mean()}, min={c[('grad(u)', 0)].min()}, max={c[('grad(u)', 0)].max()}")
        # else:
        #     logEvent("Warning: grad(u) is not available in Richards coefficients.")
        
        # Add logging for grad(u)
        # print "Picard---------------------------------------------------------------"
        # c[('df',0,0)][:] = 0.0
        # c[('da',0,0,0)][:] = 0.0
#         self.conservativeHeadRichardsMualemVanGenuchtenHetEvaluateV2(materialTypes,
#                                                                      self.rho,
#                                                                      self.beta,
#                                                                      self.gravity,
#                                                                      self.vgm_alpha_types,
#                                                                      self.vgm_n_types,
#                                                                      self.thetaR_types,
#                                                                      self.thetaSR_types,
#                                                                      self.Ksw_types,
#                                                                      c[('u',0)],
#                                                                      c[('m',0)],
#                                                                      c[('dm',0,0)],
#                                                                      c[('f',0)],
#                                                                      c[('df',0,0)],
#                                                                      c[('a',0,0)],
#                                                                      c[('da',0,0,0)])
        #mwf debug
        if (np.isnan(c[('da',0,0,0)]).any() or
            np.isnan(c[('a',0,0)]).any() or
            np.isnan(c[('df',0,0)]).any() or
            np.isnan(c[('f',0)]).any() or
            np.isnan(c[('u',0)]).any() or
            np.isnan(c[('m',0)]).any() or
            np.isnan(c[('dm',0,0)]).any()):
            import pdb
            pdb.set_trace()
    
    # def postStep(self, t, firstStep=False):
    #     if not self.outputQuantDOFs:
    #         return {}
    #     if (self.model is None or
    #             ('velocity_couple', 0) not in self.model.q or
    #             ('grad(u_v)', 0) not in self.model.q):
    #         return {}

    #     mpicomm = self._get_mpi_comm()
    #     rank = mpicomm.Get_rank()
    #     nSpace = int(getattr(self.model, 'nSpace_global',
    #                          getattr(self.model, 'nSpace', self.nd)))
    #     n_owned = self._get_owned_element_count()
    #     stab_tag = f"stab{self.STABILIZATION_TYPE}"

    #     qcoords_local = self._get_q_coordinates().reshape((-1, 3))
    #     qv_local = np.asarray(self.model.q[('velocity_couple', 0)][:n_owned]).reshape((-1, nSpace))
    #     qgrad_local = np.asarray(self.model.q[('grad(u_v)', 0)][:n_owned]).reshape((-1, nSpace))

    #     if not hasattr(self, '_wrote_q_coords_once'):
    #         qcoords_all = mpicomm.gather(qcoords_local, root=0)
    #         if rank == 0:
    #             qcoords = np.vstack(qcoords_all)
    #             np.savetxt(f"richards_q_coordinates_{stab_tag}.txt",
    #                        qcoords,
    #                        fmt="%.16e",
    #                        header=f"columns: x y z | total_rows={qcoords.shape[0]}")
    #             logEvent(f"[Richards postStep] wrote richards_q_coordinates_{stab_tag}.txt rows={qcoords.shape[0]}")
    #         self._wrote_q_coords_once = True

    #     q_profile_local = np.hstack((qcoords_local, qv_local))
    #     q_profile_all = mpicomm.gather(q_profile_local, root=0)
    #     q_grad_profile_local = np.hstack((qcoords_local, qgrad_local))
    #     q_grad_profile_all = mpicomm.gather(q_grad_profile_local, root=0)
    #     velocity_magnitude_local = np.linalg.norm(qv_local, axis=1) if qv_local.size else np.zeros((0,), 'd')
    #     vmax_local = float(velocity_magnitude_local.max()) if velocity_magnitude_local.size else 0.0
    #     vmax = Comm.get().globalMax(vmax_local)
    #     grad_magnitude_local = np.linalg.norm(qgrad_local, axis=1) if qgrad_local.size else np.zeros((0,), 'd')
    #     gmax_local = float(grad_magnitude_local.max()) if grad_magnitude_local.size else 0.0
    #     gmax = Comm.get().globalMax(gmax_local)

    #     if rank == 0:
    #         q_profile = np.vstack(q_profile_all)
    #         q_grad_profile = np.vstack(q_grad_profile_all)
    #         header_cols = "x y z vx vy" if nSpace == 2 else "x y z vx vy vz"
    #         header_grad_cols = "x y z gx gy" if nSpace == 2 else "x y z gx gy gz"
    #         np.savetxt(f"richards_q_velocity_profile_{stab_tag}_t{t:.8e}.txt",
    #                    q_profile,
    #                    fmt="%.16e",
    #                    header=f"columns: {header_cols} | t={t:.16e} | total_rows={q_profile.shape[0]}")
    #         logEvent(f"[Richards postStep] wrote richards_q_velocity_profile_{stab_tag}_t{t:.8e}.txt vmax={vmax:.6e}")
    #         np.savetxt(f"richards_q_grad_u_profile_{stab_tag}_t{t:.8e}.txt",
    #                    q_grad_profile,
    #                    fmt="%.16e",
    #                    header=f"columns: {header_grad_cols} | t={t:.16e} | total_rows={q_grad_profile.shape[0]}")
    #         logEvent(f"[Richards postStep] wrote richards_q_grad_u_profile_{stab_tag}_t{t:.8e}.txt gmax={gmax:.6e}")
    #     return {}





    
    # #def postStep(self, t, firstStep=False):
    #    import os
    #    #from proteus import Comm
    #    comm = Comm.get()
    #    if comm.isMaster():
    #        try:
    #            # Attempt to access and sum the seepage flux
    #            s_now = float(np.sum(self.model.anb_seepage_flux_n))
    #            #s_now= float(self.model.anb_seepage_flux)
    #            if s_now>0.0:
    #                with open("seepage_flux.txt", "a") as f:
    #                    if os.stat("seepage_flux.txt").st_size == 0:
    #                        f.write("time,seepage_flux\n")
    #                        f.write(f"{t:.6f},{s_now:.6f}\n")
    #       except Exception as e:
    #            logEvent(f"[postStep] Skipped logging seepage: {e}")
        
   
        
class LevelModel(proteus.Transport.OneLevelTransport):
    nCalls=0
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
                 sd = True,
                 movingDomain=False,
                 bdyNullSpace=False):
        self.bdyNullSpace=bdyNullSpace
        #
        #set the objects describing the method and boundary conditions
        #
        self.movingDomain=movingDomain
        self.tLast_mesh=None
        #
        self.name=name
        self.sd=sd
        self.Hess=False
        self.lowmem=True
        self.timeTerm=True#allow turning off  the  time derivative
        #self.lowmem=False
        self.testIsTrial=True
        self.phiTrialIsTrial=True
        self.u = uDict
        self.ua = {}#analytical solutions
        self.phi  = phiDict
        self.dphi={}
        self.matType = matType
        #mwf try to reuse test and trial information across components if spaces are the same
        self.reuse_test_trial_quadrature = reuse_trial_and_test_quadrature#True#False
        if self.reuse_test_trial_quadrature:
            for ci in range(1,coefficients.nc):
                assert self.u[ci].femSpace.__class__.__name__ == self.u[0].femSpace.__class__.__name__, "to reuse_test_trial_quad all femSpaces must be the same!"
        self.u_dof_old = None
        ## Simplicial Mesh
        self.mesh = self.u[0].femSpace.mesh #assume the same mesh for  all components for now
        self.testSpace = testSpaceDict
        self.dirichletConditions = dofBoundaryConditionsDict
        self.dirichletNodeSetList=None #explicit Dirichlet  conditions for now, no Dirichlet BC constraints
        self.coefficients = coefficients
        self.coefficients.initializeMesh(self.mesh)
        self.nc = self.coefficients.nc
        self.stabilization = stabilization
        self.shockCapturing = shockCapturing
        self.conservativeFlux = conservativeFluxDict #no velocity post-processing for now
        self.fluxBoundaryConditions=fluxBoundaryConditionsDict
        self.advectiveFluxBoundaryConditionsSetterDict=advectiveFluxBoundaryConditionsSetterDict
        self.diffusiveFluxBoundaryConditionsSetterDictDict = diffusiveFluxBoundaryConditionsSetterDictDict
        #determine whether  the stabilization term is nonlinear
        self.stabilizationIsNonlinear = False
        #anb add 
        self.anb_seepage_flux= 0.0
        self.coefficients.model = self 

        #cek come back
        if self.stabilization != None:
            for ci in range(self.nc):
                if ci in coefficients.mass:
                    for flag in list(coefficients.mass[ci].values()):
                        if flag == 'nonlinear':
                            self.stabilizationIsNonlinear=True
                if  ci in coefficients.advection:
                    for  flag  in list(coefficients.advection[ci].values()):
                        if flag == 'nonlinear':
                            self.stabilizationIsNonlinear=True
                if  ci in coefficients.diffusion:
                    for diffusionDict in list(coefficients.diffusion[ci].values()):
                        for  flag  in list(diffusionDict.values()):
                            if flag != 'constant':
                                self.stabilizationIsNonlinear=True
                if  ci in coefficients.potential:
                    for flag in list(coefficients.potential[ci].values()):
                        if  flag == 'nonlinear':
                            self.stabilizationIsNonlinear=True
                if ci in coefficients.reaction:
                    for flag in list(coefficients.reaction[ci].values()):
                        if  flag == 'nonlinear':
                            self.stabilizationIsNonlinear=True
                if ci in coefficients.hamiltonian:
                    for flag in list(coefficients.hamiltonian[ci].values()):
                        if  flag == 'nonlinear':
                            self.stabilizationIsNonlinear=True
        #determine if we need element boundary storage
        self.elementBoundaryIntegrals = {}
        for ci  in range(self.nc):
            self.elementBoundaryIntegrals[ci] = ((self.conservativeFlux != None) or 
                                                 (numericalFluxType != None) or
                                                 (self.fluxBoundaryConditions[ci] == 'outFlow') or
                                                 (self.fluxBoundaryConditions[ci] == 'mixedFlow') or
                                                 (self.fluxBoundaryConditions[ci] == 'setFlow'))
        #
        #calculate some dimensions
        #
        self.nSpace_global    = self.u[0].femSpace.nSpace_global #assume same space dim for all variables
        self.nDOF_trial_element     = [u_j.femSpace.max_nDOF_element for  u_j in list(self.u.values())]
        self.nDOF_phi_trial_element     = [phi_k.femSpace.max_nDOF_element for  phi_k in list(self.phi.values())]
        self.n_phi_ip_element = [phi_k.femSpace.referenceFiniteElement.interpolationConditions.nQuadraturePoints for  phi_k in list(self.phi.values())]
        self.nDOF_test_element     = [femSpace.max_nDOF_element for femSpace in list(self.testSpace.values())]
        self.nFreeDOF_global  = [dc.nFreeDOF_global for dc in list(self.dirichletConditions.values())]
        self.nVDOF_element    = sum(self.nDOF_trial_element)
        self.nFreeVDOF_global = sum(self.nFreeDOF_global)
        #
        NonlinearEquation.__init__(self,self.nFreeVDOF_global)
        #
        #build the quadrature point dictionaries from the input (this
        #is just for convenience so that the input doesn't have to be
        #complete)
        #
        elementQuadratureDict={}
        elemQuadIsDict = isinstance(elementQuadrature,dict)
        if elemQuadIsDict: #set terms manually
            for I in self.coefficients.elementIntegralKeys:
                if I in elementQuadrature:
                    elementQuadratureDict[I] = elementQuadrature[I]
                    
                else:
                    elementQuadratureDict[I] = elementQuadrature['default']
        else:
            for I in self.coefficients.elementIntegralKeys:
                elementQuadratureDict[I] = elementQuadrature
        if self.stabilization != None:
            for I in self.coefficients.elementIntegralKeys:
                if elemQuadIsDict:
                    if I in elementQuadrature:
                        elementQuadratureDict[('stab',)+I[1:]] = elementQuadrature[I]
                    else:
                        elementQuadratureDict[('stab',)+I[1:]] = elementQuadrature['default']
                else:
                    elementQuadratureDict[('stab',)+I[1:]] = elementQuadrature
        if self.shockCapturing != None:
            for ci in self.shockCapturing.components:
                if elemQuadIsDict:
                    if ('numDiff',ci,ci) in elementQuadrature:
                        elementQuadratureDict[('numDiff',ci,ci)] = elementQuadrature[('numDiff',ci,ci)]

                    else:
                        elementQuadratureDict[('numDiff',ci,ci)] = elementQuadrature['default']
                else:
                    elementQuadratureDict[('numDiff',ci,ci)] = elementQuadrature
        if massLumping:
            for ci in list(self.coefficients.mass.keys()):
                elementQuadratureDict[('m',ci)] = Quadrature.SimplexLobattoQuadrature(self.nSpace_global,1)
            for I in self.coefficients.elementIntegralKeys:
                elementQuadratureDict[('stab',)+I[1:]] = Quadrature.SimplexLobattoQuadrature(self.nSpace_global,1)
        if reactionLumping:
            for ci in list(self.coefficients.mass.keys()):
                elementQuadratureDict[('r',ci)] = Quadrature.SimplexLobattoQuadrature(self.nSpace_global,1)
            for I in self.coefficients.elementIntegralKeys:
                elementQuadratureDict[('stab',)+I[1:]] = Quadrature.SimplexLobattoQuadrature(self.nSpace_global,1)
        elementBoundaryQuadratureDict={}
        if isinstance(elementBoundaryQuadrature,dict): #set terms manually
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
        #mwf include tag telling me which indices are which quadrature rule?
        (self.elementQuadraturePoints,self.elementQuadratureWeights,
         self.elementQuadratureRuleIndeces) = Quadrature.buildUnion(elementQuadratureDict)
        self.nQuadraturePoints_element = self.elementQuadraturePoints.shape[0]
        self.nQuadraturePoints_global = self.nQuadraturePoints_element*self.mesh.nElements_global
        #
        #Repeat the same thing for the element boundary quadrature
        #
        (self.elementBoundaryQuadraturePoints,
         self.elementBoundaryQuadratureWeights,
         self.elementBoundaryQuadratureRuleIndeces) = Quadrature.buildUnion(elementBoundaryQuadratureDict)
        self.nElementBoundaryQuadraturePoints_elementBoundary = self.elementBoundaryQuadraturePoints.shape[0]
        self.nElementBoundaryQuadraturePoints_global = (self.mesh.nElements_global*
                                                        self.mesh.nElementBoundaries_element*
                                                        self.nElementBoundaryQuadraturePoints_elementBoundary)

        #
        #storage dictionaries
        self.scalars_element = set()
        #
        #simplified allocations for test==trial and also check if space is mixed or not
        #
        self.q={}
        self.ebq={}
        self.ebq_global={}
        self.ebqe={}
        self.phi_ip={}
        self.edge_based_cfl = np.zeros(self.u[0].dof.shape)+100
        #mesh
        self.q['x'] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,3),'d')
        self.q[('dV_u', 0)] = (1.0/ self.mesh.nElements_global) * np.ones((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.ebqe['x'] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary,3),'d')
        self.q[('u',0)] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element),'d')
        self.q[('grad(u)',0)] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,self.nSpace_global),'d')
        self.q[('grad(phi)',0)] = self.q[('u',0)]
        self.q[('dphi',0,0)] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,),'d')
        self.q[('da',0,0,0)] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,),'d')
        self.q[('grad(u_v)',0)] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,self.nSpace_global),'d')
        self.q[('velocity',0)] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,self.nSpace_global),'d')
        self.q[('velocity_couple',0)] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,self.nSpace_global),'d')      
        self.q[('m',0)] = self.q[('u',0)].copy()
        self.q[('theta',0)] = self.q[('u',0)].copy()
        self.q[('mt',0)] = self.q[('u',0)].copy()
        self.q[('m_last',0)] = self.q[('u',0)].copy()
        self.q[('m_tmp',0)] = self.q[('u',0)].copy()
        self.q[('cfl',0)] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element),'d')
        self.q[('numDiff',0,0)] =  np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element),'d')
        self.numDiff_star = self.q[('numDiff',0,0)]
        self.q[('numDiff_last',0,0)] =  np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element),'d')
        self.ebqe[('u',0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
        self.ebqe[('theta',0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
        self.ebqe[('grad(u)',0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary,self.nSpace_global),'d')
        self.ebqe[('velocity',0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary,self.nSpace_global),'d')
        self.ebqe[('velocity_couple',0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary,self.nSpace_global),'d')       
        self.ebqe[('advectiveFlux_bc_flag',0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'i')
        self.ebqe[('advectiveFlux_bc',0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
        self.ebqe[('advectiveFlux',0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
        self.ebqe[('penalty')] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
        
        self.q['rho'] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element),'d')
        self.ebqe['rho'] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
        self.q['rho'][:] = self.coefficients.rho
        self.ebqe['rho'][:] = self.coefficients.rho
        
        
        self.points_elementBoundaryQuadrature= set()
        self.scalars_elementBoundaryQuadrature= set([('u',ci) for ci in range(self.nc)])
        self.vectors_elementBoundaryQuadrature= set()
        self.tensors_elementBoundaryQuadrature= set()
        self.inflowBoundaryBC = {}
        self.inflowBoundaryBC_values = {}
        self.inflowFlux = {}
        for cj in range(self.nc):
            self.inflowBoundaryBC[cj] = np.zeros((self.mesh.nExteriorElementBoundaries_global,),'i')
            self.inflowBoundaryBC_values[cj] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nDOF_trial_element[cj]),'d')
            self.inflowFlux[cj] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
        self.internalNodes = set(range(self.mesh.nNodes_global))
        #identify the internal nodes this is ought to be in mesh
        ##\todo move this to mesh
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
        #
        del self.internalNodes
        self.internalNodes = None
        logEvent("Updating local to global mappings",2)
        self.updateLocal2Global()
        logEvent("Building time integration object",2)
        logEvent(memory("inflowBC, internalNodes,updateLocal2Global","OneLevelTransport"),level=4)
        #mwf for interpolating subgrid error for gradients etc
        if self.stabilization and self.stabilization.usesGradientStabilization:
            self.timeIntegration = TimeIntegrationClass(self,integrateInterpolationPoints=True)
        else:
             self.timeIntegration = TimeIntegrationClass(self)

        if options != None:
            self.timeIntegration.setFromOptions(options)
        logEvent(memory("TimeIntegration","OneLevelTransport"),level=4)
        logEvent("Calculating numerical quadrature formulas",2)
        self.calculateQuadrature()
        #lay out components/equations contiguously for now
        self.offset = [0]
        for ci in range(1,self.nc):
            self.offset += [self.offset[ci-1]+self.nFreeDOF_global[ci-1]]
        self.stride = [1 for ci in range(self.nc)]
        #use contiguous layout of components for parallel, requires weak DBC's
        # mql. Some ASSERTS to restrict the combination of the methods
        if self.coefficients.STABILIZATION_TYPE > 0:
            pass
            #assert self.timeIntegration.isSSP == True, "If STABILIZATION_TYPE>0, use RKEV timeIntegration within VOF model"
            #cond = 'levelNonlinearSolver' in dir(options) and (options.levelNonlinearSolver ==
            #                                                   ExplicitLumpedMassMatrixForRichards or options.levelNonlinearSolver == ExplicitConsistentMassMatrixForRichards)
            #assert cond, "If STABILIZATION_TYPE>0, use levelNonlinearSolver=ExplicitLumpedMassMatrixForRichards or ExplicitConsistentMassMatrixForRichards"
        try:
            if 'levelNonlinearSolver' in dir(options) and options.levelNonlinearSolver == ExplicitLumpedMassMatrixForRichards:
                assert self.coefficients.LUMPED_MASS_MATRIX, "If levelNonlinearSolver=ExplicitLumpedMassMatrix, use LUMPED_MASS_MATRIX=True"
        except:
            pass
        if self.coefficients.LUMPED_MASS_MATRIX == True:
            cond = self.coefficients.STABILIZATION_TYPE == 2
            assert cond, "Use lumped mass matrix just with: STABILIZATION_TYPE=2 (smoothness based stab.)"
            cond = 'levelNonlinearSolver' in dir(options) and options.levelNonlinearSolver == ExplicitLumpedMassMatrixForRichards
            assert cond, "Use levelNonlinearSolver=ExplicitLumpedMassMatrixForRichards when the mass matrix is lumped"
        
        #if not self.coefficients.LUMPED_MASS_MATRIX and self.coefficients.STABILIZATION_TYPE == 2:
        #    cond = 'levelNonlinearSolver' in dir(options) and options.levelNonlinearSolver == Newton
        
        #if self.coefficients.FCT == True:
        #    cond = self.coefficients.STABILIZATION_TYPE = 3, "Use FCT just with STABILIZATION_TYPE=3; i.e., edge based stabilization"
        
        if self.coefficients.FCT:
            valid_stabilization_types = {1, 2}  # Only allow FCT for STABILIZATION_TYPE 1 (EV_Stab) and 2 (EntropyViscosity)
            if self.coefficients.STABILIZATION_TYPE not in valid_stabilization_types:
                raise ValueError("Use FCT only with STABILIZATION_TYPE 1 (EV_Stab) or 2 (EntropyViscosity).")        
        if self.coefficients.FCT == True:
            cond = self.coefficients.STABILIZATION_TYPE > 0, "Use FCT just with STABILIZATION_TYPE>0; i.e., edge based stabilization"
        # # END OF ASSERTS

        # cek adding empty data member for low order numerical viscosity structures here for now
        self.ML = None  # lumped mass matrix
        self.MC_global = None  # consistent mass matrix
        self.cterm_global = None
        self.cterm_transpose_global = None
        # dL_global and dC_global are not the full matrices but just the CSR arrays containing the non zero entries
        self.residualComputed=False #TMP
        self.dLow= None
        self.fluxMatrix = None
        self.mDotLow = None
        self.mLow = None
        self.dt_times_dC_minus_dL = None
        self.min_m_bc = None
        self.max_m_bc = None
        # Zalesak's ratios and the assembled antidiffusive flux matrix.  Owned here
        # rather than inside the kernel so they survive between FCTStep's two passes
        # and can be ghost-scattered in between; see FCTStep.
        self.Rpos = None
        self.Rneg = None
        self.FluxCorrectionMatrix = None
        # FCT_TYPE='implicit': which of the three solves of a time step the residual is for
        # ('predictor' Galerkin, 'flux' evaluation at p^H, 'final' low order); see ImplicitFCTStep.
        self.fct_phase = 'predictor'
        self.fct_resolve = False
        self.fct_fallback_count = 0
        self.lscheme_active = False  # set by Newton in NONLINEAR_SOLVER='hybrid'
        self.alpha_nv = None
        self.m_fct = None
        # Aux quantity at DOFs to be filled by optimized code (MQL)
        self.quantDOFs = np.zeros(self.u[0].dof.shape, 'd')
        self.mLow = np.zeros(self.u[0].dof.shape, 'd')
        self.mHigh = np.zeros(self.u[0].dof.shape, 'd')
        self.mDotLow = np.zeros(self.u[0].dof.shape, 'd')
        self.fluxCorrection = np.zeros(self.u[0].dof.shape, 'd')
        self.mn = np.zeros(self.u[0].dof.shape, 'd')
        self.anb_seepage_flux_n = np.zeros(self.u[0].dof.shape, 'd')
        #per-node ROCK region, from elementMaterialTypes (the .ele region
        #attribute) via the first element containing each node.  NOT
        #mesh.nodeMaterialTypes: that array is the .node boundary-marker column
        #(0 for every interior node, segment tags on the boundary), so indexing
        #alpha/n/thetaR/thetaSR/KWs with it gives the interior the unset slot 0
        #and reads past the end of those arrays wherever a boundary tag exceeds
        #nMediaTypes.  A node on a soil interface gets the first incident
        #element's rock; the nodal scheme allows it only one theta(psi).
        self.nodeMaterialTypes_n = np.zeros((self.mesh.nNodes_global,), 'i')
        nodeMaterialTypesSet = np.zeros((self.mesh.nNodes_global,), 'i')
        for eN in range(self.mesh.nElements_global):
            for i_local in range(self.mesh.nNodes_element):
                gN = self.mesh.elementNodesArray[eN, i_local]
                if not nodeMaterialTypesSet[gN]:
                    self.nodeMaterialTypes_n[gN] = self.mesh.elementMaterialTypes[eN]
                    nodeMaterialTypesSet[gN] = 1
        self.freeDOFMaterialTypes = np.zeros((self.nFreeDOF_global[0],), 'i')
        free_l2g = np.asarray(self.l2g[0]['freeGlobal']).ravel()
        dof_l2g = np.asarray(self.u[0].femSpace.dofMap.l2g).ravel()
        for free_dof, global_dof in zip(free_l2g, dof_l2g):
            if 0 <= free_dof < self.freeDOFMaterialTypes.shape[0]:
                self.freeDOFMaterialTypes[free_dof] = self.nodeMaterialTypes_n[global_dof]
        comm = Comm.get()
        self.comm=comm
        if comm.size() > 1:
            assert numericalFluxType != None and numericalFluxType.useWeakDirichletConditions,"You must use a numerical flux to apply weak boundary conditions for parallel runs"
            self.offset = [0]
            for ci in range(1,self.nc):
                self.offset += [ci]
            self.stride = [self.nc for ci in range(self.nc)]
        #
        logEvent(memory("stride+offset","OneLevelTransport"),level=4)
        
        if numericalFluxType != None:
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
        #per-model weak Dirichlet penalty, set from the Coefficients
        #(penalty_constant=...), so a problem can stiffen or relax the IIPG
        #boundary term without editing the shared NumericalFlux class default.
        #Has to run before the penalty arrays below are filled.
        if self.numericalFlux is not None:
            if getattr(self.coefficients,'penalty_constant',None) is not None:
                self.numericalFlux.penalty_constant = float(self.coefficients.penalty_constant)
            if getattr(self.coefficients,'penalty_power',None) is not None:
                self.numericalFlux.penalty_power = float(self.coefficients.penalty_power)
            logEvent("Richards weak Dirichlet penalty: constant=%e power=%e" % (self.numericalFlux.penalty_constant,
                                                                               self.numericalFlux.penalty_power),level=2)
        #set penalty terms
        #cek todo move into numerical flux initialization
        if 'penalty' in self.ebq_global:
            for ebN in range(self.mesh.nElementBoundaries_global):
                for k in range(self.nElementBoundaryQuadraturePoints_elementBoundary):
                    self.ebq_global['penalty'][ebN,k] = self.numericalFlux.penalty_constant/(self.mesh.elementBoundaryDiametersArray[ebN]**self.numericalFlux.penalty_power)
        #penalty term
        #cek move  to Numerical flux initialization
        if 'penalty' in self.ebqe:
            for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
                ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
                for k in range(self.nElementBoundaryQuadraturePoints_elementBoundary):
                    self.ebqe['penalty'][ebNE,k] = self.numericalFlux.penalty_constant/self.mesh.elementBoundaryDiametersArray[ebN]**self.numericalFlux.penalty_power
        logEvent(memory("numericalFlux","OneLevelTransport"),level=4)
        self.elementEffectiveDiametersArray  = self.mesh.elementInnerDiametersArray
        #use post processing tools to get conservative fluxes, None by default
        from proteus import PostProcessingTools
        self.velocityPostProcessor = PostProcessingTools.VelocityPostProcessingChooser(self)  
        logEvent(memory("velocity postprocessor","OneLevelTransport"),level=4)
        #helper for writing out data storage
        from proteus import Archiver
        self.elementQuadratureDictionaryWriter = Archiver.XdmfWriter()
        self.elementBoundaryQuadratureDictionaryWriter = Archiver.XdmfWriter()
        self.exteriorElementBoundaryQuadratureDictionaryWriter = Archiver.XdmfWriter()
        #TODO get rid of this
        for ci,fbcObject  in list(self.fluxBoundaryConditionsObjectsDict.items()):
            self.ebqe[('advectiveFlux_bc_flag',ci)] = np.zeros(self.ebqe[('advectiveFlux_bc',ci)].shape,'i')
            for t,g in list(fbcObject.advectiveFluxBoundaryConditionsDict.items()):
                if ci in self.coefficients.advection:
                    self.ebqe[('advectiveFlux_bc',ci)][t[0],t[1]] = g(self.ebqe[('x')][t[0],t[1]],self.timeIntegration.t)
                    self.ebqe[('advectiveFlux_bc_flag',ci)][t[0],t[1]] = 1

        if hasattr(self.numericalFlux,'setDirichletValues'):
            self.numericalFlux.setDirichletValues(self.ebqe)
        if not hasattr(self.numericalFlux,'isDOFBoundary'):
            self.numericalFlux.isDOFBoundary = {0:np.zeros(self.ebqe[('u',0)].shape,'i')}
        if not hasattr(self.numericalFlux,'ebqe'):
            self.numericalFlux.ebqe = {('u',0):np.zeros(self.ebqe[('u',0)].shape,'d')}
        #TODO how to handle redistancing calls for calculateCoefficients,calculateElementResidual etc
        self.globalResidualDummy = None
        compKernelFlag=0
        self.delta_x_ij=None
        self.richards = cRichards_base(self.nSpace_global,
                             self.nQuadraturePoints_element,
                             self.u[0].femSpace.elementMaps.localFunctionSpace.dim,
                             self.u[0].femSpace.referenceFiniteElement.localFunctionSpace.dim,
                             self.testSpace[0].referenceFiniteElement.localFunctionSpace.dim,
                             self.nElementBoundaryQuadraturePoints_elementBoundary,
                             compKernelFlag)
        if self.movingDomain:
            self.MOVING_DOMAIN=1.0
        else:
            self.MOVING_DOMAIN=0.0
        #cek hack
        self.movingDomain=False
        self.MOVING_DOMAIN=0.0
        if self.mesh.nodeVelocityArray is None:
            self.mesh.nodeVelocityArray = np.zeros(self.mesh.nodeArray.shape,'d')
        self.dirichletConditionsForceDOF = {}
        if self.coefficients.forceStrongConditions:
            for cj in range(self.nc):
                self.dirichletConditionsForceDOF[cj] = DOFBoundaryConditions(self.u[cj].femSpace,dofBoundaryConditionsSetterDict[cj],weakDirichletConditions=False)
   
    def checkFCTParallelLayout(self):
        """Refuse a partition on which an OWNED DOF misses part of its element star.

        The FCT limiters (both FCT_TYPEs) are nonlinear in the assembled rows (ML, MC, the
        flux matrices, P_i, Q_i), so a row must be complete at every owned DOF. An additive
        residual does not need this, which is why Galerkin and the low order still agree with
        serial on such partitions. Element partitioning with nLayersOfOverlapForParallel = 0
        violates it: a node on the cut is owned by one rank but sees only that rank's elements
        (Hornung-Messing h = 1/16, 2 ranks: WRR FCT 2.8e-3 vs 6.8e-5 serial). Element
        partitioning with overlap >= 1 and node partitioning both pass.

        Test: ownedElementStarsIncomplete().
        """
        bad = self.ownedElementStarsIncomplete()
        if bad > 0:
            raise ValueError("Richards FCT in parallel: %d owned DOFs miss part of their element star, so the "
                             "limiter would act on partial rows. Use parallelPartitioningType = "
                             "MeshParallelPartitioningTypes.node, or element partitioning with "
                             "nLayersOfOverlapForParallel >= 1." % bad)

    def ownedElementStarsIncomplete(self):
        """Global number of owned DOFs whose element star is not entirely on their rank (0 in serial).

        The number of local elements containing each owned DOF must equal its global count,
        which is assembled by a reverse-add of the counts over the uniquely owned elements
        (eN < nElements_owned). Rows of element-assembled matrices (ML, MC, ...) are complete
        at owned DOFs exactly when this is 0.
        """
        par = getattr(self.u[0], 'par_dof', None)
        if par is None or self.comm.size() == 1:
            return 0
        l2g = np.asarray(self.l2g[0]['freeGlobal'])
        ndof = self.u[0].dof.shape[0]
        n_local = np.bincount(l2g.ravel(), minlength=ndof).astype('d')[:ndof]
        n_owned_elements = getattr(self.mesh, 'nElements_owned', self.mesh.nElements_global)
        saved = self.u[0].dof.copy()
        self.u[0].dof[:] = np.bincount(l2g[:n_owned_elements].ravel(), minlength=ndof).astype('d')[:ndof]
        par.scatter_reverse_add()
        n_global = self.u[0].dof.copy()
        self.u[0].dof[:] = saved
        n_owned = par.dim_proc
        bad_local = int(np.count_nonzero(n_local[:n_owned] != n_global[:n_owned]))
        return int(self.comm.globalSum(float(bad_local)))

    def femL2Norm(self, x, lumped=False):
        """Finite element L2 norm of a nodal vector x (free DOFs, local numbering with ghosts).

        ||x||_{L2} = sqrt(sum_{i owned} x_i (M_C x)_i), M_C the consistent mass matrix, i.e. the
        L2(Omega) norm of the finite element function with these nodal values (the quantity
        MFEM's GridFunction::ComputeL2Error(0) evaluates). lumped=True uses M_L instead.
        x must hold current ghost values. The consistent version needs complete rows of M_C
        at owned DOFs (node partitioning, or element partitioning with overlap >= 1); it
        raises otherwise. Used by NonlinearSolvers for unorm = 'L2' / 'L2_lumped'.
        """
        if self.MC_global is None:
            raise RuntimeError("femL2Norm: the mass matrix is assembled with the first residual")
        n_owned = getattr(getattr(self.u[0], 'par_dof', None), 'dim_proc', self.ML.shape[0])
        if lumped:
            local = float(np.dot(self.ML[:n_owned], x[:n_owned] * x[:n_owned]))
        else:
            if getattr(self, '_femL2_checked', False) is False:
                bad = self.ownedElementStarsIncomplete()
                if bad > 0:
                    raise ValueError("femL2Norm: %d owned DOFs miss part of their element star, so M_C x is "
                                     "incomplete there. Use node partitioning, element partitioning with "
                                     "overlap >= 1, or unorm = 'L2_lumped'." % bad)
                self._femL2_checked = True
            Mx = getattr(self, '_femL2_Mx', None)
            if Mx is None or Mx.shape != x.shape:
                Mx = self._femL2_Mx = np.zeros_like(x)
            self.MC_global.matvec(x, Mx)
            local = float(np.dot(x[:n_owned], Mx[:n_owned]))
        return np.sqrt(max(self.comm.globalSum(local), 0.0))

    def ghostScatter(self, *arrays):
        """Forward-insert (owner -> ghost copy) each free-DOF array.  No-op in serial.

        With one layer of overlap only the OWNED DOFs have their full element star
        and their full set of boundary faces on this rank, so anything built by a
        loop -- mDotLow, min/max_m_bc, the Zalesak ratios -- is
        incomplete at a ghost DOF.  The edge loops read those arrays at the COLUMN
        j of an owned row i, and j can be a ghost, so without this the two ranks
        sharing a cut edge build different f_ij, the antisymmetry f_ij = -f_ji is
        lost and the correction manufactures mass along the partition.  The owner
        always holds the complete value, so an insert scatter repairs every ghost.

        u[0].par_dof is the only ghosted vector this module owns, so the arrays are
        shuttled through u[0].dof (the same idiom as mphase_co2/m_comp_co2).
        Integer arrays round-trip exactly: they only ever carry small whole numbers
        (0/1 flags, material indices).
        """
        par = getattr(self.u[0], 'par_dof', None)
        if par is None:
            return
        ndof = self.u[0].dof.shape[0]
        saved = self.u[0].dof.copy()
        for a in arrays:
            if a is None or a.shape[0] != ndof:
                continue
            self.u[0].dof[:] = a
            par.scatter_forward_insert()
            a[:] = self.u[0].dof
        self.u[0].dof[:] = saved

    def lschemeOn(self):
        """True when the next residual should assemble the L-scheme matrix instead of the Jacobian."""
        c = self.coefficients
        return c.LSCHEME_L > 0.0 and (c.NONLINEAR_SOLVER == 'lscheme' or self.lscheme_active)

    def setLScheme(self, on):
        """Newton's switch for NONLINEAR_SOLVER='hybrid' (takes effect at the next residual)."""
        self.lscheme_active = bool(on)

    def nonlinearHybridOptions(self):
        """Options Newton reads for NONLINEAR_SOLVER='hybrid', else None."""
        c = self.coefficients
        if c.NONLINEAR_SOLVER != 'hybrid':
            return None
        return {'its': c.LSCHEME_RETRY_ITS, 'trigger': c.LSCHEME_TRIGGER, 'theta_max': c.LSCHEME_THETA_MAX}

    def defectCorrection(self):
        """True when the current Galerkin solve uses the L-scheme matrix (assembled by the edge kernel)."""
        return self.kernelStabilizationType() == 0 and self.lschemeOn()

    def kernelStabilizationType(self):
        """STABILIZATION_TYPE passed to the kernels. The implicit FCT changes it per phase:
        Galerkin for the predictor, the edge-based low order for the flux evaluation, and
        Implicit_FCT (the low order plus the limited correction) for the final solve."""
        if self.coefficients.FCT and self.coefficients.FCT_TYPE == 'implicit':
            return {'predictor': 0, 'low_predictor': 2, 'flux': 2, 'final': 3}[self.fct_phase]
        return self.coefficients.STABILIZATION_TYPE

    def fctPredictorFallback(self):
        """Called by Newton when the implicit FCT's Galerkin predictor failed. Switches this step's
        predictor to the low-order operator and returns True, or returns False (no fallback)."""
        c = self.coefficients
        if not (c.FCT and c.FCT_TYPE == 'implicit' and c.FCT_PREDICTOR_FALLBACK
                and self.fct_phase == 'predictor'):
            return False
        self.fct_phase = 'low_predictor'
        self.fct_fallback_count += 1
        logEvent("Implicit FCT: low-order predictor fallback #%d at t = %g"
                 % (self.fct_fallback_count, self.timeIntegration.t), level=1)
        return True

    def ImplicitFCTStep(self):
        """Implicit FCT (JNMA), called by Newton after each converged solve.

        After the Galerkin predictor: evaluate the fluxes at p^H, limit them, store the
        correction in fluxCorrection and ask Newton (fct_resolve) for the final low-order
        solve, M_L (m(p) - m^{n,FCT})/dt + A~(p) u(p) = 0, starting from p^H.
        After the final solve: nothing to do but rearm the predictor.
        """
        if self.fct_phase == 'final':
            self.fct_phase = 'predictor'
            return
        rowptr, colind, MassMatrix = self.MC_global.getCSRrepresentation()
        nFree = len(rowptr) - 1
        if self.alpha_nv is None:
            self.alpha_nv = np.zeros(nFree, 'd')
            self.m_fct = np.zeros(nFree, 'd')
        # edge-based residual at p^H: fills mLow = m(p^H), mn = m(p^n) and dt (f^H - f^L)_ij at p^H
        self.fct_phase = 'flux'
        r_scratch = np.zeros(self.u[0].dof.shape, 'd')
        self.getResidual(self.u[0].dof.copy(), r_scratch)
        self.ghostScatter(self.mLow, self.mn)
        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["bc_mask"] = self.bc_mask
        argsDict["numDOFs"] = nFree
        argsDict["dt"] = self.timeIntegration.dt
        argsDict["rho"] = self.coefficients.rho
        argsDict["gravity"] = self.coefficients.gravity
        argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["u_dof"] = self.u[0].dof
        argsDict["ML"] = self.ML
        argsDict["MC"] = MassMatrix
        argsDict["mn"] = self.mn
        argsDict["mLow"] = self.mLow
        argsDict["dt_times_fH_minus_fL"] = self.dt_times_dC_minus_dL
        argsDict["csrRowIndeces_DofLoops"] = rowptr
        argsDict["csrColumnOffsets_DofLoops"] = colind
        argsDict["thetaR"] = self.coefficients.thetaR_types
        argsDict["thetaSR"] = self.coefficients.thetaSR_types
        argsDict["freeDOFMaterialTypes"] = self.freeDOFMaterialTypes
        argsDict["prelimiter_power"] = float(self.coefficients.FCT_PRELIMITER_POWER)
        argsDict["prelimiter_head"] = int(self.coefficients.FCT_PRELIMITER_HEAD)
        argsDict["alpha_nv"] = self.alpha_nv
        argsDict["Rpos"] = self.Rpos
        argsDict["Rneg"] = self.Rneg
        argsDict["FluxCorrectionMatrix"] = self.FluxCorrectionMatrix
        argsDict["fluxCorrection"] = self.fluxCorrection
        argsDict["limited_solution"] = self.m_fct
        if getattr(self.u[0], 'par_dof', None) is None:
            argsDict["fct_pass"] = 0
            self.richards.ImplicitFCTStep(argsDict)
        else:
            # alpha_ij reads alpha_nv_j and R_j at ghost columns: the owners' values cross the cut
            argsDict["fct_pass"] = 1
            self.richards.ImplicitFCTStep(argsDict)
            self.ghostScatter(self.alpha_nv)
            argsDict["fct_pass"] = 2
            self.richards.ImplicitFCTStep(argsDict)
            self.ghostScatter(self.Rpos, self.Rneg)
            argsDict["fct_pass"] = 3
            self.richards.ImplicitFCTStep(argsDict)
        self.fct_phase = 'final'
        self.fct_resolve = True

    def FCTStep(self):
        if self.coefficients.FCT_TYPE == 'implicit':
            return self.ImplicitFCTStep()
        rowptr, colind, MassMatrix = self.MC_global.getCSRrepresentation()
        limited_solution = np.zeros((len(rowptr) - 1),'d')
        # Ghost-sync everything the limiter reads at a column before it reads it.
        # mLow/mn are pointwise in u and u_dof_old and so are already consistent,
        # but mLow is re-listed because it also carries the per-DOF soil, and ML is
        # not: ML is only ever touched at row i, never at a column.
        self.ghostScatter(self.mLow, self.mDotLow, self.min_m_bc, self.max_m_bc)
        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["bc_mask"] = self.bc_mask
        argsDict["NNZ"] = self.nnz
        argsDict["numDOFs"] = len(rowptr) - 1  # num of DOFs
        argsDict["dt"] = self.timeIntegration.dt
        argsDict["ML"] = self.ML
        argsDict["mn"] = self.mn
        argsDict["mHigh"] = self.mHigh
        argsDict["mLow"] = self.mLow
        argsDict["fluxCorrection"] = self.fluxCorrection
        argsDict["mDotLow"] = self.mDotLow
        argsDict["limited_solution"] = limited_solution
        argsDict["csrRowIndeces_DofLoops"] = rowptr
        argsDict["csrColumnOffsets_DofLoops"] = colind
        argsDict["MC"] = MassMatrix
        argsDict["dt_times_fH_minus_fL"] = self.dt_times_dC_minus_dL
        argsDict["min_m_bc"] = self.min_m_bc
        argsDict["max_m_bc"] = self.max_m_bc
        # 1 where the low-order predictor (still in u[0].dof here) is saturated
        # and the correction must vanish; all zeros leaves the limiter unchanged.
        if self.coefficients.FCT_SATURATED_LOW_ORDER and self.coefficients.PSK_type != 3:
            satLow = (self.u[0].dof[:len(rowptr) - 1] >= 0.0).astype('d')
        else:
            satLow = np.zeros((len(rowptr) - 1,), 'd')
        argsDict["satLow"] = satLow
        argsDict["Rpos"] = self.Rpos
        argsDict["Rneg"] = self.Rneg
        argsDict["FluxCorrectionMatrix"] = self.FluxCorrectionMatrix
        argsDict["LUMPED_MASS_MATRIX"] = self.coefficients.LUMPED_MASS_MATRIX
        argsDict["MONOLITHIC"] =0#cek hack self.coefficients.MONOLITHIC
        argsDict["anb_seepage_flux_n"]= self.anb_seepage_flux_n
        argsDict["elementMaterialTypes"] = self.mesh.elementMaterialTypes
        if getattr(self.u[0], 'par_dof', None) is None:
            argsDict["fct_pass"] = 0            # serial: both passes, one call
            self.richards.FCTStep(argsDict)
        else:
            # L_ij = min(Rpos_i, Rneg_j) needs the OWNER's ratio at a ghost column,
            # so the ratios have to cross the cut half way through the limiter.
            argsDict["fct_pass"] = 1            # -> FluxCorrectionMatrix, Rpos, Rneg
            self.richards.FCTStep(argsDict)
            self.ghostScatter(self.Rpos, self.Rneg)
            argsDict["fct_pass"] = 2            # -> limited_solution, fluxCorrection
            self.richards.FCTStep(argsDict)
        old_dof = self.u[0].dof.copy()
        self.invert(u=limited_solution, ulow=old_dof)
        #self.invert(u=limited_solution, ulow=self.u[0].dof) ##Original::
        #print("FCT - low",np.linalg.norm(self.u[0].dof- old_dof))
        uHigh = old_dof.copy()
        mLim  = limited_solution.copy()
        uLim  = self.u[0].dof.copy()
        du_inf = np.linalg.norm(uLim - uHigh, np.inf)
        # Conservative value to avoid instability due to large corrections -- but
        # only on the Newton-invert path (nd > 1), which is what actually goes
        # unstable when the correction is large.  With the analytic invert the cap
        # is pure damage: a threshold in psi is meaningless because dpsi/dtheta
        # spans decades over the retention curve, so a sharp front (psi jumping
        # ~10 m across one cell) trips it on most steps and throws the whole FCT
        # step away, leaving the run to alternate silently between the limited
        # scheme and the bare low-order one.
        use_newton_invert = (self.coefficients.FCT == 1 and self.coefficients.nd > 1)
        # RICHARDS_FCT_DU_MAX overrides the cap (use a huge value to disable it)
        # so the veto can be tested without editing the source.
        import os
        DU_INF_MAX = float(os.environ.get("RICHARDS_FCT_DU_MAX",
                                          1.0 if use_newton_invert else np.inf))
        # The veto has to be ONE decision for the whole domain: taken rank-locally
        # it can leave one subdomain on the limited solution and its neighbour on
        # the high order one, i.e. two different schemes inside the same step.
        # globalMax over the local flag is a no-op in serial.  du_inf below stays
        # rank-local and the log says so, rather than adding a reduction inside a
        # branch that is only entered when an env var is set.
        rejected_local = (not np.isfinite(du_inf)) or (du_inf > DU_INF_MAX)
        rejected = self.comm.globalMax(1.0 if rejected_local else 0.0) > 0.0
        if os.environ.get("RICHARDS_FCT_DBG"):
            # untouched = DOFs the invert handed back bitwise unchanged, i.e. the
            # limited mass was silently dropped there (one of the u = u_prev
            # exits in vgm_invert_newton / a zero limiter correction).
            untouched = int(np.count_nonzero(uLim == uHigh))
            dm_inf = np.linalg.norm(mLim - self.mLow, np.inf)
            m_scale = max(1.0e-30, np.linalg.norm(self.mLow, np.inf))
            logEvent("[FCT] %s du_inf=%.3e dm_inf=%.3e (rel %.3e) untouched=%d/%d dt=%.3e (rank-local norms)"
                     % ("REJECT" if rejected else "accept", du_inf, dm_inf,
                        dm_inf / m_scale, untouched, uLim.size,
                        self.timeIntegration.dt), level=1)
        if rejected:
            self.u[0].dof[:] = uHigh
        # invert() wrote every local row, ghosts included, and the ghost rows were
        # limited off an incomplete stencil.  Hand the owners' values back before
        # anything reads u again: the caller (NonlinearSolvers.Newton.solve, which
        # does u[:] = F.u[0].dof right after this) does not scatter afterwards the
        # way the shallow-water solvers in the same file do.  Scattered directly,
        # not through ghostScatter: that helper uses u[0].dof as its shuttle and
        # would restore the pre-scatter values here.
        _par = getattr(self.u[0], 'par_dof', None)
        if _par is not None:
            _par.scatter_forward_insert()
        self.timeIntegration.u[:] = self.u[0].dof
        # print("dt =", self.timeIntegration.dt)
        # print("||mLim - mLow||inf =", np.linalg.norm(mLim - self.mLow, np.inf))
        # print("||uLim - uHigh||inf =", np.linalg.norm(uLim - uHigh, np.inf))

    
    def kth_FCT_step(self):
        #import pdb
        #pdb.set_trace()
        rowptr, colind, MassMatrix = self.MC_global.getCSRrepresentation()        
        limitedFlux = np.zeros(self.nnz)
        limited_solution = np.zeros((len(rowptr) - 1),'d')
        #limited_solution[:] = self.timeIntegration.u_dof_stage[0][self.timeIntegration.lstage]
        fromFreeToGlobal=0 #direction copying
        cfemIntegrals.copyBetweenFreeUnknownsAndGlobalUnknowns(fromFreeToGlobal,
                                                               self.offset[0],
                                                               self.stride[0],
                                                               self.dirichletConditions[0].global2freeGlobal_global_dofs,
                                                               self.dirichletConditions[0].global2freeGlobal_free_dofs,
                                                               limited_solution,
                                                               self.timeintegration.u_dof_stage[0][self.timeIntegration.lstage])
                                                               #self.timeintegration.u_dof_stage[0][self.timeIntegration.lstage])


        self.richards.kth_FCT_step(
            self.timeIntegration.dt,
            self.coefficients.num_fct_iter,
            self.nnz,  # number of non zero entries
            len(rowptr) - 1,  # number of DOFs
            MassMatrix,
            self.ML,  # Lumped mass matrix
            self.u_dof_old,
            limited_solution,
            self.mDotLow,
            self.mLow,
            self.dLow,
            self.fluxMatrix,
            limitedFlux,
            rowptr,
            colind)
        #import pdb
        #pdb.set_trace()

        self.timeIntegration.u[:] = limited_solution
        #self.u[0].dof[:] = limited_solution
        fromFreeToGlobal=1 #direction copying
        cfemIntegrals.copyBetweenFreeUnknownsAndGlobalUnknowns(fromFreeToGlobal,
                                                               self.offset[0],
                                                               self.stride[0],
                                                               self.dirichletConditions[0].global2freeGlobal_global_dofs,
                                                               self.dirichletConditions[0].global2freeGlobal_free_dofs,
                                                               self.u[0].dof,
                                                               limited_solution)
    def calculateCoefficients(self):
        pass
    def calculateElementResidual(self):
        if self.globalResidualDummy != None:
            self.getResidual(self.u[0].dof,self.globalResidualDummy)
    def getResidual(self,u,r):
        import pdb
        import copy
        """
        Calculate the element residuals and add in to the global residual
        """
        cfemIntegrals.zeroJacobian_CSR(self.nNonzerosInJacobian,
                                       self.jacobian)
        if self.u_dof_old is None:
            # Pass initial condition to u_dof_old
            self.u_dof_old = np.copy(self.u[0].dof)
        rowptr, colind, nzval = self.jacobian.getCSRrepresentation()
        nnz = nzval.shape[-1]  # number of non-zero entries in sparse matrix
        r.fill(0.0)
        ########################
        ### COMPUTE C MATRIX ###
        ########################
        if self.cterm_global is None:
            # since we only need cterm_global to persist, we can drop the other self.'s
            self.cterm = {}
            self.cterm_a = {}
            self.cterm_global = {}
            self.cterm_transpose = {}
            self.cterm_a_transpose = {}
            self.cterm_global_transpose = {}
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
            np.testing.assert_almost_equal(self.ML.sum(),
                                           self.mesh.volume,
                                           err_msg="Trace of lumped mass matrix should be the domain volume", verbose=True)
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
                # C Transpose matrices
                self.cterm_transpose[d] = np.zeros((self.mesh.nElements_global,
                                                    self.nDOF_test_element[0],
                                                    self.nDOF_trial_element[0]), 'd')
                self.cterm_a_transpose[d] = nzval.copy()
                self.cterm_global_transpose[d] = SparseMat(self.nFreeDOF_global[0],
                                                           self.nFreeDOF_global[0],
                                                           nnz,
                                                           self.cterm_a_transpose[d],
                                                           colind,
                                                           rowptr)
                cfemIntegrals.zeroJacobian_CSR(self.nnz, self.cterm_global_transpose[d])
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
                                                                          self.csrRowIndeces[(0, 0)],
                                                                          self.csrColumnOffsets[(0, 0)],
                                                                          self.cterm_transpose[d],
                                                                          self.cterm_global_transpose[d])

        rowptr, colind, Cx = self.cterm_global[0].getCSRrepresentation()
        if (self.nSpace_global == 2):
            rowptr, colind, Cy = self.cterm_global[1].getCSRrepresentation()
        else:
            Cy = np.zeros(Cx.shape, 'd')
        if (self.nSpace_global == 3):
            rowptr, colind, Cz = self.cterm_global[2].getCSRrepresentation()
        else:
            Cz = np.zeros(Cx.shape, 'd')
        rowptr, colind, CTx = self.cterm_global_transpose[0].getCSRrepresentation()
        if (self.nSpace_global == 2):
            rowptr, colind, CTy = self.cterm_global_transpose[1].getCSRrepresentation()
        else:
            CTy = np.zeros(CTx.shape, 'd')
        if (self.nSpace_global == 3):
            rowptr, colind, CTz = self.cterm_global_transpose[2].getCSRrepresentation()
        else:
            CTz = np.zeros(CTx.shape, 'd')

        # This is dummy. I just care about the csr structure of the sparse matrix
        self.dLow = np.zeros(Cx.shape, 'd')
        self.fluxMatrix = np.zeros(Cx.shape, 'd')
        self.dt_times_dC_minus_dL = np.zeros(Cx.shape, 'd')
        nFree = len(rowptr)-1
        self.min_m_bc = np.ones(nFree, 'd')
        self.min_m_bc *= 1.0e10
        self.max_m_bc = np.ones(nFree, 'd')
        self.max_m_bc *= -1.0e10
        # Zalesak's ratios and the antidiffusive flux matrix.  Same lifetime as the
        # arrays above; FCTStep fills them in its first pass and consumes them in
        # its second, with a ghost scatter of the ratios in between.
        self.Rpos = np.zeros(nFree, 'd')
        self.Rneg = np.zeros(nFree, 'd')
        self.FluxCorrectionMatrix = np.zeros(Cx.shape, 'd')
        # PARALLEL (one-time): freeDOFMaterialTypes is built in __init__ from the
        # FIRST incident element in LOCAL element numbering, so on a node sitting
        # on a soil interface two ranks can pick different rocks.  mat_j selects
        # the upwind k_rw in the edge loop of Richards.h, so a disagreement makes
        # fL_ij != -fL_ji on a cut edge and the LOW ORDER scheme stops conserving,
        # with FCT off too.  The owner's choice is the one everybody must use.
        # Done here rather than in __init__ because u[0].par_dof only exists once
        # the parallel layout is built; no-op in serial and on one-material meshes.
        # The flag is only latched once the scatter could actually run, so a
        # par_dof that is not up yet on the first residual does not silently skip
        # the sync forever.  Every rank evaluates the same condition.
        if not getattr(self, '_freeDOFMaterialTypes_synced', False):
            if getattr(self.u[0], 'par_dof', None) is not None or self.comm.size() == 1:
                self.ghostScatter(self.freeDOFMaterialTypes)
                self._freeDOFMaterialTypes_synced = True
                if self.coefficients.FCT:
                    self.checkFCTParallelLayout()
        #
        # cek end computationa of cterm_global
        #
        # cek showing mquezada an example of using cterm_global sparse matrix
        # calculation y = c*x where x==1
        # direction=0
        #rowptr, colind, c = self.cterm_global[direction].getCSRrepresentation()
        #y = np.zeros((self.nFreeDOF_global[0],),'d')
        #x = np.ones((self.nFreeDOF_global[0],),'d')
        # ij=0
        # for i in range(self.nFreeDOF_global[0]):
        #    for offset in range(rowptr[i],rowptr[i+1]):
        #        j = colind[offset]
        #        y[i] += c[ij]*x[j]
        #        ij+=1
        #Load the unknowns into the finite element dof
        self.timeIntegration.calculateCoefs()
        self.timeIntegration.calculateU(u)
        self.setUnknowns(self.timeIntegration.u)
        #cek can put in logic to skip of BC's don't depend on t or u
        #Dirichlet boundary conditions
        self.numericalFlux.setDirichletValues(self.ebqe)
        #flux boundary conditions
        #cek hack, just using advective flux for flux BC for now
        for t,g in list(self.fluxBoundaryConditionsObjectsDict[0].advectiveFluxBoundaryConditionsDict.items()):
            self.ebqe[('advectiveFlux_bc',0)][t[0],t[1]] = g(self.ebqe[('x')][t[0],t[1]],self.timeIntegration.t)
            self.ebqe[('advectiveFlux_bc_flag',0)][t[0],t[1]] = 1
        # for t,g in self.fluxBoundaryConditionsObjectsDict[0].diffusiveFluxBoundaryConditionsDict.iteritems():
        #     self.ebqe[('diffusiveFlux_bc',0)][t[0],t[1]] = g(self.ebqe[('x')][t[0],t[1]],self.timeIntegration.t)
        #     self.ebqe[('diffusiveFlux_bc_flag',0)][t[0],t[1]] = 1
        #self.shockCapturing.lag=True
        self.bc_mask = np.ones_like(self.u[0].dof)
            
        if self.coefficients.forceStrongConditions:
            self.bc_mask = np.ones_like(self.u[0].dof)
            for cj in range(len(self.dirichletConditionsForceDOF)):
                for dofN,g in list(self.dirichletConditionsForceDOF[cj].DOFBoundaryConditionsDict.items()):
                    self.u[cj].dof[dofN] = g(self.dirichletConditionsForceDOF[cj].DOFBoundaryPointDict[dofN],self.timeIntegration.t)
                    # The old-time value must be g(t^n): the Newton vector that
                    # ThetaScheme copies into u_dof_old never sees the value set
                    # above, and g(t^{n+1}) would corrupt the (1-theta) fluxes.
                    dt = self.timeIntegration.dt
                    t_old = self.timeIntegration.t - (dt if dt is not None else 0.0)
                    self.u_dof_old[dofN] = g(self.dirichletConditionsForceDOF[cj].DOFBoundaryPointDict[dofN], t_old)
                    self.bc_mask[dofN] = 0.0
        degree_polynomial = 1
        try:
            degree_polynomial = self.u[0].femSpace.order
        except:
            pass
        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["bc_mask"] = self.bc_mask
        argsDict["dt"] = self.timeIntegration.dt
        # ThetaScheme.theta weights the high-order (Galerkin) fluxes; the low-order
        # edge-based operator is always backward Euler (Theta = 1).
        time_theta = getattr(self.timeIntegration, 'theta', 1.0)
        argsDict["Theta"] = time_theta if self.kernelStabilizationType() == 0 else 1.0
        argsDict["Theta_h"] = time_theta
        argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["mesh_velocity_dof"] = self.mesh.nodeVelocityArray
        argsDict["MOVING_DOMAIN"] = self.MOVING_DOMAIN
        argsDict["mesh_l2g"] = self.mesh.elementNodesArray
        argsDict["dV_ref"] = self.elementQuadratureWeights[('u',0)]
        argsDict["u_trial_ref"] = self.u[0].femSpace.psi
        argsDict["u_grad_trial_ref"] = self.u[0].femSpace.grad_psi
        argsDict["u_test_ref"] = self.u[0].femSpace.psi
        argsDict["u_grad_test_ref"] = self.u[0].femSpace.grad_psi
        argsDict["mesh_trial_trace_ref"] = self.u[0].femSpace.elementMaps.psi_trace
        argsDict["mesh_grad_trial_trace_ref"] = self.u[0].femSpace.elementMaps.grad_psi_trace
        argsDict["dS_ref"] = self.elementBoundaryQuadratureWeights[('u',0)]
        argsDict["u_trial_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["u_grad_trial_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["u_test_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["u_grad_test_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["normal_ref"] = self.u[0].femSpace.elementMaps.boundaryNormals
        argsDict["boundaryJac_ref"] = self.u[0].femSpace.elementMaps.boundaryJacobians
        argsDict["nElements_global"] = self.mesh.nElements_global
        argsDict["ebqe_penalty_ext"] = self.ebqe['penalty']
        argsDict["elementMaterialTypes"] = self.mesh.elementMaterialTypes
        argsDict["isSeepageFace"] = self.coefficients.isSeepageFace
        argsDict["a_rowptr"] = self.coefficients.sdInfo[(0,0)][0]
        argsDict["a_colind"] = self.coefficients.sdInfo[(0,0)][1]
        argsDict["rho"] = self.coefficients.rho
        argsDict["beta"] = self.coefficients.beta

        argsDict["q_rho"]= self.q['rho']
        argsDict["ebqe_rho"]= self.ebqe['rho']
        
        argsDict["gravity"] = self.coefficients.gravity
        argsDict["alpha"] = self.coefficients.vgm_alpha_types
        argsDict["n"] = self.coefficients.vgm_n_types
        argsDict["thetaR"] = self.coefficients.thetaR_types
        argsDict["thetaSR"] = self.coefficients.thetaSR_types
        argsDict["KWs"] = self.coefficients.Ksw_types
        argsDict["PSK_TYPE"] = self.coefficients.PSK_type
        argsDict["useMetrics"] = 0.0
        argsDict["alphaBDF"] = self.timeIntegration.alpha_bdf
        argsDict["lag_shockCapturing"] = 0
        argsDict["shockCapturingDiffusion"] = self.coefficients.SC
        argsDict["VMS"] = self.coefficients.VMS
        argsDict["sc_uref"] = 1.0
        argsDict["sc_alpha"] = 2.0
        argsDict["u_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["r_l2g"] = self.l2g[0]['freeGlobal']
        argsDict["elementDiameter"] = self.mesh.elementDiametersArray
        argsDict["degree_polynomial"] = degree_polynomial
        argsDict["u_dof"] = self.u[0].dof
        argsDict["u_dof_old"] = self.u_dof_old
        argsDict["velocity"] = self.q[('velocity',0)]
        
        argsDict["velocity_couple"] = self.q[('velocity_couple',0)]

        argsDict["q_m"] = self.timeIntegration.m_tmp[0]
        argsDict["q_theta"] = self.q[('theta',0)]
        ############################################
        self.q[('m',0)][:] = self.timeIntegration.m_tmp[0]
        #############################################
        #argsDict["q_x"] = self.q['x']    
        argsDict["q_u"] = self.q[('u',0)]
        argsDict["q_dV"] = self.q[('dV_u',0)]
        argsDict["q_m_betaBDF"] = self.timeIntegration.beta_bdf[0]
        argsDict["cfl"] = self.q[('cfl',0)]
        argsDict["q_numDiff_u"] = self.q[('numDiff',0,0)]
        #argsDict["q_numDiff_u_last"] = self.q[('numDiff_last',0,0)]
        argsDict["q_numDiff_u_last"] = self.numDiff_star
        argsDict["offset_u"] = self.offset[0]
        argsDict["stride_u"] = self.stride[0]
        argsDict["globalResidual"] = r
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["ebqe_velocity_ext"] = self.ebqe[('velocity',0)]
        argsDict["ebqe_velocity_ext_couple"] = self.ebqe[('velocity_couple',0)]      
        argsDict["isDOFBoundary_u"] = self.numericalFlux.isDOFBoundary[0]
        argsDict["ebqe_bc_u_ext"] = self.numericalFlux.ebqe[('u',0)]
        argsDict["isFluxBoundary_u"] = self.ebqe[('advectiveFlux_bc_flag',0)]
        argsDict["ebqe_bc_flux_ext"] = self.ebqe[('advectiveFlux_bc',0)]
        argsDict["ebqe_phi"] = self.ebqe[('u',0)]
        argsDict["epsFact"] = 0.0
        #argsDict["ebqe_x"] = self.ebqe['x']
        argsDict["ebqe_u"] = self.ebqe[('u',0)]
        argsDict["ebqe_theta"] = self.ebqe[('theta',0)]
        argsDict["ebqe_flux"] = self.ebqe[('advectiveFlux',0)]
        argsDict['STABILIZATION_TYPE'] = self.kernelStabilizationType()
        # ENTROPY VISCOSITY and ARTIFICIAL COMRPESSION
        argsDict["cE"] = self.coefficients.cE
        argsDict["cK"] = self.coefficients.cK
        # PARAMETERS FOR LOG BASED ENTROPY FUNCTION
        argsDict["uL"] = self.coefficients.uL
        argsDict["uR"] = self.coefficients.uR
        # PARAMETERS FOR EDGE VISCOSITY
        argsDict["numDOFs"] = len(rowptr) - 1  # num of DOFs
        argsDict["NNZ"] = self.nnz 
        argsDict["Cx"] = len(Cx)  # num of non-zero entries in the sparsity pattern
        argsDict["csrRowIndeces_DofLoops"] = rowptr  # Row indices for Sparsity Pattern (convenient for DOF loops)
        argsDict["csrColumnOffsets_DofLoops"] = colind  # Column indices for Sparsity Pattern (convenient for DOF loops)
        argsDict["csrRowIndeces_CellLoops"] = self.csrRowIndeces[(0, 0)]  # row indices (convenient for element loops)
        argsDict["csrColumnOffsets_CellLoops"] = self.csrColumnOffsets[(0, 0)]  # column indices (convenient for element loops)
        argsDict["csrColumnOffsets_eb_CellLoops"] = self.csrColumnOffsets_eb[(0, 0)]  # indices for boundary terms
        argsDict["globalJacobian"] = self.jacobian.getCSRrepresentation()[2]
        # C matrices
        argsDict["Cx"] = Cx
        argsDict["Cy"] = Cy
        argsDict["Cz"] = Cz
        argsDict["CTx"] = CTx
        argsDict["CTy"] = CTy
        argsDict["CTz"] = CTz
        argsDict["ML"] = self.ML
        if self.delta_x_ij is None:
            self.delta_x_ij = -np.ones((self.nNonzerosInJacobian*3,),'d')
        argsDict["delta_x_ij"] = self.delta_x_ij
        argsDict["MC"] = self.MC_a
        # PARAMETERS FOR 1st or 2nd ORDER MPP METHOD
        argsDict["LUMPED_MASS_MATRIX"] = self.coefficients.LUMPED_MASS_MATRIX
        argsDict["STABILIZATTION_TYPE"] = self.kernelStabilizationType()
        argsDict["ENTROPY_TYPE"] = self.coefficients.ENTROPY_TYPE
        # FLUX CORRECTED TRANSPORT
        argsDict["dLow"] = self.dLow
        argsDict["fluxMatrix"] = self.fluxMatrix
        argsDict["mDotLow"] = self.mDotLow
        argsDict["fluxCorrection"] = self.fluxCorrection
        limited_solution = np.zeros((len(rowptr) - 1),'d')
        argsDict["limited_solution"] = limited_solution
        argsDict["MONOLITHIC"] =0
        argsDict["mLow"] = self.mLow
        argsDict["dt_times_fH_minus_fL"] = self.dt_times_dC_minus_dL
        argsDict["min_m_bc"] = self.min_m_bc
        argsDict["max_m_bc"] = self.max_m_bc
        # Needed here as well as in FCTStep: STABILIZATION_TYPE=Implicit_FCT calls
        # FCTStep from inside the residual off this same dict, and arguments_dict
        # throws on a missing key.  fct_pass=0 keeps that in-kernel call single
        # pass -- it cannot scatter, so it stays serial only.
        argsDict["Rpos"] = self.Rpos
        argsDict["Rneg"] = self.Rneg
        argsDict["FluxCorrectionMatrix"] = self.FluxCorrectionMatrix
        argsDict["fct_pass"] = 0
        argsDict["quantDOFs"] = self.quantDOFs
        argsDict["mn"] = self.mn
        argsDict["anb_seepage_flux_n"]= self.anb_seepage_flux_n
        argsDict["freeDOFMaterialTypes"] = self.freeDOFMaterialTypes
        ######################################################################################
        argsDict["pn"] = self.u[0].dof
        argsDict["mHigh"] = self.mHigh

        rowptr, colind, MassMatrix = self.MC_global.getCSRrepresentation()
        argsDict["MassMatrix"] = MassMatrix
        
######################################################################################        
        #argsDict["anb_seepage_flux"] = self.coefficients.anb_seepage_flux
        argsDict["anb_seepage_flux"] = self.anb_seepage_flux
        argsDict["q_velocity"] = self.q[('grad(u_v)', 0)]
        argsDict["csrRowIndeces_u_u"] = self.csrRowIndeces[(0,0)]
        argsDict["csrColumnOffsets_u_u"] = self.csrColumnOffsets[(0,0)]
        argsDict["csrColumnOffsets_eb_u_u"] = self.csrColumnOffsets_eb[(0,0)]
        #argsDict["q_grad_psi"] = self.q[('velocity', 0)] 
        #argsDict["anb_seepage_flux_n"] = self.coefficients.anb_seepage_flux_n
        
        # from mpi4py import MPI
        # comm = MPI.COMM_WORLD
        # rank = comm.Get_rank()
        #
        # seepage_flux_value = np.sum(self.anb_seepage_flux_n)
        # if seepage_flux_value > 0.0:
        # # Each processor writes its own flux with its rank
        #     with open("seepage_flux_try.txt", "a") as f:
        #         f.write(f"Rank {rank}:, {self.timeIntegration.t:.6f}, {seepage_flux_value:.8f}\n")

        if (self.kernelStabilizationType() == 0):  # SUPG
            self.calculateResidual = self.richards.calculateResidual
            self.calculateJacobian = self.richards.calculateJacobian
        else:
            self.calculateResidual = self.richards.calculateResidual_entropy_viscosity
            self.calculateJacobian = self.richards.calculateMassMatrix
        
        if self.delta_x_ij is None:
            self.delta_x_ij = -np.ones((self.nNonzerosInJacobian*3,),'d')
        # edge-based kernels assemble their own (Newton or L-scheme) matrix with the residual
        argsDict["LSCHEME_L"] = float(self.coefficients.LSCHEME_L) if self.lschemeOn() else 0.0
        if self.defectCorrection():
            # Defect correction: assemble the low-order L-scheme matrix (L/dt) M_L + A~(p^k) into
            # self.jacobian through the edge-based kernel (its residual is discarded); getJacobian
            # then uses it as is, while r below is the Galerkin residual.
            if not hasattr(self, '_dc_scratch') or self._dc_scratch.shape != r.shape:
                self._dc_scratch = np.zeros(r.shape, 'd')
            self._dc_scratch[:] = 0.0
            argsDict["STABILIZATION_TYPE"] = 2
            argsDict["STABILIZATTION_TYPE"] = 2
            argsDict["Theta"] = 1.0
            argsDict["globalResidual"] = self._dc_scratch
            argsDict["LSCHEME_L"] = float(self.coefficients.LSCHEME_L)
            self.richards.calculateResidual_entropy_viscosity(argsDict)
            argsDict["STABILIZATION_TYPE"] = 0
            argsDict["STABILIZATTION_TYPE"] = 0
            argsDict["Theta"] = time_theta
            argsDict["globalResidual"] = r
            argsDict["LSCHEME_L"] = 0.0
        self.calculateResidual(argsDict)
        # if getattr(self, "_theta_log_count", 0) < 5:
        #     q_theta = self.q[('theta',0)]
        #     ebqe_theta = self.ebqe[('theta',0)]
        #     logEvent("[Richards q_theta] t={:.6e} q(min,max,mean)=({:.6e},{:.6e},{:.6e}) ebqe(min,max,mean)=({:.6e},{:.6e},{:.6e}) q_zero_count={} ebqe_zero_count={}".format(
        #              self.timeIntegration.t,
        #              float(np.min(q_theta)), float(np.max(q_theta)), float(np.mean(q_theta)),
        #              float(np.min(ebqe_theta)), float(np.max(ebqe_theta)), float(np.mean(ebqe_theta)),
        #              int(np.count_nonzero(q_theta == 0.0)), int(np.count_nonzero(ebqe_theta == 0.0))),
        #              level=2)
        #     self._theta_log_count = getattr(self, "_theta_log_count", 0) + 1
        


        self.q[('mt',0)][:] =self.timeIntegration.m_tmp[0]
        #self.q[('mt',0)] *= self.timeIntegration.alpha_bdf
        #self.q[('mt',0)] += self.timeIntegration.beta_bdf[0]
        #self.timeIntegration.calculateElementCoefficients(self.q)
        if self.coefficients.forceStrongConditions:#
            for cj in range(len(self.dirichletConditionsForceDOF)):#
                for dofN,g in list(self.dirichletConditionsForceDOF[cj].DOFBoundaryConditionsDict.items()):
                     r[self.offset[cj]+self.stride[cj]*dofN] = 0
        if self.stabilization:
            self.stabilization.accumulateSubgridMassHistory(self.q)
        logEvent("Global residual",level=9,data=r)
        self.nonlinear_function_evaluations += 1
        if self.globalResidualDummy is None:
            self.globalResidualDummy = np.zeros(r.shape,'d')

    def invert(self, u, r=None, ulow=None):
        """
        Unified invert method that handles both standard and FCT modes
        using self.coefficients.FCT as the switching flag.
        """
        import numpy as np
        import copy
    
        self.mHigh[:] = u
        if ulow is not None:
            self.u[0].dof[:] = ulow
    
        rowptr, colind, nzval = self.jacobian.getCSRrepresentation()
        nnz = nzval.shape[-1]
    
        rowptr, colind, Cx = self.cterm_global[0].getCSRrepresentation()
        Cy = self.cterm_global[1].getCSRrepresentation()[2] if self.nSpace_global >= 2 else np.zeros(Cx.shape, 'd')
        Cz = self.cterm_global[2].getCSRrepresentation()[2] if self.nSpace_global == 3 else np.zeros(Cx.shape, 'd')
    
        rowptr, colind, CTx = self.cterm_global_transpose[0].getCSRrepresentation()
        CTy = self.cterm_global_transpose[1].getCSRrepresentation()[2] if self.nSpace_global >= 2 else np.zeros(CTx.shape, 'd')
        CTz = self.cterm_global_transpose[2].getCSRrepresentation()[2] if self.nSpace_global == 3 else np.zeros(CTx.shape, 'd')
    
        degree_polynomial = getattr(self.u[0].femSpace, "order", 1)
    
        if self.delta_x_ij is None:
            self.delta_x_ij = -np.ones((self.nNonzerosInJacobian * 3,), 'd')
    
        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["dt"] = self.timeIntegration.dt
        argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["mesh_velocity_dof"] = self.mesh.nodeVelocityArray
        argsDict["MOVING_DOMAIN"] = self.MOVING_DOMAIN
        argsDict["mesh_l2g"] = self.mesh.elementNodesArray
        argsDict["dV_ref"] = self.elementQuadratureWeights[('u',0)]
        argsDict["u_trial_ref"] = self.u[0].femSpace.psi
        argsDict["u_grad_trial_ref"] = self.u[0].femSpace.grad_psi
        argsDict["u_test_ref"] = self.u[0].femSpace.psi
        argsDict["u_grad_test_ref"] = self.u[0].femSpace.grad_psi
        argsDict["mesh_trial_trace_ref"] = self.u[0].femSpace.elementMaps.psi_trace
        argsDict["mesh_grad_trial_trace_ref"] = self.u[0].femSpace.elementMaps.grad_psi_trace
        argsDict["dS_ref"] = self.elementBoundaryQuadratureWeights[('u',0)]
        argsDict["u_trial_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["u_grad_trial_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["u_test_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["u_grad_test_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["normal_ref"] = self.u[0].femSpace.elementMaps.boundaryNormals
        argsDict["boundaryJac_ref"] = self.u[0].femSpace.elementMaps.boundaryJacobians
        argsDict["nElements_global"] = self.mesh.nElements_global
        argsDict["ebqe_penalty_ext"] = self.ebqe['penalty']
        argsDict["elementMaterialTypes"] = self.mesh.elementMaterialTypes
        argsDict["isSeepageFace"] = self.coefficients.isSeepageFace
        argsDict["a_rowptr"] = self.coefficients.sdInfo[(0,0)][0]
        argsDict["a_colind"] = self.coefficients.sdInfo[(0,0)][1]
        argsDict["rho"] = self.coefficients.rho
        argsDict["beta"] = self.coefficients.beta
        argsDict["gravity"] = self.coefficients.gravity
        argsDict["alpha"] = self.coefficients.vgm_alpha_types
        argsDict["n"] = self.coefficients.vgm_n_types
        argsDict["thetaR"] = self.coefficients.thetaR_types
        argsDict["thetaSR"] = self.coefficients.thetaSR_types
        argsDict["KWs"] = self.coefficients.Ksw_types
        argsDict["PSK_TYPE"] = self.coefficients.PSK_type
        argsDict["useMetrics"] = 0.0
        argsDict["alphaBDF"] = self.timeIntegration.alpha_bdf
        argsDict["lag_shockCapturing"] = 0
        argsDict["shockCapturingDiffusion"] = self.coefficients.SC
        argsDict["sc_uref"] = 1.0
        argsDict["sc_alpha"] = 2.0
        argsDict["u_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["r_l2g"] = self.l2g[0]['freeGlobal']
        argsDict["elementDiameter"] = self.mesh.elementDiametersArray
        argsDict["degree_polynomial"] = degree_polynomial
        argsDict["u_dof"] = self.u[0].dof
        argsDict["u_dof_old"] = self.u[0].dof
        argsDict["velocity"] = self.q['velocity',0]
        argsDict["q_m"] = self.timeIntegration.m_tmp[0]
        argsDict["q_theta"] = self.q[('theta',0)]
        argsDict["q_u"] = self.q[('u',0)]
        argsDict["q_dV"] = self.q[('dV_u',0)]
        argsDict["q_m_betaBDF"] = self.timeIntegration.beta_bdf[0]
        argsDict["cfl"] = self.q[('cfl',0)]
        argsDict["q_numDiff_u"] = self.q[('numDiff',0,0)]
        argsDict["q_numDiff_u_last"] = self.q[('numDiff_last',0,0)]
        argsDict["q_numDiff_u_last"] = self.numDiff_star
        argsDict["offset_u"] = self.offset[0]
        argsDict["stride_u"] = self.stride[0]
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["ebqe_velocity_ext"] = self.ebqe['velocity',0]
        argsDict["isDOFBoundary_u"] = self.numericalFlux.isDOFBoundary[0]
        argsDict["ebqe_bc_u_ext"] = self.numericalFlux.ebqe[('u',0)]
        argsDict["isFluxBoundary_u"] = self.ebqe[('advectiveFlux_bc_flag',0)]
        argsDict["ebqe_bc_flux_ext"] = self.ebqe[('advectiveFlux_bc',0)]
        argsDict["ebqe_phi"] = self.ebqe[('u',0)]
        argsDict["epsFact"] = 0.0
        argsDict["ebqe_u"] = self.ebqe[('u',0)]
        argsDict["ebqe_theta"] = self.ebqe[('theta',0)]
        argsDict["ebqe_flux"] = self.ebqe[('advectiveFlux',0)]
        argsDict["STABILIZATION_TYPE"] = self.coefficients.STABILIZATION_TYPE
        argsDict["cE"] = self.coefficients.cE
        argsDict["cK"] = self.coefficients.cK
        argsDict["uL"] = self.coefficients.uL
        argsDict["uR"] = self.coefficients.uR
        argsDict["numDOFs"] = len(rowptr) - 1
        argsDict["NNZ"] = self.nnz
        argsDict["csrRowIndeces_DofLoops"] = rowptr
        argsDict["csrColumnOffsets_DofLoops"] = colind
        argsDict["csrRowIndeces_CellLoops"] = self.csrRowIndeces[(0, 0)]
        argsDict["csrColumnOffsets_CellLoops"] = self.csrColumnOffsets[(0, 0)]
        argsDict["csrColumnOffsets_eb_CellLoops"] = self.csrColumnOffsets_eb[(0, 0)]
        argsDict["Cx"] = Cx
        argsDict["Cy"] = Cy
        argsDict["Cz"] = Cz
        argsDict["CTx"] = CTx
        argsDict["CTy"] = CTy
        argsDict["CTz"] = CTz
        argsDict["ML"] = self.ML
        argsDict["delta_x_ij"] = self.delta_x_ij
        argsDict["LUMPED_MASS_MATRIX"] = self.coefficients.LUMPED_MASS_MATRIX
        argsDict["ENTROPY_TYPE"] = self.coefficients.ENTROPY_TYPE
        argsDict["dLow"] = self.dLow
        argsDict["fluxMatrix"] = self.fluxMatrix
        argsDict["mDotLow"] = self.mDotLow
        argsDict["dt_times_fH_minus_fL"] = self.dt_times_dC_minus_dL
        argsDict["min_m_bc"] = self.min_m_bc
        argsDict["max_m_bc"] = self.max_m_bc
        argsDict["quantDOFs"] = self.quantDOFs
        argsDict["mn"] = self.mn
        argsDict["anb_seepage_flux"] = self.coefficients.anb_seepage_flux
        argsDict["limited_solution"] = u
        argsDict["mLow"] = self.u[0].dof
        argsDict["freeDOFMaterialTypes"] = self.freeDOFMaterialTypes
        argsDict["USE_NEWTON_INVERT"] = 1 if (self.coefficients.FCT==1 and self.coefficients.nd > 1) else 0
        self.richards.invert(argsDict)
     
    def getJacobian(self,jacobian):
        if (self.kernelStabilizationType() == 0 and not self.defectCorrection()):  # SUPG
            cfemIntegrals.zeroJacobian_CSR(self.nNonzerosInJacobian,
                                           jacobian)
        degree_polynomial = 1
        try:
            degree_polynomial = self.u[0].femSpace.order
        except:
            pass
        if self.delta_x_ij is None:
            self.delta_x_ij = -np.ones((self.nNonzerosInJacobian*3,),'d')
        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["dt"] = self.timeIntegration.dt
        argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["mesh_velocity_dof"] = self.mesh.nodeVelocityArray
        argsDict["MOVING_DOMAIN"] = self.MOVING_DOMAIN
        argsDict["mesh_l2g"] = self.mesh.elementNodesArray
        argsDict["dV_ref"] = self.elementQuadratureWeights[('u',0)]
        argsDict["u_trial_ref"] = self.u[0].femSpace.psi
        argsDict["u_grad_trial_ref"] = self.u[0].femSpace.grad_psi
        argsDict["u_test_ref"] = self.u[0].femSpace.psi
        argsDict["u_grad_test_ref"] = self.u[0].femSpace.grad_psi
        argsDict["mesh_trial_trace_ref"] = self.u[0].femSpace.elementMaps.psi_trace
        argsDict["mesh_grad_trial_trace_ref"] = self.u[0].femSpace.elementMaps.grad_psi_trace
        argsDict["dS_ref"] = self.elementBoundaryQuadratureWeights[('u',0)]
        argsDict["u_trial_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["u_grad_trial_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["u_test_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["u_grad_test_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["normal_ref"] = self.u[0].femSpace.elementMaps.boundaryNormals
        argsDict["boundaryJac_ref"] = self.u[0].femSpace.elementMaps.boundaryJacobians
        argsDict["nElements_global"] = self.mesh.nElements_global
        argsDict["ebqe_penalty_ext"] = self.ebqe['penalty']
        argsDict["elementMaterialTypes"] = self.mesh.elementMaterialTypes
        argsDict["isSeepageFace"] = self.coefficients.isSeepageFace
        argsDict["a_rowptr"] = self.coefficients.sdInfo[(0,0)][0]
        argsDict["a_colind"] = self.coefficients.sdInfo[(0,0)][1]
        argsDict["rho"] = self.coefficients.rho
        argsDict["beta"] = self.coefficients.beta

        argsDict["q_rho"]= self.q['rho']
        argsDict["ebqe_rho"]= self.ebqe['rho']
        
        argsDict["gravity"] = self.coefficients.gravity
        argsDict["alpha"] = self.coefficients.vgm_alpha_types
        argsDict["n"] = self.coefficients.vgm_n_types
        argsDict["thetaR"] = self.coefficients.thetaR_types
        argsDict["thetaSR"] = self.coefficients.thetaSR_types
        argsDict["KWs"] = self.coefficients.Ksw_types
        argsDict["PSK_TYPE"] = self.coefficients.PSK_type
        argsDict["useMetrics"] = 0.0
        argsDict["alphaBDF"] = self.timeIntegration.alpha_bdf
        argsDict["lag_shockCapturing"] = 0
        argsDict["shockCapturingDiffusion"] = 0.1
        argsDict["u_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["r_l2g"] = self.l2g[0]['freeGlobal']
        argsDict["elementDiameter"] = self.mesh.elementDiametersArray
        argsDict["degree_polynomial"] = degree_polynomial
        argsDict["u_dof"] = self.u[0].dof
        argsDict["velocity"] = self.q['velocity',0]
        argsDict["q_m_betaBDF"] = self.timeIntegration.beta_bdf[0]
        argsDict["cfl"] = self.q[('cfl',0)]
        argsDict["q_numDiff_u"] = self.q[('numDiff',0,0)]
        argsDict["q_numDiff_u_last"] = self.q[('numDiff',0,0)]
        argsDict["csrRowIndeces_u_u"] = self.csrRowIndeces[(0,0)]
        argsDict["csrColumnOffsets_u_u"] = self.csrColumnOffsets[(0,0)]
        argsDict["globalJacobian"] = jacobian.getCSRrepresentation()[2]
        argsDict["delta_x_ij"] = self.delta_x_ij
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["ebqe_velocity_ext"] = self.ebqe['velocity',0 ]
        argsDict["isDOFBoundary_u"] = self.numericalFlux.isDOFBoundary[0]
        argsDict["ebqe_bc_u_ext"] = self.numericalFlux.ebqe[('u',0)]
        argsDict["isFluxBoundary_u"] = self.ebqe[('advectiveFlux_bc_flag',0)]
        argsDict["ebqe_bc_flux_ext"] = self.ebqe[('advectiveFlux_bc',0)]
        argsDict["csrColumnOffsets_eb_u_u"] = self.csrColumnOffsets_eb[(0,0)]
        argsDict["LUMPED_MASS_MATRIX"] = self.coefficients.LUMPED_MASS_MATRIX
        argsDict["VMS"] = self.coefficients.VMS
        argsDict["Theta"] = getattr(self.timeIntegration, 'theta', 1.0) if self.kernelStabilizationType() == 0 else 1.0
        #argsDict["anb_seepage_flux"] = self.coefficients.anb_seepage_flux

        if not self.defectCorrection():
            self.calculateJacobian(argsDict)
        # else: the L-scheme matrix was assembled with the residual (see getResidual)
        if self.coefficients.forceStrongConditions:
            for dofN in list(self.dirichletConditionsForceDOF[0].DOFBoundaryConditionsDict.keys()):
                global_dofN = self.offset[0]+self.stride[0]*dofN
                self.nzval[np.where(self.colind == global_dofN)] = 0.0 #column
                self.nzval[self.rowptr[global_dofN]:self.rowptr[global_dofN+1]] = 0.0 #row
                zeroRow=True
                for i in range(self.rowptr[global_dofN],self.rowptr[global_dofN+1]):#row
                    if (self.colind[i] == global_dofN):
                        self.nzval[i] = 1.0
                        zeroRow = False
                if zeroRow:
                    raise RuntimeError("Jacobian has a zero row because sparse matrix has no diagonal entry at row "+repr(global_dofN)+". You probably need add diagonal mass or reaction term")
            #scaling = 1.0#probably want to add some scaling to match non-dirichlet diagonals in linear system 
            #for cj in range(self.nc):
            #    for dofN in list(self.dirichletConditionsForceDOF[cj].DOFBoundaryConditionsDict.keys()):
            #        global_dofN = self.offset[cj]+self.stride[cj]*dofN
            #        for i in range(self.rowptr[global_dofN],self.rowptr[global_dofN+1]):
            #            if (self.colind[i] == global_dofN):
            #                self.nzval[i] = scaling
            #            else:
            #                self.nzval[i] = 0.0
        logEvent("Jacobian ",level=10,data=jacobian)
        #mwf decide if this is reasonable for solver statistics
        self.nonlinear_function_jacobian_evaluations += 1
        return jacobian
    def calculateElementQuadrature(self):
        """
        Calculate the physical location and weights of the quadrature rules
        and the shape information at the quadrature points.

        This function should be called only when the mesh changes.
        """
        #self.u[0].femSpace.elementMaps.getValues(self.elementQuadraturePoints,
        #                                         self.q['x'])
        self.u[0].femSpace.elementMaps.getBasisValuesRef(self.elementQuadraturePoints)
        self.u[0].femSpace.elementMaps.getBasisGradientValuesRef(self.elementQuadraturePoints)
        self.u[0].femSpace.getBasisValuesRef(self.elementQuadraturePoints)
        self.u[0].femSpace.getBasisGradientValuesRef(self.elementQuadraturePoints)
        self.coefficients.initializeElementQuadrature(self.timeIntegration.t,self.q)
        if self.stabilization != None:
            self.stabilization.initializeElementQuadrature(self.mesh,self.timeIntegration.t,self.q)
            self.stabilization.initializeTimeIntegration(self.timeIntegration)
        if self.shockCapturing != None:
            self.shockCapturing.initializeElementQuadrature(self.mesh,self.timeIntegration.t,self.q)
    def calculateElementBoundaryQuadrature(self):
        pass
    def calculateExteriorElementBoundaryQuadrature(self):
        """
        Calculate the physical location and weights of the quadrature rules
        and the shape information at the quadrature points on global element boundaries.

        This function should be called only when the mesh changes.
        """
        #
        #get physical locations of element boundary quadrature points
        #
        #assume all components live on the same mesh
        self.u[0].femSpace.elementMaps.getBasisValuesTraceRef(self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.elementMaps.getBasisGradientValuesTraceRef(self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.getBasisValuesTraceRef(self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.getBasisGradientValuesTraceRef(self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.elementMaps.getValuesGlobalExteriorTrace(self.elementBoundaryQuadraturePoints,
                                                                    self.ebqe['x'])
        self.fluxBoundaryConditionsObjectsDict = dict([(cj,FluxBoundaryConditions(self.mesh,
                                                                                  self.nElementBoundaryQuadraturePoints_elementBoundary,
                                                                                  self.ebqe[('x')],
                                                                                  getAdvectiveFluxBoundaryConditions=self.advectiveFluxBoundaryConditionsSetterDict[cj],
                                                                                  getDiffusiveFluxBoundaryConditions=self.diffusiveFluxBoundaryConditionsSetterDictDict[cj]))
                                                       for cj in list(self.advectiveFluxBoundaryConditionsSetterDict.keys())])
        self.coefficients.initializeGlobalExteriorElementBoundaryQuadrature(self.timeIntegration.t,self.ebqe)
        #argsDict = cArgumentsDict.ArgumentsDict()
        #argsDict["anb_seepage_flux"] = self.coefficients.anb_seepage_flux
        #print("Hi", self.coefficients.anb_seepage_flux)

        #print("The seepage is ", anb_seepage_flux)
    def estimate_mt(self):
        pass
    def calculateSolutionAtQuadrature(self):
        pass
    def calculateAuxiliaryQuantitiesAfterStep(self):
        pass
    #def postStep(self, t, firstStep=False):
    #    with open('seepage_flux_nnnnk', "a") as f:
    #        f.write("\n Time"+ ",\t" +"Seepage\n")
    #        f.write(repr(t)+ ",\t" +repr(self.coefficients.anb_seepage_flux))

    

#argsDict["anb_seepage_flux"] = self.coefficients.anb_seepage_flux        
#anb_seepage_flux= self.coefficients.anb_seepage_flux
#print("Hi",anb_seepage_flux)


#print("Hello from the python file", self.coefficients.anb_seepage_flux)
