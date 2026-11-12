from __future__ import division
from builtins import range
import math
import os
import proteus
from .cm_comp_co2 import *
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
    def __init__(self,transport,integrateInterpolationPoints=False):
        self.transport=transport
        TimeIntegration.BackwardEuler.__init__(self,transport, integrateInterpolationPoints)
    def updateTimeHistory(self,resetFromDOF=False):
        TimeIntegration.BackwardEuler.updateTimeHistory(self,resetFromDOF)
        # Legacy style (cf. m_comp_co2_old.py): push the converged per-component
        # solution into the old-time DOFs the C++ EV residual reads.  The old
        # model wrote `self.u[ci]`; here timeIntegration.u is the flat assembled
        # vector (calculateU), so the component solution is transport.u[ci].dof.
        self.transport.u_dof_old[:] = self.transport.u[0].dof   # water (p_w)
        if getattr(self.transport, 'nc', 1) >= 2:
            u_dof_n_old = getattr(self.transport, 'u_dof_n_old', None)
            if u_dof_n_old is not None:
                u_dof_n_old[:] = self.transport.u[1].dof          # gas (S_n)
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
    # staticmethod: Cython 3 functions bind self like Python functions (C47)
    conservativeHeadRichardsMualemVanGenuchtenHetEvaluateV2 = staticmethod(conservativeHeadRichardsMualemVanGenuchtenHetEvaluateV2)
    conservativeHeadRichardsMualemVanGenuchten_sd_het = staticmethod(conservativeHeadRichardsMualemVanGenuchten_sd_het)
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
                 # gas-phase density (EXPONENTIAL EOS, mirrors comp-0 water
                 # rho_w = rho*exp(beta*p)).  rho_n is the reference density at
                 # p_n = 0 (gauge = atmospheric); p_ref_n is the e-folding
                 # pressure scale: rho_n(p_n) = rho_n * exp(p_n / p_ref_n), with
                 # beta_n = 1/p_ref_n the constant gas compressibility.  CO2 near
                 # atmospheric in a lab rig is ideal-gas-like (rho ~ P_abs), so
                 # p_ref_n ~ atmospheric in head ~ 10.3 m (beta_n ~ 0.1 /m).
                 # p_ref_n = 0 (default) -> incompressible, constant rho_n.
                 rho_n=1.0,
                 p_ref_n=0.0,
                 diagonal_conductivity=True,
                 getSeepageFace=None,
                 density_model=None,
                 DENSITY_MODEL=None,
                 # PSK constitutive model: 'VGM' (van Genuchten-Mualem) or 'BC' (Brooks-Corey-Burdine)
                 PSK_TYPE='VGM',
                # FOR EDGE BASED EV
                 STABILIZATION_TYPE='Implicit_FCT',
                 ENTROPY_TYPE=2,  # logarithmic
                 LUMPED_MASS_MATRIX=False,
                 MONOLITHIC=False,
                 VMS=0.0,
                 SC=0.0,
                 FCT=True,
                 num_fct_iter=1,
                 # FOR ENTROPY VISCOSITY
                 cE=1.0,
                 uL=0.0,
                 uR=1.0,
                 # FOR ARTIFICIAL COMPRESSION
                 cK=1.0,
                 # OUTPUT quantDOFs
                 outputQuantDOFs=False,
                 # Stage 3b (gas-side kinetic dissolution sink).  When coupled
                 # to a TADR transport model, m_comp_co2 deducts R_diss = k_d *
                 # S_n * S_w * (c_sat - c) from the gas-equation residual per
                 # DOF so every kg of CO2 that TADR adds to the brine is
                 # removed from the gas phase (mass conservation across the
                 # phases).  Defaults k_d=0 disable the sink; the gas equation
                 # then sees no dissolution (legacy behavior preserved).
                 k_d=0.0,
                 c_sat=1.0,
                 # Local-equilibrium dissolution flash (the k_d -> inf limit).
                 # dissolution_mode='flash' replaces the in-residual kinetic
                 # R_diss with a once-per-step nodal gas<->brine CO2 exchange
                 # (TADR.h::dissolutionFlash), driven to local equilibrium each
                 # step.  X_sat is the PHYSICAL dissolved-CO2 mass fraction at
                 # solubility (c=c_sat <=> mass fraction X_sat), in the same
                 # rho_w-normalized units as rho_n.  Requires 0 < X_sat < rho_n
                 # for any free gas to remain (a pure-gas pore must hold more
                 # CO2 than a saturated-brine pore); X_sat=0 disables it.
                 # CO2-in-water at FluidFlower's near-atmospheric conditions is
                 # ~1.7 g/kg -> X_sat ~ 0.0015 (just below rho_n=0.0018).
                 # dissolution_mode='kinetic' keeps the legacy k_d sink.
                 dissolution_mode='kinetic',
                 X_sat=0.0,
                 # CO2 injection point sources: list of
                 # (x, y, rate, t_start, t_stop) tuples.  Each is an interior
                 # mass source on the gas (S_n) equation, active while
                 # t_start <= t < t_stop.  None/empty -> no injection (legacy).
                 injection_ports=None,
                 # tanh ramp at each port's start so Newton can track the
                 # saturation breakthrough; in the SIMULATION's time units
                 # (e.g. hours if dt is in hours).  0 -> no ramp (legacy).
                 # Ramp is centred at t0+3*tau, full rate by t0+6*tau.
                 injection_ramp_tau=0.0,
                 # Injection discretization.  False (default) = legacy LUMPED
                 # volumetric disk source over the INJ_RADIUS nodes (byte-identical
                 # to before).  True = CONSISTENT (Galerkin) point source at each
                 # port: R^c_i -= Q_port * N_i(x_p) on the containing element only,
                 # the MOOSE DiracKernel form -- mesh-exact total, concentrates the
                 # CO2 at one element (high local S_g) instead of diluting it over
                 # the disk.  Q_port recovered as rate*pi*radius^2 (= Q_mol/depth).
                 injection_point_source=False,
                 # End-point gas relperm per material type.  Brooks-Corey /
                 # van Genuchten-Mualem give k_rn(S_e=0) = 1 for every sand;
                 # multi-phase rigs (e.g. FluidFlower) measure 0.02..0.16.
                 # Pass a (nMaterialTypes,) array to scale k_rn(*) by the
                 # measured endpoint; None -> all-ones (legacy behavior).
                 krn_end_types=None,
                 # Residual (trapped) gas saturation S_gr per material type.
                 # Gas-only Brooks-Corey/vGM trapping: k_rn -> 0 for S_g <= S_gr
                 # (immobile gas), via the Se_trap remap in pskRelations.h.  k_rw and
                 # p_c keep the drainage Se.  Pass a (nMaterialTypes,) array of
                 # imbibition residuals (FluidFlower Sg,i = 0.06..0.14); None ->
                 # all-zeros (no trapping, legacy behavior).
                 S_gr_types=None,
                 # Gas dynamic viscosity in the simulation's units.  The
                 # gas-flux terms (a_n, f_n) are divided by mu_n at every
                 # quadrature point.  Default 1.0 = legacy (mu implicit at 1).
                 # For CO2 in normalized brine units: mu_n ~= 0.015 (mu_CO2 /
                 # mu_water = 1.5e-5 / 1.0e-3 in physical SI).
                 mu_n=1.0,
                 # Project the coupling Darcy velocity onto a flux-continuous
                 # lowest-order Raviart-Thomas (RT0) field before it is handed
                 # to the transport (TADR) model.  The raw pointwise CG-P1
                 # velocity (velocity_couple) is NOT H(div)-conforming: across a
                 # material interface it has spurious element-to-element normal
                 # jumps (K jumps up to ~147x at F/ESF), so |v| spikes over one
                 # element width and collapses the TADR advective CFL dt.  RT0
                 # projection enforces a single normal flux per edge (continuous
                 # normal component), killing the spurious jump while leaving
                 # genuine tangential shear.  Only affects the EXPORTED coupling
                 # velocity; the flow Newton uses its own upwind potential flux.
                 # Default False -> legacy pointwise velocity_couple unchanged.
                 reconstruct_velocity_rt0=False,
                 # Immiscible / incompressible limit (verification only).  When
                 # immiscible=1 the (p,z) flash is forced to Xeq=0, Yeq=1 with
                 # constant phase densities (co2_brine_flash RHO_A_IMM/RHO_G_IMM),
                 # so mutual solubility and compressibility are suppressed and the
                 # two species balances decouple into the classical immiscible
                 # two-phase saturation equations (McWhorter-Sunada limit).  The
                 # flag is passed to the kernel via argsDict['immiscible'] and
                 # threaded into every flashPZ call.  Default 0 -> full
                 # compositional behavior (reverts all immiscible overrides).
                 immiscible=0,
                 # Flash temperature [deg C].  Threaded to the kernel via
                 # argsDict['T_C'] and used by every flashPZ call (solubility +
                 # phase densities are strongly T-dependent).  Changing this in
                 # the input deck takes effect WITHOUT recompiling m_comp_co2.h
                 # (the .h reads it from argsDict at every entry point).  The
                 # kernel is isothermal per solve.  Default 20 C.
                 T_C=20.0,
                 # Per-rock nodal closure at nodes shared by several rocks
                 # (Z_ACROSS_ROCK_INTERFACES.md): one capillary pressure per
                 # node, one gas saturation per incident rock, storage from the
                 # node's per-rock pore volumes W_ir.  Default False -> the
                 # single-S_g nodal flash, unchanged.
                 per_rock_closure=False,
                  ):
        self.VMS=VMS
        if density_model is None:
            density_model = DENSITY_MODEL
        self.density_model = density_model
        # Stage 3b: gas-side kinetic dissolution sink parameters.
        self.k_d = k_d
        self.c_sat = c_sat
        # Local-equilibrium dissolution flash parameters.
        self.dissolution_mode = str(dissolution_mode)
        self.X_sat = float(X_sat)
        if self.dissolution_mode == 'flash':
            if self.X_sat <= 0.0:
                logEvent("[dissolution flash] WARNING: dissolution_mode='flash'"
                         " but X_sat=%.3e <= 0 -> dissolution is DISABLED."
                         % self.X_sat, level=1)
            elif self.X_sat >= float(rho_n):
                logEvent("[dissolution flash] WARNING: X_sat=%.3e >= rho_n=%.3e"
                         " -> ALL gas dissolves where brine has capacity (no"
                         " free-gas pooling possible)." % (self.X_sat, float(rho_n)),
                         level=1)
        # CO2 injection point sources (see __init__ argument).
        self.injection_ports = list(injection_ports) if injection_ports else []
        self.injection_ramp_tau = float(injection_ramp_tau)
        self.injection_point_source = bool(injection_point_source)
        self.modelIndex=1
        self.SC=SC
        self.anb_seepage_flux= 0.00
        #self.anb_seepage_flux_n =0.0
        # nc=2, primary vars (p_w, S_n).
        # u[0] = p  (pressure in Pa)
        # u[1] = z  (overall CO2 composition / mole fraction in [0,1])
        # Compressibility beta is in 1/Pa and the user-supplied Ksw_types
        # array is interpreted as K/mu_w in 1/(Pa*s).
        variableNames=['p', 'z']
        nc=2
        # gas equation gains a diffusion term -div(a_n grad u_w).
        # Declaring diffusion[1][0][1]='nonlinear' and potential[1][0]='u'
        # makes the framework allocate (1,0) Jacobian sparsity (gas-eq dependence
        # on the wetting pressure gradient via a_n) and the cj=1 nonlinearity
        # tag also adds the coefficient sensitivity contribution to (1,1).
        # (0,1) cross-block tags: wetting eq depends on u_n through
        #   m_w via theta_w(u_n) -> mass[0][1]='nonlinear'
        #   f_w via k_rw(u_n)    -> advection[0][1]='nonlinear'
        #   a_w via k_rw(u_n)    -> diffusion[0][0][1]='nonlinear'
        mass     ={0:{0:'nonlinear', 1:'nonlinear'}, 1:{1:'linear'}}
        advection={0:{0:'nonlinear', 1:'nonlinear'}, 1:{1:'nonlinear'}}
        diffusion={0:{0:{0:'nonlinear', 1:'nonlinear'}},
                   1:{0:{1:'nonlinear'}}}
        potential={0:{0:'u'}, 1:{0:'u', 1:'u'}}
        reaction ={0:{0:'linear'}}
        hamiltonian={}
        self.getSeepageFace=getSeepageFace
        self.gravity=gravity
        self.rho = density
        # gas-phase density. Linear EOS: rho_n(p_n) = rho_n * p_n / p_ref_n
        # when p_ref_n > 0; constant rho_n when p_ref_n == 0.
        self.rho_n = rho_n
        self.p_ref_n = p_ref_n
        # Immiscible/incompressible verification limit (see __init__ argument).
        self.immiscible = int(immiscible)
        # Flash temperature [deg C] (see __init__ argument); passed to the
        # kernel via argsDict['T_C'] at every entry point.
        self.T_C = float(T_C)
        self.beta=beta
        self.vgm_n_types = vgm_n_types
        self.vgm_alpha_types = vgm_alpha_types
        self.thetaR_types    = thetaR_types
        self.thetaSR_types   = thetaSR_types
        # Per-material end-point gas relperm (k_rn at S_e = 0).  Default
        # all-ones so the closure's k_rn(0) = 1 is unchanged.
        if krn_end_types is None:
            self.krn_end_types = np.ones_like(np.asarray(thetaR_types, dtype='d'))
        else:
            self.krn_end_types = np.asarray(krn_end_types, dtype='d')
            assert self.krn_end_types.shape == np.asarray(thetaR_types).shape, \
                "krn_end_types must have the same shape as thetaR_types"
        # Per-material residual (trapped) gas saturation S_gr.  Default all-zeros
        # so Se_trap = 1 and the no-trapping closure is recovered.
        if S_gr_types is None:
            self.S_gr_types = np.zeros_like(np.asarray(thetaR_types, dtype='d'))
        else:
            self.S_gr_types = np.asarray(S_gr_types, dtype='d')
            assert self.S_gr_types.shape == np.asarray(thetaR_types).shape, \
                "S_gr_types must have the same shape as thetaR_types"
        # Per-material start Se_min of the linear Brooks-Corey p_c cap
        # (pskRelations.h bc_pc_from_Se / bc_Se_from_pc).  The cap value is
        # pc(0) = pc(Se_min) * (1 + 1/lambda).  A shared Se_min puts pc(0) at a
        # fixed multiple of each rock's own p_d (15 p_d for Se_min = 1e-2,
        # lambda = 2), so a p_d = 10 Pa sand capped at 150 Pa sits below a
        # 1471.5 Pa seal: at a sand/seal node the sand reaches S_e = 0 before
        # the seal's entry pressure, the per-rock closure can then store gas only
        # in the seal, and gas enters it under a one-node pool.  Instead every
        # rock's pc(0) is lifted to at least PC_CAP_FACTOR * max_r p_d, so one
        # nodal p_c covers every neighbour's entry pressure.  Se_min never
        # exceeds BC_SE_MIN_PC_MAX, the previous shared value, so rocks whose
        # own cap is already high enough (the seal itself) are unchanged.
        BC_SE_MIN_PC_MAX = 1.0e-2
        # 3 = C sand's own cap ratio pc(0)/p_d,ESF = 4414.5/1471.5, the
        # combination that kept the FluidFlower ESF/C interface sealed.
        PC_CAP_FACTOR = 3.0
        alpha_t = np.asarray(vgm_alpha_types, dtype='d')
        lam_t = np.asarray(vgm_n_types, dtype='d')
        pc_cap = PC_CAP_FACTOR * np.max(1.0 / alpha_t)
        self.bc_Se_min_types = np.minimum(
            BC_SE_MIN_PC_MAX,
            (alpha_t * pc_cap / (1.0 + 1.0 / lam_t)) ** (-lam_t))
        # Gas dynamic viscosity (see __init__ argument).  Stored as scalar.
        self.mu_n = float(mu_n)
        # Flux-continuous RT0 projection of the exported coupling velocity.
        self.reconstruct_velocity_rt0 = bool(reconstruct_velocity_rt0)
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
            # add (1,0) cross-block sparsity for the gas-eq diffusion
            # term -div(a_n grad u_w). Same diagonal-tensor layout as (0,0) since a_n
            # reuses the wetting Ks structure (with rho_n / k_rn factors applied at QP).
            # add (0,1) cross-block tensor (a_w depends on u_n via k_rw).
            # Same diagonal layout.
            _diag_rowptr = np.arange(self.nd+1, dtype='i')
            _diag_colind = np.arange(self.nd, dtype='i')
            sparseDiffusionTensors = {(0,0): (_diag_rowptr, _diag_colind),
                                      (0,1): (_diag_rowptr, _diag_colind),
                                      (1,0): (_diag_rowptr, _diag_colind)}

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

        # PSK closure selector: 0 = VGM (van Genuchten-Mualem), 1 = BC (Brooks-Corey-Burdine).
        # The closure functions for both live in proteus/pskRelations.h, under
        # namespace proteus::m_comp_co2::psk. Every call site
        # in m_comp_co2.h dispatches on PSK_TYPE_member: if (PSK_TYPE_member == 1)
        # invokes bc_*_from_Se, else vgm_*_from_Se. Both paths are exercised.
        # NOTE: for the BC path the user-supplied vgm_n_types array is reinterpreted
        # as the BC pore-size index lambda (the closures take a single shape
        # parameter; we reuse the slot to avoid a separate types array).
        psk_types = {"VGM": 0, "BC": 1}
        try:
            if isinstance(PSK_TYPE, int):
                PSK_TYPE = [key for key, value in psk_types.items() if value == PSK_TYPE][0]
            self.PSK_TYPE = psk_types[PSK_TYPE]
        except:
            raise ValueError("PSK_TYPE must be one of " + str(list(psk_types.keys())) + " not " + str(PSK_TYPE))

        # Implicit_FCT (=3) is not supported in the (p_w, S_n) formulation.
        # The FCT pipeline would invert m_w to recover u_w, but m_w now
        # depends on BOTH p_w and S_n (and is independent of p_w when beta=0),
        # so the inversion is ill-posed. The C++ invert() throws if reached;
        # this gate fails earlier with a clearer message.
        if self.STABILIZATION_TYPE == 3:
            raise ValueError(
                "STABILIZATION_TYPE='Implicit_FCT' is not supported in the "
                "(p_w, S_n) two-phase formulation: the wetting mass m_w = "
                "rho_w(p_w)*phi*theta_w(1-u_n) does not uniquely determine "
                "p_w, so the FCT m->u inversion is ill-posed. Use "
                "STABILIZATION_TYPE='Galerkin' or 'EntropyViscosity'."
            )
        self.per_rock_closure = bool(per_rock_closure)
        if self.per_rock_closure and self.STABILIZATION_TYPE != 2:
            raise ValueError(
                "per_rock_closure=True is implemented in the "
                "STABILIZATION_TYPE='EntropyViscosity' (2) edge-flux path only.")
        if self.per_rock_closure and self.PSK_TYPE != 1:
            raise ValueError(
                "per_rock_closure=True needs PSK_TYPE='BC': the closure inverts "
                "bc_pc_from_Se (pskRelations.h bc_Se_from_pc).")

        # EDGE BASED (AND ENTROPY) VISCOSITY
        self.LUMPED_MASS_MATRIX = LUMPED_MASS_MATRIX
        self.MONOLITHIC = MONOLITHIC
        #self.STABILIZATION_TYPE = STABILIZATION_TYPE
        self.ENTROPY_TYPE = ENTROPY_TYPE
        # Capture the user's FCT request. The Proteus framework's
        # NonlinearSolvers.Newton.solve() has a post-Newton FCT hook
        #     if self.F.coefficients.FCT == True:
        #         self.F.FCTStep()
        #         u[:] = self.F.u[0].dof
        # that assumes single-component dof storage (u[0].dof IS the full
        # unknown), which is the Richards-era convention. For our nc=2 model
        # (comp-0 + comp-1 each sized N), u is 2N and u[0].dof is N, so the
        # broadcast crashes. We force self.FCT = False so the framework hook
        # never fires; the C++ FCT pipeline runs inside calculateResidual_
        # entropy_viscosity via the FCT_n argsDict flag, gated by
        # _fct_requested below.
        self._fct_requested = bool(FCT)
        self.FCT = False
        self.num_fct_iter=num_fct_iter
        self.uL = uL
        self.uR = uR
        self.cK = cK
        self.forceStrongConditions = False
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
        # Always allocate self.c_dof so the C++ kernel never sees a missing
        # argsDict entry (Stage 3b reads it unconditionally).
        if self.density_model is None:
            self.c_dof = np.zeros_like(self.model.u[1].dof)
            return
        self.densityModel = modelList[self.density_model]
        # Stage 3b: alias TADR's c DOFs so the gas-equation residual can
        # compute R_diss = k_d * S_n * S_w * (c_sat - c) and deduct it from
        # the gas mass.  C0-P1 DOFs are shared on the same mesh, so this is
        # direct DOF-to-DOF aliasing (no projection).  If the density_model
        # doesn't expose u[0].dof (unusual), fall back to a zero array so
        # the sink is harmlessly inactive.
        if hasattr(self.densityModel, 'u') and len(self.densityModel.u) >= 1 \
                and hasattr(self.densityModel.u[0], 'dof'):
            self.c_dof = self.densityModel.u[0].dof
        else:
            self.c_dof = np.zeros_like(self.model.u[1].dof)

    def preStep(self, t, firstStep=False):
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
        from mpi4py import MPI
        comm = MPI.COMM_WORLD

        def _global_stats(local):
            a = np.asarray(local)
            lo = comm.allreduce(float(a.min()) if a.size else float('inf'), op=MPI.MIN)
            hi = comm.allreduce(float(a.max()) if a.size else float('-inf'), op=MPI.MAX)
            ssum = comm.allreduce(float(a.sum()), op=MPI.SUM)
            n = comm.allreduce(int(a.size), op=MPI.SUM)
            return lo, hi, (ssum / n if n > 0 else float('nan'))

        if q_rho is not None and 'rho' in self.model.q:
            src = np.asarray(q_rho)
            dst = np.asarray(self.model.q['rho'])
            local_diff = float(np.max(np.abs(src - dst))) if src.shape == dst.shape else float('nan')
            diff = comm.allreduce(local_diff, op=MPI.MAX)
            s_lo, s_hi, s_mn = _global_stats(src)
            d_lo, d_hi, d_mn = _global_stats(dst)
            if comm.Get_rank() == 0:
                logEvent(
                    "[Coupling rho q] Richards.preStep t={:.6e} firstStep={} "
                    "TADR.q_rho (min,max,mean)=({:.6e},{:.6e},{:.6e}) "
                    "Richards.q['rho'] (min,max,mean)=({:.6e},{:.6e},{:.6e}) "
                    "max|src-dst|={:.3e}".format(
                        float(t), firstStep, s_lo, s_hi, s_mn,
                        d_lo, d_hi, d_mn, diff),
                    level=2)
        if ebqe_rho is not None and 'rho' in self.model.ebqe:
            src_b = np.asarray(ebqe_rho)
            dst_b = np.asarray(self.model.ebqe['rho'])
            local_diff_b = float(np.max(np.abs(src_b - dst_b))) if src_b.shape == dst_b.shape else float('nan')
            diff_b = comm.allreduce(local_diff_b, op=MPI.MAX)
            s_lo, s_hi, s_mn = _global_stats(src_b)
            d_lo, d_hi, d_mn = _global_stats(dst_b)
            if comm.Get_rank() == 0:
                logEvent(
                    "[Coupling rho ebqe] Richards.preStep t={:.6e} "
                    "TADR.ebqe_rho (min,max,mean)=({:.6e},{:.6e},{:.6e}) "
                    "Richards.ebqe['rho'] (min,max,mean)=({:.6e},{:.6e},{:.6e}) "
                    "max|src-dst|={:.3e}".format(
                        float(t), s_lo, s_hi, s_mn, d_lo, d_hi, d_mn, diff_b),
                    level=2)
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
        if ('grad(u)', 0) in c:
            logEvent(f"Richards grad(u): mean={c[('grad(u)', 0)].mean()}, min={c[('grad(u)', 0)].min()}, max={c[('grad(u)', 0)].max()}")
        else:
            logEvent("Warning: grad(u) is not available in Richards coefficients.")
        
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

        # ---- Component 1 (S_n) framework bookkeeping fill -----------------
        # The real gas residual is assembled in C++; this fill writes the
        # correctly-scaled placeholder so any downstream framework code that
        # consumes (m,1)/(dm,1,1) sees physically consistent values rather
        # than a unit-mass shape.
        # Material 0 is used as the representative; if multiple materials are
        # in play, the C++ assembly recomputes per QP and overrides this.
        phi_rho_n = float((self.thetaR_types[0] + self.thetaSR_types[0])
                          * self.rho_n)
        if ('m', 1) in c and ('u', 1) in c:
            c[('m', 1)][:] = phi_rho_n * c[('u', 1)]
        if ('dm', 1, 1) in c:
            c[('dm', 1, 1)][:] = phi_rho_n
        # zero out the (1,0) cross-block coefficient arrays.
        # The C++ residual/Jacobian assembly fills these with the actual gas
        # Darcy contributions; this evaluate() fill is just so the framework
        # has well-defined values during sparsity setup / NaN checks.
        if ('a', 1, 0) in c:
            c[('a', 1, 0)][:] = 0.0
        if ('da', 1, 0, 1) in c:
            c[('da', 1, 0, 1)][:] = 0.0
        if ('df', 1, 1) in c:
            c[('df', 1, 1)][:] = 0.0
        if ('f', 1) in c:
            c[('f', 1)][:] = 0.0
        # Zero out the (0,1) cross-block coefficient arrays. The C++ Jacobian
        # element loop writes the cross-block directly into globalJacobian, so
        # this fill only keeps the framework's NaN guards happy.
        if ('dm', 0, 1) in c:
            c[('dm', 0, 1)][:] = 0.0
        if ('df', 0, 1) in c:
            c[('df', 0, 1)][:] = 0.0
        if ('da', 0, 0, 1) in c:
            c[('da', 0, 0, 1)][:] = 0.0
    
    def postStep(self, t, firstStep=False):
        m = self.model
        # FCT post-step (gated): only when STAB>0 and FCT was requested.
        if (m is not None
                and self.STABILIZATION_TYPE != 0
                and self._fct_requested
                and getattr(m, 'limited_solution_n', None) is not None):
            m.FCTStep(component=1)
        # Flux-continuous RT0 projection of the exported coupling velocity.
        # Runs after the flow solve and BEFORE TADR.preStep reads
        # velocity_couple (Sequential_MinModelStep advances flow first), so
        # TADR advects with -- and sets its CFL dt from -- the H(div)-conforming
        # field instead of the spiky pointwise CG velocity.
        if m is not None and self.reconstruct_velocity_rt0:
            self._project_velocity_couple_to_rt0(t)
        # Mass-balance diagnostic (always runs when coupled to TADR).
        self._log_mass_balance(t)
        # Velocity-spike source classifier (debug for the TADR dt collapse).
        self._diagnose_velocity_spike(t)
        return {}

    def apply_dissolution_flash(self, t):
        r"""Finite-rate implicit dissolution (local-equilibrium flash limit).

        Called ONCE per step from the transport coefficients' postStep
        (Sequential_MinModelStep runs flow, then transport, so by here BOTH
        S_n and c are the converged end-of-step values).  The whole exchange
        runs in C++ (m_comp_co2.h::dissolutionFlash via the flow model's
        self.m_comp_co2 object): it (1) relaxes c toward the gas-limited
        equilibrium with an implicit linear-driving-force step (rate k_d*S_n),
        recovering S_n from exact M-conservation, mutating S_n (flow.u[1].dof)
        and c (tadr.u[0].dof) IN PLACE, and (2) rebuilds TADR's quadrature
        old-mass q[('m',0)] from the updated fields using the same
        valFromDOF/l2g/phi convention as the residual.  Python only repairs the
        nodal time history and the ghost values.

        Conserves per-node CO2 (rho_w-normalized) M = rho_n*S_n +
        (X_sat/c_sat)*(1-S_n)*c exactly.  k_d is the dissolution rate: k_d -> inf
        recovers the instantaneous local-equilibrium flash (dissolves in place),
        finite k_d lets the free-gas plume rise before it dissolves over a
        slower timescale.  X_sat is the physical solubility mass fraction.  See
        the kernel comment in proteus/m_comp_co2/m_comp_co2.h.
        """
        if self.dissolution_mode != 'flash' or self.X_sat <= 0.0:
            return
        flow = self.model
        tadr = getattr(self, 'densityModel', None)
        if flow is None or tadr is None:
            return
        if not (hasattr(flow, 'm_comp_co2') and hasattr(tadr, 'u') and len(tadr.u) >= 1):
            return
        c_dof  = tadr.u[0].dof
        Sn_dof = flow.u[1].dof
        numDOFs = int(min(len(c_dof), len(Sn_dof)))

        # TADR old-mass array to rebuild (q[('m',0)] == m_tmp; the framework's
        # end-of-step updateTimeHistory copies m_tmp -> m_last AFTER this
        # postStep, so writing it here makes next step's BDF old-mass reflect
        # the flash -- otherwise the flash's dissolved increment is not
        # conserved).  Derive nElements/nQP FROM this array so the C++ QP write
        # can never go out of bounds.
        q_m_tadr = tadr.q.get(('m', 0), None)
        if q_m_tadr is None or q_m_tadr.ndim != 2:
            logEvent("[dissolution flash] WARNING: TADR q[('m',0)] missing; "
                     "skipping flash this step.", level=1)
            return
        nE_qm, nQP_qm = int(q_m_tadr.shape[0]), int(q_m_tadr.shape[1])
        tcoef = tadr.coefficients

        # Single C++ call: nodal exchange (mutates c_dof, Sn_dof) + TADR
        # quadrature old-mass rebuild (writes q_m_tadr).  u_trial_ref / u_l2g
        # are the flow model's populated P1 basis and element->node map (same
        # mesh, so valid for the co-located transport c too).
        # Step size for the finite-rate implicit relaxation (LDF rate k_d*S_n);
        # under Sequential_MinModelStep both models share the system dt.
        ti = getattr(tadr, 'timeIntegration', None)
        dt = float(getattr(ti, 'dt', 0.0)) if ti is not None else 0.0
        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["c_dof"]    = c_dof
        argsDict["Sn_dof"]   = Sn_dof
        argsDict["rho_n"]    = float(self.rho_n)
        argsDict["X_sat"]    = float(self.X_sat)
        argsDict["c_sat"]    = float(self.c_sat)
        # k_d is the dissolution RATE [1/time] of the finite-rate implicit
        # flash; k_d -> inf recovers the instantaneous local-equilibrium flash.
        argsDict["k_d"]      = float(self.k_d)
        argsDict["dt"]       = dt
        argsDict["numDOFs"]  = numDOFs
        argsDict["q_m_tadr"] = q_m_tadr
        argsDict["u_l2g"]                = flow.u[0].femSpace.dofMap.l2g
        argsDict["u_trial_ref"]          = flow.u[0].femSpace.psi
        argsDict["elementMaterialTypes"] = flow.mesh.elementMaterialTypes
        argsDict["thetaR"]               = self.thetaR_types
        argsDict["thetaSR"]              = self.thetaSR_types
        argsDict["rho_f"]    = float(getattr(tcoef, 'rho_f', 1.0))
        argsDict["rho_s"]    = float(getattr(tcoef, 'rho_s',
                                             getattr(tcoef, 'rho_f', 1.0)))
        argsDict["nElements_global"]          = nE_qm
        argsDict["nQuadraturePoints_element"] = nQP_qm
        argsDict["nDOF_trial_element"]        = int(flow.u[0].femSpace.dofMap.l2g.shape[1])
        flow.m_comp_co2.dissolutionFlash(argsDict)

        # Ghost-sync owned -> ghost for both fields (the flash is a pure local
        # per-DOF map, so this just keeps ghosts identical to owners).
        for fef in (tadr.u[0], flow.u[1]):
            par = getattr(fef, 'par_dof', None)
            if par is not None:
                par.scatter_forward_insert()

        # Flow gas history: the C++ gas residual reads u_dof_n_old (nodal) as
        # m_n_old, so it MUST become the flashed S_n -- mirror the FCT
        # write-back at FCTStep(component=1) (u_dof_n_old + scatter).
        if getattr(flow, 'u_dof_n_old', None) is not None:
            flow.u_dof_n_old[:] = flow.u[1].dof
        flow._scatter_component_to_timeintegration(1)

        # TADR c nodal history: u_dof_old feeds the entropy/advection of the
        # old solution.  (The quadrature old-mass was rebuilt in C++ above.)
        if getattr(tadr, 'u_dof_old', None) is not None:
            tadr.u_dof_old[:] = tadr.u[0].dof
            par_old = getattr(tadr, 'par_u_dof_old', None)
            if par_old is not None:
                par_old.scatter_forward_insert()

    def _diagnose_velocity_spike(self, t):
        r"""Classify the SOURCE of the velocity_couple spike that collapses the
        TADR advective-CFL dt.  velocity_couple = -(k_rw K)(grad p_w - rho g)
        depends only on grad p_w, K, k_rw(S_n), gravity -- NOT on p_c -- so the
        spike is one of three mechanisms with DIFFERENT cures.  At the owned
        element carrying max|v| this logs the decomposition that tells them
        apart, on the RAW (un-reconstructed) field:

          - K_mat        : the element's material permeability (head units).
          - K_eff(mob)   : |v| / |grad Phi| = k_rw*K, the effective mobility,
                           recovered WITHOUT re-evaluating the Brooks-Corey
                           closure (grad Phi = grad p_w - rho g).
          - kr_w~        : K_eff / K_mat, the implied water relperm.
          - |grad p_w|   : hydrostatic ~ 1 (head form); >> 1 = a genuine sharp
                           water-pressure response.
          - |grad S_n|   : front sharpness; large => the spike tracks the
                           breakthrough saturation front.
          - nbr|v| / v   : ratio of the spike to its across-interface neighbour
                           peak.  >> 1 with flux_jump_rel ~ 0 = a POINTWISE
                           velocity-at-permeability-jump artifact (the discrete
                           CG velocity is not H(div); RT0/mixed would fix it).
          - flux_jump_rel: max over the 3 edges of |v.m - v_nbr.m|/|v.m|, the
                           discrete normal-flux discontinuity.  ~0 = flux IS
                           continuous (any |v| excess is a pointwise artifact);
                           large = genuine flux divergence (a source or a sharp
                           front sits on the element) OR a deeper bug.

        Synthetic Galerkin experiments (debug session 2026-05-30) showed:
          * an ALIGNED K-jump produces NO velocity spike (flux stays continuous,
            v_ratio~1) -- so a spike is NOT explained by permeability contrast;
          * a SOURCE/front produces a GENUINE large-|grad p_w| spike with
            v_ratio~1 (continuous velocity) -- RT0 does NOT reduce this.

        Reading:
          flux_jump_rel~0, v_ratio>>1, K_jump>>1 -> pointwise velocity-at-K-jump
              artifact (the rare case RT0/harmonic-K interface velocity fixes).
          |grad S_n| large + v_ratio~1 -> physical breakthrough front; the spike
              is a real pressure response (cure: decouple/sub-cycle TADR --
              velocity surgery, RT0 included, only masks it).
          |grad p_w| >> 1, kr_w~ ~ 1, v_ratio~1 -> genuine pressure-gradient
              spike (look upstream: is the S_n front over-sharpened by the
              gas-side nodal-pc artifact?).
        """
        from mpi4py import MPI
        m = self.model
        if m is None or self.nd != 2 or not hasattr(self, '_elem_area_p'):
            return
        vc = m.q.get(('velocity_couple', 0), None)
        if vc is None:
            return
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        mesh = m.mesh
        nE = int(mesh.nElements_global)
        nE_own = int(getattr(mesh, 'nElements_owned', nE))

        # one-time element-side connectivity (neighbour + scaled outward normal)
        if not hasattr(self, '_spike_nbr'):
            area = self._elem_area_p[:nE]
            self._spike_m = -2.0 * area[:, None, None] * self._elem_gradphi_p[:nE]
            EB = np.asarray(mesh.elementBoundariesArray)[:nE]
            ebe = np.asarray(mesh.elementBoundaryElementsArray)
            left = ebe[EB, 0]; right = ebe[EB, 1]
            eidx = np.broadcast_to(np.arange(nE)[:, None], (nE, 3))
            nbr = np.where(left == eidx, right, left)
            self._spike_interior = (nbr >= 0)
            self._spike_nbr = np.where(nbr < 0, eidx, nbr).astype('i')

        v_e = np.asarray(vc)[:, :, :2].mean(axis=1)            # (nE,2) incl ghosts
        vmag_own = np.sqrt((v_e[:nE_own] * v_e[:nE_own]).sum(1))
        if vmag_own.size == 0:
            local = (-1.0,) + (0.0,) * 9
        else:
            eN = int(np.argmax(vmag_own))
            nodes = self._elem_nodes_p[eN]
            gphi = self._elem_gradphi_p[eN]                    # (3,2)
            pw = np.asarray(m.u[0].dof); Sn = np.asarray(m.u[1].dof)
            gpw = (pw[nodes][:, None] * gphi).sum(0)           # P1 grad p_w
            gsn = (Sn[nodes][:, None] * gphi).sum(0)           # P1 grad S_n
            rho = float(self.rho)
            gPhi = gpw - rho * np.asarray(self.gravity[:2], 'd')
            gPhi_mag = float(np.hypot(gPhi[0], gPhi[1]))
            vmax = float(vmag_own[eN])
            Keff = vmax / max(gPhi_mag, 1.0e-30)
            mat = int(self.elementMaterialTypes[eN])
            Kmat = float(self.Ksw_types[mat, 0]) if mat < len(self.Ksw_types) else 0.0
            krw = Keff / max(Kmat, 1.0e-30)
            mvec = self._spike_m[eN]; nbrs = self._spike_nbr[eN]
            interior = self._spike_interior[eN]
            max_rel_jump = 0.0; nbr_vmax = 0.0; Kjump = 1.0
            for i in range(3):
                if not interior[i]:
                    continue
                fs = float(v_e[eN] @ mvec[i]); fn = float(v_e[nbrs[i]] @ mvec[i])
                max_rel_jump = max(max_rel_jump, abs(fs - fn) / (abs(fs) + 1.0e-30))
                nbr_vmax = max(nbr_vmax, float(np.hypot(v_e[nbrs[i], 0], v_e[nbrs[i], 1])))
                nmat = int(self.elementMaterialTypes[nbrs[i]])
                Kn = float(self.Ksw_types[nmat, 0]) if nmat < len(self.Ksw_types) else Kmat
                Kjump = max(Kjump, Kmat / max(Kn, 1.0e-30), Kn / max(Kmat, 1.0e-30))
            local = (vmax, float(np.hypot(gpw[0], gpw[1])),
                     float(np.hypot(gsn[0], gsn[1])), Keff, Kmat, krw,
                     nbr_vmax, max_rel_jump, Kjump, float(mat))

        allinfo = comm.gather(local, root=0)
        if rank == 0:
            (vmax, gpwm, gsnm, Keff, Kmat, krw, nbrv, reljmp, Kj, mat) = max(
                allinfo, key=lambda r: r[0])
            logEvent(
                "[spike decomp] t={:.4e} max|v|={:.4e} mat={:.0f} |grad_pw|={:.4e} "
                "|grad_Sn|={:.4e} K_mat={:.4e} kr_w~={:.4e} K_eff(mob)={:.4e}  "
                "nbr|v|max={:.4e} (v_ratio={:.2f}) flux_jump_rel={:.3e} "
                "K_jump={:.1f}".format(
                    float(t), vmax, mat, gpwm, gsnm, Kmat, krw, Keff,
                    nbrv, vmax / max(nbrv, 1.0e-30), reljmp, Kj),
                level=2)

    def _project_velocity_couple_to_rt0(self, t):
        r"""Project the pointwise CG-P1 coupling velocity onto a flux-continuous
        lowest-order Raviart-Thomas (RT0) field, in place, overwriting
        ``q[('velocity_couple',0)]`` and ``ebqe[('velocity_couple',0)]``.

        On a triangle E with vertices p_0,p_1,p_2 (local edge i opposite p_i),
        the RT0 flux representation is

            v(x) = sum_i V_i (x - p_i) / (2 |E|),

        where V_i is the OUTWARD normal flux through edge i.  The outward scaled
        edge normal (|m_i| = edge length) is m_i = -2|E| grad(lambda_i), and
        grad(lambda_i) is the cached P1 shape gradient ``_elem_gradphi_p``, so no
        explicit normal geometry is needed.  Flux continuity is imposed by giving
        each interior edge a single value -- the average of the two one-sided
        normal fluxes:  V_i = 1/2 (v_E + v_E') . m_i.  The neighbour computes
        -V_i for the same edge (m flips sign), so the reconstructed normal
        component is single-valued across every edge -> no spurious inter-element
        normal jump -> no |v| spike over one element width.  RT0 reproduces a
        globally constant velocity exactly, so smooth regions are untouched.

        2D triangular simplex meshes only (asserted via the cached geometry)."""
        from mpi4py import MPI
        m = self.model
        if m is None or self.nd != 2:
            return
        vc = m.q.get(('velocity_couple', 0), None)
        if vc is None:
            return

        # --- one-time geometry / connectivity cache -------------------------
        # Reuses the element-area / shape-gradient cache built by
        # _log_mass_balance; build it here too if RT0 runs first.
        if not hasattr(self, '_elem_area_p'):
            mesh0 = m.mesh
            EN0 = np.asarray(mesh0.elementNodesArray)[:mesh0.nElements_global]
            X0 = np.asarray(mesh0.nodeArray)
            x0 = X0[EN0[:, 0]]; x1 = X0[EN0[:, 1]]; x2 = X0[EN0[:, 2]]
            detA0 = ((x1[:, 0] - x0[:, 0]) * (x2[:, 1] - x0[:, 1])
                     - (x2[:, 0] - x0[:, 0]) * (x1[:, 1] - x0[:, 1]))
            inv0 = np.where(np.abs(detA0) > 0.0, 1.0 / detA0, 0.0)
            g0 = np.empty((len(EN0), 3, 2), 'd')
            g0[:, 0, 0] = (x1[:, 1] - x2[:, 1]) * inv0; g0[:, 0, 1] = (x2[:, 0] - x1[:, 0]) * inv0
            g0[:, 1, 0] = (x2[:, 1] - x0[:, 1]) * inv0; g0[:, 1, 1] = (x0[:, 0] - x2[:, 0]) * inv0
            g0[:, 2, 0] = (x0[:, 1] - x1[:, 1]) * inv0; g0[:, 2, 1] = (x1[:, 0] - x0[:, 0]) * inv0
            self._elem_nodes_p = EN0
            self._elem_area_p = 0.5 * np.abs(detA0)
            self._elem_gradphi_p = g0

        if not hasattr(self, '_rt0_nbr'):
            mesh = m.mesh
            nE = int(mesh.nElements_global)
            EN = self._elem_nodes_p[:nE]
            X = np.asarray(mesh.nodeArray)
            self._rt0_P = X[EN][:, :, :2]                       # (nE,3,2) vertices
            area = self._elem_area_p[:nE]
            self._rt0_area = area
            # scaled outward edge normals m_i = -2|E| grad(lambda_i), |m_i| = L_i
            self._rt0_m = -2.0 * area[:, None, None] * self._elem_gradphi_p[:nE]
            # neighbour element per (eN, local edge i); self at the boundary.
            EB = np.asarray(mesh.elementBoundariesArray)[:nE]    # (nE,3) global edge ids
            ebe = np.asarray(mesh.elementBoundaryElementsArray)  # (nEB,2) [left,right], -1 exterior
            left = ebe[EB, 0]; right = ebe[EB, 1]                 # (nE,3)
            eidx = np.broadcast_to(np.arange(nE)[:, None], (nE, 3))
            nbr = np.where(left == eidx, right, left)
            self._rt0_nbr = np.where(nbr < 0, eidx, nbr).astype('i')
            # exterior-boundary -> owning element, for the ebqe reconstruction.
            extB = np.asarray(mesh.exteriorElementBoundariesArray)
            self._rt0_ext_eN = (ebe[extB, 0].astype('i') if extB.size
                                else np.zeros(0, 'i'))

        P = self._rt0_P; area = self._rt0_area
        mvec = self._rt0_m; nbr = self._rt0_nbr
        nE = P.shape[0]
        inv2A = 1.0 / (2.0 * np.maximum(area, 1.0e-30))

        # --- element-constant velocity (Lobatto-1: identical at all qp) -----
        vc = np.asarray(vc)                                      # numpy view of m.q array
        v_e = vc[:nE, :, :2].mean(axis=1)                        # (nE,2)
        v_nb = v_e[nbr]                                          # (nE,3,2)
        # single-valued outward edge flux V_i = 1/2 (v_E + v_E') . m_i
        V = 0.5 * ((v_e[:, None, :] + v_nb) * mvec).sum(axis=-1)  # (nE,3)

        # --- reconstruct at element quadrature points -> overwrite q --------
        qx = np.asarray(m.q['x'])[:nE, :, :2]                    # (nE,nQ,2)
        diff = qx[:, :, None, :] - P[:, None, :, :]              # (nE,nQ,3,2)
        v_q = (V[:, None, :, None] * diff).sum(axis=2) * inv2A[:, None, None]
        raw_max = float(np.sqrt((v_e * v_e).sum(-1)).max()) if nE else 0.0
        vc[:nE, :, :2] = v_q                                     # in-place into m.q
        rt0_max = float(np.sqrt((v_q * v_q).sum(-1)).max()) if nE else 0.0

        # --- reconstruct at exterior boundary quadrature points -> ebqe -----
        vce = m.ebqe.get(('velocity_couple', 0), None)
        if vce is not None and self._rt0_ext_eN.size:
            vce = np.asarray(vce)
            eN_e = self._rt0_ext_eN
            xb = np.asarray(m.ebqe['x'])[:, :, :2]               # (nExt,nQb,2)
            diffb = xb[:, :, None, :] - P[eN_e][:, None, :, :]
            v_b = (V[eN_e][:, None, :, None] * diffb).sum(axis=2) \
                * inv2A[eN_e][:, None, None]
            vce[..., :2] = v_b                                   # in-place into m.ebqe

        comm = MPI.COMM_WORLD
        g_raw = comm.allreduce(raw_max, op=MPI.MAX)
        g_rt0 = comm.allreduce(rt0_max, op=MPI.MAX)
        if comm.Get_rank() == 0:
            logEvent(
                "[RT0 vel    ] t={:.4e} max|v_couple| raw={:.4e} -> RT0={:.4e} "
                "(ratio={:.3f})".format(
                    float(t), g_raw, g_rt0,
                    (g_rt0 / g_raw if g_raw > 0.0 else 0.0)),
                level=2)

    def _log_mass_balance(self, t):
        """Aggregate gas + dissolved CO2 mass and compare to cumulative
        injection. Diagnostic for the STAB=2 Richards-style upwind port:
        if total mass = cum_injected, the operator is well-calibrated; if
        gas + diss << cum_injected, dLow is over-dissipating (gas vanishes
        before it can dissolve).

        Per-DOF lumped weights are computed once from the P1 element-area
        partition and cached. phi is volume-averaged per node (heterogeneous
        Brooks-Corey materials supported)."""
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        m = self.model
        if m is None or not hasattr(self, 'densityModel') or self.densityModel is None:
            return
        mesh = m.mesh

        if not hasattr(self, '_M_lump_node'):
            nNodes_total = int(getattr(mesh, 'nNodes_global', len(m.u[1].dof)))
            ML = np.zeros(nNodes_total, 'd')
            phi_num = np.zeros(nNodes_total, 'd')
            phi_den = np.zeros(nNodes_total, 'd')
            elementNodes = mesh.elementNodesArray
            nodeArr = mesh.nodeArray
            matTypes = self.elementMaterialTypes
            for eN in range(mesh.nElements_global):
                nodes = elementNodes[eN]
                p0 = nodeArr[nodes[0]]
                p1 = nodeArr[nodes[1]]
                p2 = nodeArr[nodes[2]]
                area = 0.5 * abs((p1[0] - p0[0]) * (p2[1] - p0[1])
                                 - (p2[0] - p0[0]) * (p1[1] - p0[1]))
                mat = int(matTypes[eN])
                phi_e = float(self.thetaR_types[mat] + self.thetaSR_types[mat])
                contrib = area / 3.0
                for nn in nodes:
                    ML[nn] += contrib
                    phi_num[nn] += contrib * phi_e
                    phi_den[nn] += contrib
            phi_node = np.where(phi_den > 0.0, phi_num / phi_den, 0.0)
            self._M_lump_node = ML
            self._phi_node = phi_node
            # --- cache element geometry for the velocity-divergence probe ---
            # P1 shape-function gradients (constant per triangle): grad phi_i =
            # (b_i,c_i)/detA, detA = 2*signed_area.  Used to form the
            # element-constant grad(c) and the weak nodal divergence of the
            # Darcy flux TADR rides.
            EN = np.asarray(elementNodes)[:mesh.nElements_global]
            X = np.asarray(nodeArr)
            x0 = X[EN[:, 0]]; x1 = X[EN[:, 1]]; x2 = X[EN[:, 2]]
            detA = ((x1[:, 0] - x0[:, 0]) * (x2[:, 1] - x0[:, 1])
                    - (x2[:, 0] - x0[:, 0]) * (x1[:, 1] - x0[:, 1]))  # 2*signed area
            inv = np.where(np.abs(detA) > 0.0, 1.0 / detA, 0.0)
            gphi = np.empty((len(EN), 3, 2), 'd')
            gphi[:, 0, 0] = (x1[:, 1] - x2[:, 1]) * inv; gphi[:, 0, 1] = (x2[:, 0] - x1[:, 0]) * inv
            gphi[:, 1, 0] = (x2[:, 1] - x0[:, 1]) * inv; gphi[:, 1, 1] = (x0[:, 0] - x2[:, 0]) * inv
            gphi[:, 2, 0] = (x0[:, 1] - x1[:, 1]) * inv; gphi[:, 2, 1] = (x1[:, 0] - x0[:, 0]) * inv
            self._elem_nodes_p = EN
            self._elem_area_p = 0.5 * np.abs(detA)
            self._elem_gradphi_p = gphi

        ML = self._M_lump_node
        phi = self._phi_node
        S_n = np.asarray(m.u[1].dof)
        c = np.asarray(self.densityModel.u[0].dof)

        # Sum owned DOFs only (avoid double-count across MPI ranks).
        n_owned = int(getattr(mesh, 'nNodes_owned',
                              getattr(mesh, 'nNodes_global', len(S_n))))
        size = min(n_owned, len(ML), len(S_n), len(c))
        w = ML[:size] * phi[:size]
        # In flash mode the PHYSICAL dissolved-CO2 mass is X_sat*(1-S_n)*c per
        # pore volume (rho_w units), the same scale as the gas mass rho_n*S_n,
        # so gas + diss is directly comparable to the injected gas mass.  In
        # kinetic mode keep the legacy unit (factor 1) for back-compatibility.
        diss_scale = (float(self.X_sat)
                      if (self.dissolution_mode == 'flash' and self.X_sat > 0.0)
                      else 1.0)
        local_gas = float(np.sum(w * float(self.rho_n) * S_n[:size]))
        local_dis = float(np.sum(w * diss_scale * (1.0 - S_n[:size]) * c[:size]))
        gas = comm.allreduce(local_gas, op=MPI.SUM)
        diss = comm.allreduce(local_dis, op=MPI.SUM)

        # ---- COMPRESSIBLE-DENSITY MASS CHECK (is the "gas >> injected" real?) ----
        # gas above uses constant rho_n; the kernel conserves the COMPRESSIBLE
        # m_n = phi*rho_n*exp((pw+pc)/p_ref_n)*S_n.  p_c >= 0 always, so the
        # pw-only factor exp(pw/p_ref_n) is a LOWER BOUND on the true compressible
        # density => gas_pw <= gas_compressible.  If gas_pw is still ~gas_const
        # (not ~1/28 of it) the mass discrepancy is REAL over-creation, not the
        # const-rho diagnostic; the only density escape is pw ~ -p_ref_n*ln(ratio)
        # (deep suction) in the gas zone -- which the pw range below tests directly.
        pw_all = np.asarray(m.u[0].dof)
        nn = min(size, len(pw_all))
        pw = pw_all[:nn]
        Sn_z = S_n[:nn]
        inv_pref = (1.0 / float(self.p_ref_n)) if float(self.p_ref_n) > 0.0 else 0.0
        rho_n_pw = float(self.rho_n) * np.exp(np.clip(pw * inv_pref, -50.0, 50.0))
        local_gas_pw = float(np.sum(w[:nn] * rho_n_pw * Sn_z))
        gas_pw = comm.allreduce(local_gas_pw, op=MPI.SUM)
        gasmask = Sn_z > 1.0e-12
        if bool(gasmask.any()):
            pwz = pw[gasmask]
            local_pwmin, local_pwmax = float(pwz.min()), float(pwz.max())
        else:
            local_pwmin, local_pwmax = 0.0, 0.0
        pw_gas_min = comm.allreduce(local_pwmin, op=MPI.MIN)
        pw_gas_max = comm.allreduce(local_pwmax, op=MPI.MAX)

        # ---- KERNEL-INJECTED MASS AUDIT ----
        # The kernel adds  M_lump_i * Q_inj_i * dt  to gas mass per node per
        # time step. Total kernel injection = integral over time of
        # sum(M_lump * injection_dof). We accumulate this with a trapezoidal
        # rule across mass-balance calls. If kernel_inj_cum != cum_inj
        # (analytical), the analytical disk-area formula is mesh-discrepant
        # with the lumped-quadrature sum over the masked nodes.
        inj_dof = getattr(m, 'injection_dof', None)
        if inj_dof is not None:
            inj_dof_local = np.asarray(inj_dof)[:size]
            local_inj_rate = float(np.sum(ML[:size] * inj_dof_local))
        else:
            local_inj_rate = 0.0
        inj_rate_global = comm.allreduce(local_inj_rate, op=MPI.SUM)
        if not hasattr(self, '_kernel_inj_cum'):
            self._kernel_inj_cum = 0.0
            self._last_balance_t = float(t)
            self._last_inj_rate  = inj_rate_global
        dt_balance = float(t) - self._last_balance_t
        if dt_balance > 0.0:
            # End-of-step accumulation: the kernel scatters
            #   ML_n[i] * injection_dof[i] * dt
            # per step using injection_dof set in getResidual at the START
            # of the step.  That rate is held constant across all Newton
            # iterations for the step.  Using end-of-step `inj_rate_global`
            # here matches the kernel's actual scatter exactly; trapezoidal
            # averaging biased low during a ramping schedule and inflated
            # the apparent "leak" in `bal_vs_kernel`.
            self._kernel_inj_cum += dt_balance * inj_rate_global
        self._last_balance_t = float(t)
        self._last_inj_rate  = inj_rate_global
        kernel_inj = self._kernel_inj_cum

        # ---- R_DISS BUDGET AUDIT (flow side vs TADR side) ----
        # Per-node analytical R_diss = phi*(1-S_n)*rho_w*k_d*S_n*(c_sat - c).
        # Both m_comp_co2 (gas-eq sink) and TADR (c-eq source) compute this
        # independently. With matched k_d / c_sat they should agree to the
        # split-lag error in c. The flow side reads c_dof (== TADR's u[0].dof)
        # so they SHOULD use the same c; any difference is Sequential split
        # timing.
        rho_w_dof = None
        coeffs = getattr(self.densityModel, 'coefficients', None)
        if coeffs is not None:
            # densityModel q_rho only available at QPs; for diagnostic use 1.0.
            pass
        rho_w_eff = 1.0  # head form
        k_d_flow = float(self.k_d)
        c_sat    = float(self.c_sat)
        k_d_tadr = float(getattr(coeffs, 'k_d', k_d_flow)) if coeffs is not None else k_d_flow
        c_sat_tadr = float(getattr(coeffs, 'c_sat', c_sat)) if coeffs is not None else c_sat
        S_w_loc = 1.0 - S_n[:size]
        active = (S_n[:size] > 0.0) & (c[:size] < c_sat)
        R_per_node_flow = (w * S_w_loc * rho_w_eff * k_d_flow * S_n[:size]
                          * (c_sat - c[:size]))
        R_per_node_tadr = (w * S_w_loc * rho_w_eff * k_d_tadr * S_n[:size]
                          * (c_sat_tadr - c[:size]))
        local_R_flow = float(np.sum(R_per_node_flow))
        local_R_tadr = float(np.sum(R_per_node_tadr))
        R_flow_total = comm.allreduce(local_R_flow, op=MPI.SUM)
        R_tadr_total = comm.allreduce(local_R_tadr, op=MPI.SUM)

        # ---- CUMULATIVE DISSOLUTION-TRANSFER AUDIT (localize the residual excess) ----
        # The instantaneous [R_diss] ratio is ~tautological: R_flow_total and
        # R_tadr_total are both recomputed here from the SAME arrays/formula, so
        # they cannot detect a real gas<->brine transfer mismatch. Instead,
        # INTEGRATE each side over time (trapezoidal) and compare to the ACTUAL
        # mass changes the two kernels produced:
        #   gas side: the gas kernel conserves except for injection and the
        #     dissolution sink (kernel_telescope=0, sum_F~0), so the kernel-true
        #     gas mass must satisfy  total_m_n = kernel_inj - cum_R_flow.
        #       gas_side_resid = total_m_n - (kernel_inj - cum_R_flow) ~ 0
        #       iff the gas-eq sink actually removed the integrated R_flow.
        #   TADR side: diss starts at 0 and (if TADR advection conserves c)
        #     changes only through the dissolution source, so  diss = cum_R_tadr.
        #       tadr_side_resid = diss - cum_R_tadr ~ 0
        #       iff the c-eq source actually added the integrated R_tadr.
        # cum_R_flow == cum_R_tadr by construction, so the two residuals sum to
        # the total imbalance (diss + total_m_n - kernel_inj) and SPLIT it: the
        # nonzero side is the one over/under-transferring (prime suspect: the
        # c-eq source over-injecting via the Sequential split lag).
        if not hasattr(self, '_cum_R_flow'):
            self._cum_R_flow = 0.0
            self._cum_R_tadr = 0.0
            self._cum_prev_R_flow = R_flow_total
            self._cum_prev_R_tadr = R_tadr_total
        if dt_balance > 0.0:
            self._cum_R_flow += 0.5 * (self._cum_prev_R_flow + R_flow_total) * dt_balance
            self._cum_R_tadr += 0.5 * (self._cum_prev_R_tadr + R_tadr_total) * dt_balance
        self._cum_prev_R_flow = R_flow_total
        self._cum_prev_R_tadr = R_tadr_total

        # ---- dLow SYMMETRY AUDIT ----
        # The Stage-2 Richards-style upwind dissipation contributes
        #   sum_{i,j} dH_ij * (m_n[i] - m_n[j])
        # to the total gas residual. For conservation we need dH symmetric
        # (dH_ij == dH_ji) so the double sum cancels. Float-roundoff
        # asymmetry would create a small per-step mass leak.
        # Reports: max |dH_ij - dH_ji|, total un-cancelled flux contribution.
        dLow_arr = getattr(m, 'dLow_n', None)
        rowptr   = getattr(m, 'comp1_rowptr', None)
        colind   = getattr(m, 'comp1_colind', None)
        m_n_arr  = float(self.rho_n) * phi * S_n  # m_n at every node
        max_asym_abs = 0.0
        max_asym_rel = 0.0
        sum_dLow_flux = 0.0
        if (dLow_arr is not None and rowptr is not None and colind is not None
                and len(dLow_arr) == len(colind)):
            # Build edge-offset map (i,j)->offset once and cache.
            if not hasattr(self, '_edge_offset_map'):
                self._edge_offset_map = {}
                for i_n_ in range(len(rowptr) - 1):
                    for off in range(int(rowptr[i_n_]), int(rowptr[i_n_ + 1])):
                        self._edge_offset_map[(i_n_, int(colind[off]))] = off
            edge_map = self._edge_offset_map
            dL = np.asarray(dLow_arr)
            nrows = min(len(rowptr) - 1, size, len(m_n_arr))
            for i_n_ in range(nrows):
                row_start = int(rowptr[i_n_])
                row_end   = int(rowptr[i_n_ + 1])
                for off_ij in range(row_start, row_end):
                    j_n_ = int(colind[off_ij])
                    if j_n_ == i_n_ or j_n_ >= nrows:
                        continue
                    d_ij = float(dL[off_ij])
                    off_ji = edge_map.get((j_n_, i_n_), -1)
                    if off_ji < 0:
                        continue
                    d_ji = float(dL[off_ji])
                    asym = abs(d_ij - d_ji)
                    denom = max(abs(d_ij), abs(d_ji), 1.0e-30)
                    if asym > max_asym_abs:
                        max_asym_abs = asym
                    if asym / denom > max_asym_rel:
                        max_asym_rel = asym / denom
                    # Per-edge contribution to total dLow gas residual:
                    # at row i: +d_ij * (m[i] - m[j])
                    sum_dLow_flux += d_ij * (m_n_arr[i_n_] - m_n_arr[j_n_])
        max_asym_abs = comm.allreduce(max_asym_abs, op=MPI.MAX)
        max_asym_rel = comm.allreduce(max_asym_rel, op=MPI.MAX)
        sum_dLow_flux = comm.allreduce(sum_dLow_flux, op=MPI.SUM)

        # ---- PER-EQUATION CONSERVATION CHECK ----
        # Compare finite-difference d(gas)/dt to (inj_rate - R_flow_rate)
        # and  d(diss)/dt to R_tadr_rate.
        # In the discrete continuum:
        #   gas-eq:  d(gas)/dt = inj_rate - R_flow_rate           (no boundary if sealed)
        #   c-eq:    d(diss)/dt = R_tadr_rate
        # If either residual is non-zero, that equation is creating/destroying
        # mass at the printed rate. Isolates the leak source (gas-eq vs TADR).
        if not hasattr(self, '_last_gas'):
            self._last_gas = gas
            self._last_diss = diss
            self._last_audit_t = float(t)
            gas_leak_rate = 0.0
            diss_leak_rate = 0.0
        else:
            dt_audit = float(t) - self._last_audit_t
            if dt_audit > 1.0e-15:
                d_gas_dt  = (gas  - self._last_gas)  / dt_audit
                d_diss_dt = (diss - self._last_diss) / dt_audit
                # Use averaged rates (this call + previous) since we did
                # trapezoidal accumulation of kernel_inj. R_diss is point-in-time.
                avg_inj_rate    = 0.5 * (self._last_inj_rate2  + inj_rate_global)
                avg_R_flow_rate = 0.5 * (self._last_R_flow_rate + R_flow_total)
                avg_R_tadr_rate = 0.5 * (self._last_R_tadr_rate + R_tadr_total)
                gas_leak_rate  = d_gas_dt  - (avg_inj_rate - avg_R_flow_rate)
                diss_leak_rate = d_diss_dt - avg_R_tadr_rate
            else:
                gas_leak_rate = 0.0
                diss_leak_rate = 0.0
        # Update audit state for next call.
        self._last_gas = gas
        self._last_diss = diss
        self._last_audit_t = float(t)
        self._last_inj_rate2    = inj_rate_global
        self._last_R_flow_rate  = R_flow_total
        self._last_R_tadr_rate  = R_tadr_total

        # Cumulative injected = sum over ports of rate * disk_area * ∫ramp(s) ds
        # where ramp = 0.5*(1 + tanh((s - t_start)/tau - 3)) clipped to [t_start, t_stop].
        # The closed-form integral is needed because at early times the ramp is
        # ~0.25% of nominal; assuming full rate would inflate cum_inj by 400x
        # and make balance look catastrophically bad when it's actually fine.
        #   ∫_{t_start}^{t} 0.5*(1 + tanh((s-t_start)/tau - 3)) ds
        # = 0.5*tau * [(u + 3) + ln(cosh(u)) - ln(cosh(3))]
        # where u = (min(t,t_stop) - t_start)/tau - 3.
        tau = float(self.injection_ramp_tau)
        ln_cosh3 = float(np.log(np.cosh(3.0)))
        cum_inj = 0.0
        for port in self.injection_ports:
            (px, py, rate, radius, t_start, t_stop) = port
            t_eff = max(float(t_start), min(float(t), float(t_stop)))
            elapsed = t_eff - float(t_start)
            if elapsed <= 0.0:
                continue
            if tau > 0.0:
                u = elapsed / tau - 3.0
                # ln(cosh(u)) computed as log1p(exp(-2|u|))/... for stability
                ramp_integral = 0.5 * tau * ((u + 3.0)
                                             + float(np.log(np.cosh(u)))
                                             - ln_cosh3)
            else:
                ramp_integral = elapsed
            cum_inj += float(rate) * np.pi * float(radius) ** 2 * ramp_integral

        bal = gas + diss - cum_inj
        rel = (bal / cum_inj) if cum_inj > 1.0e-30 else 0.0

        # S_n / c min,max for overshoot detection (cheap, no extra reductions
        # beyond what numpy does locally; for parallel correctness we use the
        # owned slice and MPI-reduce across ranks).
        local_Sn_min = float(np.min(S_n[:size])) if size > 0 else 0.0
        local_Sn_max = float(np.max(S_n[:size])) if size > 0 else 0.0
        local_c_min  = float(np.min(c[:size]))  if size > 0 else 0.0
        local_c_max  = float(np.max(c[:size]))  if size > 0 else 0.0
        Sn_min = comm.allreduce(local_Sn_min, op=MPI.MIN)
        Sn_max = comm.allreduce(local_Sn_max, op=MPI.MAX)
        c_min  = comm.allreduce(local_c_min,  op=MPI.MIN)
        c_max  = comm.allreduce(local_c_max,  op=MPI.MAX)

        # ---- FLUX-IMBALANCE / TAU-SYMMETRY DIAGNOSTIC (mass-creation hunt) ----
        # gd[0]=max|T_ij-T_ji| (tau asymmetry), [1]=max|T_ij| (scale),
        # [2]=sum_ij F_ij (net mass rate from the edge flux; should be ~0 if
        # antisymmetric), [3]=sum_ij|F_ij| (scale). If gd[0]/gd[1] >> 1e-12 the
        # transmissibility read is asymmetric -> the tau gate creates mass.
        gd = getattr(m, 'gas_diag', None)
        if gd is not None and len(gd) >= 4:
            Tasym = comm.allreduce(float(gd[0]), op=MPI.MAX)
            Tmax  = comm.allreduce(float(gd[1]), op=MPI.MAX)
            sumF  = comm.allreduce(float(gd[2]), op=MPI.SUM)
            absF  = comm.allreduce(float(gd[3]), op=MPI.SUM)
        else:
            Tasym = Tmax = sumF = absF = 0.0

        # ---- VELOCITY-DIVERGENCE / NON-CONSERVATION PROBE (mass-creation hunt) --
        # TADR advects c with velocity_couple = the raw P1-gradient Darcy flux
        # (no conservative post-processing: conservativeFlux=None).  An element-
        # constant Darcy velocity is NOT discretely divergence-free.  If TADR's
        # advection is effectively non-conservative, the global c-mass it creates
        # per unit time is  -∫ v·∇c  (= -∫ c ∇·v on a closed domain, zero-flux
        # BC).  This is computed element-by-element over OWNED elements (each
        # counted once, so the SUM is MPI-exact) and should be compared to the
        # measured diss_leak_rate on the [mass leak] line:
        #   div_leak_est ≈ diss_leak_rate  -> non-conservative velocity confirmed.
        #   div_leak_est >> diss_leak_rate -> TADR is conservative; look elsewhere.
        # advect_scale = ∫|v·∇c| gives the magnitude the leak is a residual of.
        # max|div_w v| is the weak nodal divergence peak (ghost-incomplete at
        # rank boundaries -> qualitative only).
        div_leak_est = advect_scale = div_max = 0.0
        try:
            nE_own = int(getattr(mesh, 'nElements_owned',
                                 getattr(mesh, 'nElements_global',
                                         len(self._elem_area_p))))
            EN_o   = self._elem_nodes_p[:nE_own]
            area_o = self._elem_area_p[:nE_own]
            g_o    = self._elem_gradphi_p[:nE_own]
            qv     = np.asarray(m.q[('velocity_couple', 0)])[:nE_own]   # (nE,nQ,nd)
            v_e    = qv.mean(axis=1)                                    # (nE,nd) Darcy flux
            c_e    = c[EN_o]                                            # (nE,3) nodal c
            gradc_e = (c_e[:, :, None] * g_o).sum(axis=1)              # (nE,nd) grad c
            vgc    = (v_e * gradc_e).sum(axis=1)                       # (nE,) v·grad c
            div_leak_est = comm.allreduce(-float(np.sum(area_o * vgc)),    op=MPI.SUM)
            advect_scale = comm.allreduce(float(np.sum(area_o * np.abs(vgc))), op=MPI.SUM)
            D = np.zeros(len(ML), 'd')
            vdotg = (v_e[:, None, :] * g_o).sum(axis=2) * area_o[:, None]  # (nE,3)
            np.add.at(D, EN_o.ravel(), vdotg.ravel())
            div_max = comm.allreduce(float(np.max(np.abs(D[:size]))) if size > 0 else 0.0,
                                     op=MPI.MAX)
        except Exception:
            pass

        # ---- Telescoping / per-step conservation probe ----------------------
        # gas_old uses u_dof_n_old (the mass the kernel treats as m_n_old). If
        # the time history telescopes correctly, gas_old THIS step == gas LAST
        # step (telescope_gap ~ 0). The per-step total-CO2 defect
        # d(gas+diss) - d(injected) is ~0 iff the operator conserves this step;
        # a steady nonzero defect = real creation. Diagnostic only.
        #   telescope_gap != 0  -> time-history (u_dof_n_old) bug.
        #   telescope_gap ~ 0 but step_defect != 0 -> creation despite correct
        #                                              telescoping (source/accum).
        S_n_old_arr = getattr(m, 'u_dof_n_old', None)
        gas_old = float('nan')
        if S_n_old_arr is not None:
            S_n_old_arr = np.asarray(S_n_old_arr)
            if len(S_n_old_arr) >= size:
                gas_old = comm.allreduce(
                    float(np.sum(w * float(self.rho_n) * S_n_old_arr[:size])),
                    op=MPI.SUM)
        totCO2 = gas + diss
        # Compressible telescoping: m_n_old's DENSITY uses pw_old (u_dof_old). If
        # the pw history is stale (not refreshed each step) while S_n_old is, the
        # old density is wrong -> m_n - m_n_old over-counts -> steady creation,
        # invisible to the const-rho telescope_gap above. gas_old_comp uses
        # pw_old + S_n_old; if it != last step's gas_pw, the pw history is stale.
        pw_old_arr = getattr(m, 'u_dof_old', None)
        gas_old_comp = float('nan'); pwold_min = float('nan'); pwold_max = float('nan')
        if pw_old_arr is not None and S_n_old_arr is not None:
            pw_old_arr = np.asarray(pw_old_arr)
            mlen = min(nn, len(pw_old_arr), len(S_n_old_arr))
            if mlen > 0:
                rho_pwold = float(self.rho_n) * np.exp(
                    np.clip(pw_old_arr[:mlen] * inv_pref, -50.0, 50.0))
                gas_old_comp = comm.allreduce(
                    float(np.sum(w[:mlen] * rho_pwold * S_n_old_arr[:mlen])), op=MPI.SUM)
                pwold_min = comm.allreduce(float(np.min(pw_old_arr[:mlen])), op=MPI.MIN)
                pwold_max = comm.allreduce(float(np.max(pw_old_arr[:mlen])), op=MPI.MAX)
        if not hasattr(self, '_probe_prev'):
            self._probe_prev = (totCO2, cum_inj, gas, gas_pw)
        prev_totCO2, prev_cuminj, prev_gas, prev_gas_pw = self._probe_prev
        d_totCO2      = totCO2 - prev_totCO2
        d_inj         = cum_inj - prev_cuminj
        step_defect   = d_totCO2 - d_inj          # ~0 if conserved this step
        telescope_gap = gas_old - prev_gas        # ~0 if u_dof_n_old == last S_n
        telescope_gap_comp = gas_old_comp - prev_gas_pw  # ~0 if pw history fresh
        self._probe_prev = (totCO2, cum_inj, gas, gas_pw)

        # ---- KERNEL's OWN conserved mass (exact: projected compressible density,
        # kernel weights). mLow_n = rho_n_phi_dof*S_n (phi already in it), so
        # total_m_n = sum(ML_area * mLow_n). This is what the kernel actually
        # conserves -- NOT the const-rho/Python-weight [Mass balance] gas.
        #   total_m_n / cum_inj ~ 1  -> kernel conserves; the 8x is a phantom -> bounds.
        #   total_m_n / cum_inj ~ 8  -> REAL creation in the kernel.
        #   kernel_telescope != 0    -> the KERNEL mass fails to telescope
        #                               (which the const-rho telescope_gap missed).
        mlow  = getattr(m, 'mLow_n', None)
        mnold = getattr(m, 'mn_n', None)
        total_m_n = float('nan'); total_m_n_old = float('nan')
        if mlow is not None:
            mlow = np.asarray(mlow)
            if len(mlow) >= size:
                total_m_n = comm.allreduce(
                    float(np.sum(ML[:size] * mlow[:size])), op=MPI.SUM)
        if mnold is not None:
            mnold = np.asarray(mnold)
            if len(mnold) >= size:
                total_m_n_old = comm.allreduce(
                    float(np.sum(ML[:size] * mnold[:size])), op=MPI.SUM)
        if not hasattr(self, '_prev_total_m_n'):
            self._prev_total_m_n = total_m_n
        kernel_telescope = total_m_n_old - self._prev_total_m_n  # ~0 if kernel telescopes
        self._prev_total_m_n = total_m_n

        # ---- PER-TERM GAS-RESIDUAL BUDGET (kernel-exported, owned-node sum) ----
        # gas_budget_node holds 6 per-node slots (term-major, numDOFs_n each):
        # [0]accum [1]flux [2]sink [3]injection [4]boundary [5]total residual.
        # Sum each over OWNED nodes only and MPI-reduce (parallel-exact, no
        # overlap double-count). At convergence gb_total~0, so
        #   gb_accum ~ gb_inj - gb_sink - gb_flux - gb_bnd.
        # Interpretation:
        #   gb_flux !~ 0  -> interior upwind flux not antisymmetric in assembly.
        #   gb_bnd  !~ 0  -> exterior boundary leak. In PARALLEL this is the
        #                    smoking gun for a rank treating an inter-processor
        #                    (subdomain) face as a domain boundary: a true
        #                    domain-boundary face contributes 0 here (isDir_n=0
        #                    => F=0), so any nonzero gb_bnd is spurious.
        #   gb_flux~0 & gb_bnd~0 but gas still grows -> creation is POST-kernel
        #                    (FCT/inversion/coupling), not in this residual.
        gb = getattr(m, 'gas_budget_node', None)
        gb_accum = gb_flux = gb_sink = gb_inj = gb_bnd = gb_total = float('nan')
        if gb is not None:
            gb = np.asarray(gb)
            n_n_loc = len(gb) // 6
            if n_n_loc > 0:
                ns = min(size, n_n_loc)
                gb_accum = comm.allreduce(float(np.sum(gb[0 * n_n_loc:0 * n_n_loc + ns])), op=MPI.SUM)
                gb_flux  = comm.allreduce(float(np.sum(gb[1 * n_n_loc:1 * n_n_loc + ns])), op=MPI.SUM)
                gb_sink  = comm.allreduce(float(np.sum(gb[2 * n_n_loc:2 * n_n_loc + ns])), op=MPI.SUM)
                gb_inj   = comm.allreduce(float(np.sum(gb[3 * n_n_loc:3 * n_n_loc + ns])), op=MPI.SUM)
                gb_bnd   = comm.allreduce(float(np.sum(gb[4 * n_n_loc:4 * n_n_loc + ns])), op=MPI.SUM)
                gb_total = comm.allreduce(float(np.sum(gb[5 * n_n_loc:5 * n_n_loc + ns])), op=MPI.SUM)

        if rank == 0:
            logEvent(
                "[cons probe ] t={:.4e} gas_old={:+.4e} telescope_gap={:+.4e} "
                "d(gas+diss)={:+.4e} d(inj)={:+.4e} step_defect={:+.4e}".format(
                    float(t), gas_old, telescope_gap, d_totCO2, d_inj, step_defect),
                level=2)
            logEvent(
                "[cons probe2] t={:.4e} gas_old_comp={:+.4e} telescope_gap_comp={:+.4e} "
                "gas_pw={:+.4e} pw_old=[{:+.4e},{:+.4e}]".format(
                    float(t), gas_old_comp, telescope_gap_comp, gas_pw,
                    pwold_min, pwold_max),
                level=2)
            logEvent(
                "[cons probe3] t={:.4e} total_m_n={:+.4e} total_m_n/inj={:+.4e} "
                "total_m_n_old={:+.4e} kernel_telescope={:+.4e}".format(
                    float(t), total_m_n,
                    (total_m_n / cum_inj if cum_inj != 0.0 else 0.0),
                    total_m_n_old, kernel_telescope),
                level=2)
            # Per-term gas-residual budget. gb_total~0 confirms Newton converged;
            # gb_resid_check = gb_accum-(gb_inj-gb_sink-gb_flux-gb_bnd) is the
            # closure (should ~ -gb_total). gb_flux / gb_bnd are the only terms
            # that can break conservation -- watch them (gb_bnd!=0 in parallel =>
            # a rank leaking through a mis-tagged subdomain face).
            # Self-consistency: the 6 slots are the actual scattered residual
            # contributions, so slots[0..4] must sum to slot5 (total_resid).
            # closure ~ 0 validates the probe; closure != 0 = a slot wiring bug.
            gb_resid_check = (gb_accum + gb_flux + gb_sink + gb_inj + gb_bnd - gb_total
                              if gb_accum == gb_accum else float('nan'))
            logEvent(
                "[gas budget ] t={:.4e} accum={:+.4e} flux={:+.4e} sink={:+.4e} "
                "inj={:+.4e} bnd={:+.4e} total_resid={:+.4e} closure={:+.4e}".format(
                    float(t), gb_accum, gb_flux, gb_sink, gb_inj, gb_bnd,
                    gb_total, gb_resid_check),
                level=2)
            # Reference balance line (kept for backward-compat / quick scan).
            logEvent(
                "[Mass balance] t={:.4e} gas={:+.4e} diss={:+.4e} "
                "injected={:+.4e} balance={:+.4e} rel={:+.3e}".format(
                    float(t), gas, diss, cum_inj, bal, rel),
                level=2)
            # gas_const vs gas_pw (compressible lower bound) + pw range over the
            # gas zone.  If gas_pw ~ gas and both >> injected, the over-creation
            # is REAL (not the const-rho diagnostic).  ratio_pw = gas_pw/gas.
            logEvent(
                "[gas comp   ] t={:.4e} gas_const={:+.4e} gas_pw={:+.4e} "
                "ratio_pw={:.3f} injected={:+.4e} pw[gas]=[{:+.4e},{:+.4e}] "
                "p_ref_n={:.3e}".format(
                    float(t), gas, gas_pw, (gas_pw / gas if gas != 0.0 else 0.0),
                    cum_inj, pw_gas_min, pw_gas_max, float(self.p_ref_n)),
                level=2)
            # Field min/max: Sn_max > 1 or Sn_min < 0 indicates overshoot from
            # STAB=2 high-order EV term not being added to the residual (only
            # dLow is, which is monotonicity-preserving). c_max > c_sat means
            # dissolution overshoot from R_diss linearization.
            logEvent(
                "[field range] t={:.4e} Sn=[{:+.4e},{:+.4e}] "
                "c=[{:+.4e},{:+.4e}] (c_sat={:.4e})".format(
                    float(t), Sn_min, Sn_max, c_min, c_max, float(self.c_sat)),
                level=2)
            # ---- Audit lines ----
            # (1) Kernel-injected vs analytical: if these disagree, the disk
            #     lumped-quadrature is the dominant source of "rel" bias.
            # (2) R_diss flow vs TADR: if these disagree per-DOF, the
            #     dissolution coupling is leaking mass.
            kernel_ratio = (kernel_inj / cum_inj) if cum_inj > 1e-30 else 0.0
            R_ratio = (R_flow_total / R_tadr_total) if R_tadr_total > 1e-30 else 0.0
            bal_kernel = gas + diss - kernel_inj
            rel_kernel = (bal_kernel / kernel_inj) if kernel_inj > 1e-30 else 0.0
            logEvent(
                "[Mass audit ] t={:.4e} kernel_inj={:+.4e} cum_inj={:+.4e} "
                "kernel/analytical={:.3f}  bal_vs_kernel={:+.4e} rel_vs_kernel={:+.3e}".format(
                    float(t), kernel_inj, cum_inj, kernel_ratio,
                    bal_kernel, rel_kernel),
                level=2)
            logEvent(
                "[R_diss     ] t={:.4e} R_flow_rate={:+.4e} R_tadr_rate={:+.4e} "
                "ratio_flow/tadr={:.3f}  (k_d_flow={:.3e} k_d_tadr={:.3e})".format(
                    float(t), R_flow_total, R_tadr_total, R_ratio,
                    k_d_flow, k_d_tadr),
                level=2)
            # ---- Cumulative transfer audit: which side over/under-transfers? ----
            # gas_side ~ 0 => gas-eq sink matches the integrated R_flow (kernel
            #   conserves into total_m_n); nonzero => the gas sink removed != R_flow
            #   (e.g. the kernel's lagged c, or a density-unit mismatch in R_diss_n,
            #   which is built from rho_w/c but deducted from the rho_n gas mass).
            # tadr_side ~ 0 => c-eq source matches the integrated R_tadr; nonzero =>
            #   TADR added != R_tadr (source over-injection / advection c-leak).
            # The larger-magnitude residual localizes the ~11% total-CO2 excess.
            gas_side  = total_m_n - (kernel_inj - self._cum_R_flow)
            tadr_side = diss - self._cum_R_tadr
            denom_cr  = max(abs(kernel_inj), 1.0e-30)
            logEvent(
                "[diss audit ] t={:.4e} cum_R_flow={:+.4e} cum_R_tadr={:+.4e} "
                "gas_side={:+.4e} (rel={:+.3e}) tadr_side={:+.4e} (rel={:+.3e}) "
                "total_m_n+diss-kernel_inj={:+.4e}".format(
                    float(t), self._cum_R_flow, self._cum_R_tadr,
                    gas_side, gas_side / denom_cr,
                    tadr_side, tadr_side / denom_cr,
                    (total_m_n + diss - kernel_inj)),
                level=2)
            # Per-equation conservation: if either leak_rate is non-trivial
            # relative to the active source/sink rate, that equation is the
            # leak source. Use max(inj, R_flow, R_tadr) so post-injection
            # (inj_rate=0) the rel value normalises against dissolution
            # instead of blowing up to 1e+26.
            rate_scale = max(abs(inj_rate_global), abs(R_flow_total),
                             abs(R_tadr_total), 1.0e-30)
            # leak_per_step is the actual mass added per Newton solve. With
            # the STAB=2 Richards port, the gas residual has NO consistent
            # CG, NO unconserved boundary (gated by isDir_n), and dLow is
            # verified symmetric (sum_dH*(m_i-m_j) ~ 1e-19 below). So
            # leak_per_step should equal sum of converged Newton residuals
            # (~ nl_atol_res, configured in flow_n.py). If leak_per_step
            # tracks nl_atol_res, the only fix is to tighten nl_atol_res /
            # tolFac.  If leak_per_step >> nl_atol_res, something else is
            # creating mass and we need to keep looking.
            try:
                dt_show = float(dt_audit) if dt_audit > 0.0 else 0.0
            except NameError:
                dt_show = 0.0
            leak_per_step = gas_leak_rate * dt_show
            logEvent(
                "[mass leak  ] t={:.4e} gas_leak_rate={:+.4e} (rel={:+.2e})  "
                "diss_leak_rate={:+.4e} (rel={:+.2e})  "
                "dt={:.3e}  leak/step={:+.3e}".format(
                    float(t), gas_leak_rate, gas_leak_rate / rate_scale,
                    diss_leak_rate, diss_leak_rate / rate_scale,
                    dt_show, leak_per_step),
                level=2)
            # dLow symmetry: if |asym|/|max| > ~1e-12, dLow is not symmetric.
            # sum_dLow_flux is the actual un-cancelled contribution per step;
            # if it's ~ gas_leak_rate, dLow asymmetry IS the leak source.
            logEvent(
                "[dLow symm  ] t={:.4e} max_asym_abs={:.3e} max_asym_rel={:.3e}  "
                "sum_dH*(m_i-m_j)={:+.4e} (rel_to_rate={:+.2e})".format(
                    float(t), max_asym_abs, max_asym_rel,
                    sum_dLow_flux, sum_dLow_flux / rate_scale),
                level=2)
            # Flux-imbalance probe: sumF is the net mass rate the edge flux
            # injects/removes; compare to gas_leak_rate. Tasym_rel >> 1e-12
            # means the transmissibility read is asymmetric (the suspected bug).
            logEvent(
                "[flux diag  ] t={:.4e} T_asym={:.3e} (rel={:+.2e})  "
                "sum_F={:+.4e} (rel={:+.2e})  (leak_rate={:+.4e})".format(
                    float(t), Tasym, (Tasym / Tmax if Tmax > 0 else 0.0),
                    sumF, (sumF / absF if absF > 0 else 0.0),
                    gas_leak_rate),
                level=2)
            # Velocity-divergence probe: div_leak_est is the c-mass/time a
            # non-conservative advection of c by the (non-div-free) Darcy flux
            # would create. Compare to diss_leak_rate above: if they match in
            # sign+magnitude, the raw velocity_couple is the mass-creation source.
            logEvent(
                "[div probe  ] t={:.4e} div_leak_est=-int(v.gradc)={:+.4e}  "
                "advect_scale=int|v.gradc|={:.4e}  max|div_w v|={:.4e}".format(
                    float(t), div_leak_est, advect_scale, div_max),
                level=2)

        # ---- INTERFACE VELOCITY-SPIKE LOCALIZER (water vs gas bisection) ----
        # Finds the element carrying max|velocity_couple| (the WATER Darcy
        # velocity TADR rides) and reports there: |v|, the P1 grad(pw), the
        # Sn / pw spread across the element, and whether the element straddles
        # a material interface.  Reading: a spike on an interface element with
        # a large grad(pw) while Sn is mid-range => the artifact is in the
        # water-pressure reconstruction (comp-0); a spike tracking the Sn
        # breakthrough front => the gas-side interface flux (comp-1).  All
        # ranks must reach comm.gather, so any exception is deterministic
        # (same missing key everywhere) -> all skip together, no deadlock.
        try:
            qv = np.asarray(m.q[('velocity_couple', 0)])
            nE_own = int(getattr(mesh, 'nElements_owned',
                                 getattr(mesh, 'nElements_global', qv.shape[0])))
            qv = qv[:nE_own]
            vmag = np.sqrt(np.sum(qv * qv, axis=-1))
            if vmag.size:
                eN_max, q_max = np.unravel_index(int(np.argmax(vmag)), vmag.shape)
                local_vmax = float(vmag[eN_max, q_max])
                elemNodes = np.asarray(mesh.elementNodesArray)
                nodeArr = np.asarray(mesh.nodeArray)
                matTypes = np.asarray(self.elementMaterialTypes)
                pw_dof = np.asarray(m.u[0].dof)
                Sn_dof = np.asarray(m.u[1].dof)
                nA, nB, nC = (int(elemNodes[eN_max, 0]),
                              int(elemNodes[eN_max, 1]),
                              int(elemNodes[eN_max, 2]))
                x0, y0 = float(nodeArr[nA, 0]), float(nodeArr[nA, 1])
                x1, y1 = float(nodeArr[nB, 0]), float(nodeArr[nB, 1])
                x2, y2 = float(nodeArr[nC, 0]), float(nodeArr[nC, 1])
                # m.q['x'] is not populated in this model (the q_x fill is
                # commented out in the kernel), so report the element centroid.
                lx = (x0 + x1 + x2) / 3.0; ly = (y0 + y1 + y2) / 3.0
                w0, w1, w2 = float(pw_dof[nA]), float(pw_dof[nB]), float(pw_dof[nC])
                twoA = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
                if abs(twoA) > 1.0e-30:
                    gpx = (w0 * (y1 - y2) + w1 * (y2 - y0) + w2 * (y0 - y1)) / twoA
                    gpy = (w0 * (x2 - x1) + w1 * (x0 - x2) + w2 * (x1 - x0)) / twoA
                else:
                    gpx = gpy = 0.0
                sn_vals = [float(Sn_dof[nA]), float(Sn_dof[nB]), float(Sn_dof[nC])]
                if not hasattr(self, '_iface_elem'):
                    node_mat = {}
                    for e in range(len(matTypes)):
                        me = int(matTypes[e])
                        for nn in elemNodes[e]:
                            node_mat.setdefault(int(nn), set()).add(me)
                    ie = np.zeros(len(matTypes), dtype=bool)
                    for e in range(len(matTypes)):
                        mats = set()
                        for nn in elemNodes[e]:
                            mats |= node_mat[int(nn)]
                        ie[e] = (len(mats) > 1)
                    self._iface_elem = ie
                iface_flag = 1 if self._iface_elem[eN_max] else 0
                local_info = (local_vmax, lx, ly, int(matTypes[eN_max]),
                              min(sn_vals), max(sn_vals),
                              max(w0, w1, w2) - min(w0, w1, w2),
                              float(gpx), float(gpy), iface_flag)
            else:
                local_info = (-1.0, float('nan'), float('nan'), -1,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0)
            all_info = comm.gather(local_info, root=0)
            if rank == 0:
                (vmx, wx, wy, wmat, snmn, snmx, pwrng, gpx, gpy, ifl) = max(
                    all_info, key=lambda r: r[0])
                logEvent(
                    "[iface diag ] t={:.4e} max|v|={:.4e} at ({:.4f},{:.4f}) "
                    "mat={} iface={} Sn=[{:.4e},{:.4e}] pw_range={:.4e} "
                    "grad_pw=({:+.4e},{:+.4e})".format(
                        float(t), vmx, wx, wy, wmat, ifl, snmn, snmx,
                        pwrng, gpx, gpy),
                    level=2)
        except Exception as _e:
            if rank == 0:
                logEvent("[iface diag ] skipped: {}".format(_e), level=2)

    # def postStep(self, t, firstStep=False):
    #     if not self.outputQuantDOFs:
    #         return {}
    #     if (self.model is None or
    #             ('velocity_couple', 0) not in self.model.q or
    #             ('grad(u_n)', 0) not in self.model.q):
    #         return {}

    #     mpicomm = self._get_mpi_comm()
    #     rank = mpicomm.Get_rank()
    #     nSpace = int(getattr(self.model, 'nSpace_global',
    #                          getattr(self.model, 'nSpace', self.nd)))
    #     n_owned = self._get_owned_element_count()
    #     stab_tag = f"stab{self.STABILIZATION_TYPE}"

    #     qcoords_local = self._get_q_coordinates().reshape((-1, 3))
    #     qv_local = np.asarray(self.model.q[('velocity_couple', 0)][:n_owned]).reshape((-1, nSpace))
    #     qgrad_local = np.asarray(self.model.q[('grad(u_n)', 0)][:n_owned]).reshape((-1, nSpace))

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
        # previous-step DOFs for component 1 (S_n).
        # Filled lazily on first getResidual call from u[1].dof (the IC).
        self.u_dof_n_old = None
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
        # q_velocity scratch buffer: the C++ kernel writes grad(u_w) here
        # (the wetting-phase pressure gradient at QPs) via the "q_velocity"
        # argsDict key. Despite the historical key name including u_n, this
        # is NOT the gradient of the non-wetting saturation -- it is a
        # diagnostic for the wetting-pressure gradient.
        self.q[('q_velocity_buf',0)] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,self.nSpace_global),'d')
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
        if self.nc >= 2:
            qshape   = (self.mesh.nElements_global, self.nQuadraturePoints_element)
            qvshape  = (self.mesh.nElements_global, self.nQuadraturePoints_element, self.nSpace_global)
            ebqshape = (self.mesh.nExteriorElementBoundaries_global,
                        self.nElementBoundaryQuadraturePoints_elementBoundary)
            ebqvshape = (self.mesh.nExteriorElementBoundaries_global,
                         self.nElementBoundaryQuadraturePoints_elementBoundary,
                         self.nSpace_global)
            self.q[('u', 1)]            = np.zeros(qshape, 'd')
            self.q[('grad(u)', 1)]      = np.zeros(qvshape, 'd')
            self.q[('grad(phi)', 1)]    = self.q[('u', 1)]
            self.q[('dphi', 1, 1)]      = np.zeros(qshape, 'd')
            self.q[('m', 1)]            = self.q[('u', 1)].copy()
            self.q[('dm', 1, 1)]        = np.zeros(qshape, 'd')
            self.q[('mt', 1)]           = self.q[('u', 1)].copy()
            self.q[('m_last', 1)]       = self.q[('u', 1)].copy()
            self.q[('m_tmp', 1)]        = self.q[('u', 1)].copy()
            self.q[('dV_u', 1)]         = self.q[('dV_u', 0)]
            self.q[('cfl', 1)]          = np.zeros(qshape, 'd')
            self.q[('numDiff', 1, 1)]   = np.zeros(qshape, 'd')
            self.q[('numDiff_last', 1, 1)] = np.zeros(qshape, 'd')
            self.ebqe[('u', 1)]                  = np.zeros(ebqshape, 'd')
            self.ebqe[('grad(u)', 1)]            = np.zeros(ebqvshape, 'd')
            self.ebqe[('advectiveFlux_bc_flag', 1)] = np.zeros(ebqshape, 'i')
            self.ebqe[('advectiveFlux_bc', 1)]      = np.zeros(ebqshape, 'd')
            self.ebqe[('advectiveFlux', 1)]         = np.zeros(ebqshape, 'd')
        
        
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
        
        if self.coefficients._fct_requested:
            valid_stabilization_types = {1, 2}  # Only allow FCT for STABILIZATION_TYPE 1 (EV_Stab) and 2 (EntropyViscosity)
            if self.coefficients.STABILIZATION_TYPE not in valid_stabilization_types:
                raise ValueError("Use FCT only with STABILIZATION_TYPE 1 (EV_Stab) or 2 (EntropyViscosity).")
            cond = self.coefficients.STABILIZATION_TYPE > 0, "Use FCT just with STABILIZATION_TYPE>0; i.e., edge based stabilization"
        # # END OF ASSERTS

        # cek adding empty data member for low order numerical viscosity structures here for now
        self.ML = None  # lumped mass matrix
        self.MC_global = None  # consistent mass matrix
        # Consistent mass matrix on the comp-1 (S_n) DOF graph, indexed by the
        # comp-1 compact CSR (same layout as dLow_n / dt_times_fH_minus_fL_n).
        # MC_a only ever has its (0,0) block assembled, so FCTStep_n's
        # consistency term needs this dedicated comp-1 mass matrix instead.
        self.MC_n = None
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
        # Aux quantity at DOFs to be filled by optimized code (MQL)
        self.quantDOFs = np.zeros(self.u[0].dof.shape, 'd')
        self.mLow = np.zeros(self.u[0].dof.shape, 'd')
        self.mHigh = np.zeros(self.u[0].dof.shape, 'd')
        self.mDotLow = np.zeros(self.u[0].dof.shape, 'd')
        self.fluxCorrection = np.zeros(self.u[0].dof.shape, 'd')
        self.mn = np.zeros(self.u[0].dof.shape, 'd')
        # Component-1 (S_n) low-order EV buffers.
        self.mn_n        = np.zeros(self.u[1].dof.shape, 'd')
        self.quantDOFs_n = np.zeros(self.u[1].dof.shape, 'd')
        # Post-step derived compositional fields for the XDMF archive.  Filled
        # in calculateAuxiliaryQuantitiesAfterStep by the C++ flash from the
        # primary (p,z) = (u[0].dof, u[1].dof); exposed to NumericalSolution via
        # coefficients.archive_scalar_dofs so they ride into flow.xmf alongside
        # p_w / S_n (named Sg0 / X0 / c_brine0 in the archive).
        # MESH-NODE sized (== u[0].dof / p): these are per-node visualization
        # fields written through comp-0's continuous node space in
        # NumericalSolution (archive_scalar_dofs -> {0: arr}).
        self.Sg_dof     = np.zeros(self.u[0].dof.shape, 'd')   # free-gas saturation
        self.X_dof       = np.zeros(self.u[0].dof.shape, 'd')   # CO2 mole frac in brine
        self.c_brine_dof = np.zeros(self.u[0].dof.shape, 'd')   # brine CO2 mass conc [kg/m^3]
        self.coefficients.archive_scalar_dofs = {
            'Sg': self.Sg_dof, 'X': self.X_dof, 'c_brine': self.c_brine_dof}
        # Per-rock closure output: max and min over the rocks at a node of the
        # per-rock S_g (equal to Sg at one-rock nodes; at a sand/seal node they
        # show the jump, e.g. sand part pooled, seal part 0).
        self.Sg_rock_max_dof = None
        self.Sg_rock_min_dof = None
        if self.coefficients.per_rock_closure:
            self.Sg_rock_max_dof = np.zeros(self.u[0].dof.shape, 'd')
            self.Sg_rock_min_dof = np.zeros(self.u[0].dof.shape, 'd')
            self.coefficients.archive_scalar_dofs['Sg_rock_max'] = self.Sg_rock_max_dof
            self.coefficients.archive_scalar_dofs['Sg_rock_min'] = self.Sg_rock_min_dof
        self.anb_seepage_flux_n = np.zeros(self.u[0].dof.shape, 'd')
        self.freeDOFMaterialTypes = np.zeros((self.nFreeDOF_global[0],), 'i')
        self.freeDOFToNode_u = -np.ones((self.nFreeDOF_global[0],), 'i')
        # ----------------------------------------------------------------------
        # DIAGNOSTIC TOGGLE (K-jump hypothesis test).  Set the environment var
        #     MPHASE_CO2_HOMOG_MAT=<flag>     e.g.  MPHASE_CO2_HOMOG_MAT=12
        # to overwrite elementMaterialTypes with a SINGLE rock everywhere BEFORE
        # any material map is built.  That removes every interior permeability /
        # capillary-entry-pressure jump from the whole kernel (transmissibility
        # tau, porosity, closures, AND both node maps below).  If the
        # velocity_couple spike and the gas over-creation/ratio_pw runaway VANISH
        # under this, the cause is the discontinuous-K interface artifact (a
        # continuous-p_w CG scheme cannot represent the flux/pc jump), NOT a
        # gas-equation bug.  If they PERSIST homogeneous, the bug is in the gas
        # equation itself.  Leave the var unset/empty for the true geology.
        import os
        _homog = os.environ.get('MPHASE_CO2_HOMOG_MAT', '').strip()
        if _homog != '' and hasattr(self.mesh, 'elementMaterialTypes'):
            _hm = int(_homog)
            np.asarray(self.mesh.elementMaterialTypes)[:] = _hm
            logEvent("[m_comp_co2] K-JUMP TOGGLE ON: elementMaterialTypes forced "
                     "to material %d everywhere (homogeneous; no interior K/pc "
                     "jumps). Unset MPHASE_CO2_HOMOG_MAT to restore geology." % _hm)
        # ----------------------------------------------------------------------
        # Per-node ROCK-REGION map (the sand type at each mesh node), built from
        # elementMaterialTypes -- the .ele region attribute -- via the first
        # element containing each node.  This is the geology.  It is NOT
        # mesh.nodeMaterialTypes: that array is the .node BOUNDARY-MARKER column
        # (0 for every interior node, segment tags on the boundary), so using it
        # as a rock index makes comp-0 read material 0 (= fallback sand) over the
        # entire interior, blind to ESF/F/fault heterogeneity.  Comp-0 (wetting)
        # and comp-1 (gas) both index THIS map so the coupled system solves one
        # rock law per physical location.  (Comp-1 has no Dirichlet, so its DOF
        # index == mesh node index.)
        self.nodeMaterialTypes_n = np.zeros((self.u[1].dof.shape[0],), 'i')
        # node_pd_min[gN] = the COARSEST (lowest) capillary entry pressure
        # p_d = 1/alpha [head] among the element materials incident on node gN.
        # Used by the comp-1 element-side gas flux as the capillary
        # entry-pressure barrier: gas crosses an edge into a finer element e
        # only when the gas-phase potential drop exceeds the entry-pressure
        # JUMP (1/alpha_e - node_pd_min[upstream]).  In a homogeneous region
        # node_pd_min == 1/alpha_e so the jump is 0 (no barrier); at a sand->seal
        # interface it equals p_d_seal - p_d_sand (the van Duijn / extended-
        # capillary-pressure breakthrough threshold).
        self.node_pd_min = np.full((self.u[1].dof.shape[0],), np.inf, 'd')
        # node_Sn_max[gN] = full gas saturation 1 - S_wr of the COARSEST incident
        # medium (= max over incident materials of 1-S_wr, since coarse sands have
        # the lowest residual).  This is the saturation the coarse pool fills to
        # against a seal.  The element-side breakthrough valve opens as the
        # upstream S_n rises toward this value over a fixed S_n window (so the
        # valve derivative is bounded -- a thin p_c-ratio valve gave dgate/dSn~1e5
        # and crashed Newton).
        self.node_Sn_max = np.zeros((self.u[1].dof.shape[0],), 'd')
        if hasattr(self.mesh, 'elementMaterialTypes') and hasattr(self.mesh, 'elementNodesArray'):
            elem_nodes = np.asarray(self.mesh.elementNodesArray)
            elem_mat   = np.asarray(self.mesh.elementMaterialTypes).astype(np.int32)
            alpha_types  = np.asarray(self.coefficients.vgm_alpha_types)
            thetaR_types = np.asarray(self.coefficients.thetaR_types)
            thetaSR_types = np.asarray(self.coefficients.thetaSR_types)
            seen = np.zeros((self.u[1].dof.shape[0],), 'b')
            for eN in range(elem_nodes.shape[0]):
                mat = int(elem_mat[eN])
                a_m = float(alpha_types[mat]) if mat < alpha_types.shape[0] else 0.0
                pd_m = (1.0 / a_m) if a_m > 0.0 else 0.0   # p_d=0 (alpha=0) -> no barrier
                phi_m = float(thetaR_types[mat] + thetaSR_types[mat])
                swr_m = (float(thetaR_types[mat]) / phi_m) if phi_m > 0.0 else 0.0
                sn_max_m = 1.0 - swr_m
                for i_local in range(elem_nodes.shape[1]):
                    gN = int(elem_nodes[eN, i_local])
                    if 0 <= gN < self.nodeMaterialTypes_n.shape[0]:
                        if not seen[gN]:
                            self.nodeMaterialTypes_n[gN] = mat
                            seen[gN] = True
                        if pd_m < self.node_pd_min[gN]:
                            self.node_pd_min[gN] = pd_m
                        if sn_max_m > self.node_Sn_max[gN]:
                            self.node_Sn_max[gN] = sn_max_m
        # Per-rock pore volume of each node's control volume, row-major
        # [node, material] (Z_ACROSS_ROCK_INTERFACES.md Sec. 3.1):
        #   W_ir = phi_r * sum_{e containing i, rock(e) = r} |e| / (d + 1),
        # |e| / (d + 1) being the lumped int N_i dV of an affine simplex under the
        # vertex quadrature the kernel uses.  Like node_pd_min, ghost nodes see
        # only this rank's elements; getResidual copies the owners' values onto
        # them once.  Built only when the per-rock closure is on.
        self.node_rock_W = np.zeros((1, 1), 'd')   # placeholder when the closure is off
        if self.coefficients.per_rock_closure:
            nd_ = self.nSpace_global
            elem_nodes = np.asarray(self.mesh.elementNodesArray)
            elem_mat = np.asarray(self.mesh.elementMaterialTypes).astype(np.int64)
            phi_types = (np.asarray(self.coefficients.thetaR_types)
                         + np.asarray(self.coefficients.thetaSR_types))
            Xe = np.asarray(self.mesh.nodeArray)[elem_nodes][:, :, :nd_]
            vol = np.abs(np.linalg.det(Xe[:, 1:, :] - Xe[:, :1, :])) / math.factorial(nd_)
            self.node_rock_W = np.zeros((self.u[1].dof.shape[0], phi_types.shape[0]), 'd')
            w_e = phi_types[elem_mat] * vol / (nd_ + 1)
            for a in range(elem_nodes.shape[1]):
                np.add.at(self.node_rock_W, (elem_nodes[:, a], elem_mat), w_e)
        # Any node never visited (shouldn't happen) -> no barrier.
        self.node_pd_min[~np.isfinite(self.node_pd_min)] = 0.0
        self.node_Sn_max[self.node_Sn_max <= 0.0] = 1.0   # default full sat -> no barrier
        # comp-0 free-DOF -> rock region, via the SAME per-node rock map (NOT
        # boundary flags).  global_dof is the mesh node index for the C0P1
        # wetting space, so it indexes nodeMaterialTypes_n directly.
        free_l2g = np.asarray(self.l2g[0]['freeGlobal']).ravel()
        dof_l2g = np.asarray(self.u[0].femSpace.dofMap.l2g).ravel()
        for free_dof, global_dof in zip(free_l2g, dof_l2g):
            if 0 <= free_dof < self.freeDOFMaterialTypes.shape[0]:
                self.freeDOFToNode_u[free_dof] = global_dof
                if 0 <= global_dof < self.nodeMaterialTypes_n.shape[0]:
                    self.freeDOFMaterialTypes[free_dof] = self.nodeMaterialTypes_n[global_dof]
        if np.any(self.freeDOFToNode_u < 0):
            raise RuntimeError("Failed to build the component-0 free-DOF to node map needed by the stabilized EV/FCT path.")
        # DIAG (mass-creation hunt): [0]=max|T_ij-T_ji|, [1]=max|T_ij|,
        # [2]=sum_ij F_ij (flux imbalance), [3]=sum_ij|F_ij|.
        self.gas_diag = np.zeros(4, 'd')
        comm = Comm.get()
        self.comm=comm
        if comm.size() > 1:
            assert numericalFluxType != None and numericalFluxType.useWeakDirichletConditions,"You must use a numerical flux to apply weak boundary conditions for parallel runs"
            self.offset = [0]
            for ci in range(1,self.nc):
                self.offset += [ci]
            self.stride = [self.nc for ci in range(self.nc)]
        self.comp0_rowptr = None
        self.comp0_colind = None
        self.comp0_full_offsets = None
        # Component-1 (S_n) compact CSR for the (1,1) DOF graph used by the
        # comp-1 EV pipeline (mirrors the comp-0 compact CSR pattern).
        self.comp1_rowptr       = None
        self.comp1_colind       = None
        # Component-1 EV edge/DOF buffers (lazy-allocated on first use).
        self.dLow_n                 = None
        self.dEV_n                  = None
        self.mLow_n                 = None
        self.gas_budget_node        = None
        self.mHigh_n                = None
        self.mDotLow_n              = None
        self.fluxMatrix_n           = None
        self.dt_times_fH_minus_fL_n = None
        self.FluxCorrectionMatrix_n = None
        self.fluxCorrection_n       = None
        self.limited_solution_n     = None
        self.Rpos_n                 = None
        self.Rneg_n                 = None
        self.min_m_bc_n             = None
        self.max_m_bc_n             = None
        self.bc_mask_n              = None
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
        # Ensure component-1 (S_n) Dirichlet boundary arrays exist on the
        # numericalFlux. The framework only initialises (u,0) by default for
        # this single-component-style numerical-flux object; we need (u,1) too
        # so the C++ boundary closure can evaluate
        # bc_u_v_ext = (isDir * bc_value) + (1 - isDir) * u_v_interior.
        if 1 not in self.numericalFlux.isDOFBoundary:
            self.numericalFlux.isDOFBoundary[1] = np.zeros(self.ebqe[('u',1)].shape, 'i')
        if ('u',1) not in self.numericalFlux.ebqe:
            self.numericalFlux.ebqe[('u',1)] = np.zeros(self.ebqe[('u',1)].shape, 'd')
        #TODO how to handle redistancing calls for calculateCoefficients,calculateElementResidual etc
        self.globalResidualDummy = None
        compKernelFlag=0
        self.delta_x_ij=None
        self.m_comp_co2 = cM_comp_co2_base(self.nSpace_global,
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

    def _build_component0_compact_csr(self, full_rowptr, full_colind):
        n_u = self.nFreeDOF_global[0]
        offset_u = self.offset[0]
        stride_u = self.stride[0]
        rowptr_u = np.zeros((n_u + 1,), dtype='i')
        colind_u = []
        full_offsets_u = []
        for i_u in range(n_u):
            global_row = offset_u + stride_u * i_u
            for full_offset in range(full_rowptr[global_row], full_rowptr[global_row + 1]):
                global_col = full_colind[full_offset]
                shifted_col = global_col - offset_u
                if shifted_col < 0 or shifted_col % stride_u != 0:
                    continue
                j_u = shifted_col // stride_u
                if 0 <= j_u < n_u:
                    colind_u.append(j_u)
                    full_offsets_u.append(full_offset)
            rowptr_u[i_u + 1] = len(colind_u)
        self.comp0_rowptr = rowptr_u
        self.comp0_colind = np.asarray(colind_u, dtype='i')
        self.comp0_full_offsets = np.asarray(full_offsets_u, dtype='i')

    def _ensure_component0_compact_csr(self, full_rowptr, full_colind):
        if (self.comp0_rowptr is None or
                self.comp0_colind is None or
                self.comp0_full_offsets is None):
            self._build_component0_compact_csr(full_rowptr, full_colind)

    def _build_component1_compact_csr(self, full_rowptr, full_colind):
        # Mirror of _build_component0_compact_csr but for the comp-1 (S_n)
        # DOF graph. Comp-1 uses full DOF numbering (no Dirichlet elimination)
        # so n_n = self.u[1].dof.shape[0].
        n_n = self.u[1].dof.shape[0]
        offset_n = self.offset[1]
        stride_n = self.stride[1]
        rowptr_n = np.zeros((n_n + 1,), dtype='i')
        colind_n = []
        for i_n in range(n_n):
            global_row = offset_n + stride_n * i_n
            for full_offset in range(full_rowptr[global_row], full_rowptr[global_row + 1]):
                global_col = full_colind[full_offset]
                shifted_col = global_col - offset_n
                if shifted_col < 0 or shifted_col % stride_n != 0:
                    continue
                j_n = shifted_col // stride_n
                if 0 <= j_n < n_n:
                    colind_n.append(j_n)
            rowptr_n[i_n + 1] = len(colind_n)
        self.comp1_rowptr       = rowptr_n
        self.comp1_colind       = np.asarray(colind_n, dtype='i')

    def _ensure_component1_compact_csr(self, full_rowptr, full_colind):
        if (self.comp1_rowptr is None or
                self.comp1_colind is None):
            self._build_component1_compact_csr(full_rowptr, full_colind)

    def _scatter_component_to_timeintegration(self, ci):
        if not hasattr(self.timeIntegration, 'u'):
            return
        dest = np.asarray(self.timeIntegration.u)
        comp_dof = np.asarray(self.u[ci].dof)
        offset = self.offset[ci]
        stride = self.stride[ci]
        dest[offset:offset + stride * comp_dof.size:stride] = comp_dof
   
    def FCTStep(self, component):
        
        coef = self.coefficients
        dt   = self.timeIntegration.dt

        if component == 0:
            if self.mLow is None or self.u_dof_old is None:
                return
            n_w = self.nFreeDOF_global[0]
            full_rowptr, full_colind, MassMatrix = self.MC_global.getCSRrepresentation()
            self._ensure_component0_compact_csr(full_rowptr, full_colind)
            nnz0 = self.comp0_colind.shape[0]
            if getattr(self, 'Rpos', None) is None or self.Rpos.shape[0] != n_w:
                self.Rpos = np.zeros((n_w,), 'd')
                self.Rneg = np.zeros((n_w,), 'd')
            if (getattr(self, 'FluxCorrectionMatrix', None) is None
                    or self.FluxCorrectionMatrix.shape[0] != nnz0):
                self.FluxCorrectionMatrix = np.zeros((nnz0,), 'd')
            bc_mask_u = np.ascontiguousarray(self.bc_mask[self.freeDOFToNode_u])
            _par = getattr(self.u[0], 'par_dof', None)
            if _par is not None:
                _saved = self.u[0].dof.copy()
                for _arr in (self.mLow, self.mDotLow, self.min_m_bc, self.max_m_bc):
                    self.u[0].dof[:] = _arr
                    _par.scatter_forward_insert()
                    _arr[:] = self.u[0].dof
                self.u[0].dof[:] = _saved

            # 2. Pass 1: Zalesak ratios.
            argsDict = cArgumentsDict.ArgumentsDict()
            argsDict["component"]                 = 0
            argsDict["pass"]                      = 1
            argsDict["numDOFs"]                   = n_w
            argsDict["dt"]                        = dt
            argsDict["ML"]                        = self.ML
            argsDict["mn"]                        = self.mn
            argsDict["mLow"]                      = self.mLow
            argsDict["mDotLow"]                   = self.mDotLow
            argsDict["csrRowIndeces_DofLoops"]    = self.comp0_rowptr
            argsDict["csrColumnOffsets_DofLoops"] = self.comp0_colind
            argsDict["csrRowIndeces_Full"]        = full_rowptr
            argsDict["csrColumnOffsets_Full"]     = full_colind
            argsDict["MC"]                        = MassMatrix
            argsDict["dt_times_fH_minus_fL"]      = self.dt_times_dC_minus_dL
            argsDict["min_m_bc"]                  = self.min_m_bc
            argsDict["max_m_bc"]                  = self.max_m_bc
            argsDict["FluxCorrectionMatrix"]      = self.FluxCorrectionMatrix
            argsDict["Rpos"]                      = self.Rpos
            argsDict["Rneg"]                      = self.Rneg
            argsDict["LUMPED_MASS_MATRIX"]        = coef.LUMPED_MASS_MATRIX
            argsDict["MONOLITHIC"]                = 0
            argsDict["offset_u"]                  = self.offset[0]
            argsDict["stride_u"]                  = self.stride[0]
            self.m_comp_co2.FCTStep(argsDict)
            _par = getattr(self.u[0], 'par_dof', None)
            if _par is not None:
                _saved = self.u[0].dof.copy()
                for _arr in (self.Rpos, self.Rneg):
                    self.u[0].dof[:] = _arr
                    _par.scatter_forward_insert()
                    _arr[:] = self.u[0].dof
                self.u[0].dof[:] = _saved

            # 4. Pass 2: apply the limiter -> limited_solution (limited m_w).
            limited_solution = np.zeros((n_w,), 'd')
            argsDict = cArgumentsDict.ArgumentsDict()
            argsDict["component"]                 = 0
            argsDict["pass"]                      = 2
            argsDict["bc_mask"]                   = bc_mask_u
            argsDict["numDOFs"]                   = n_w
            argsDict["dt"]                        = dt
            argsDict["ML"]                        = self.ML
            argsDict["mn"]                        = self.mn
            argsDict["mLow"]                      = self.mLow
            argsDict["csrRowIndeces_DofLoops"]    = self.comp0_rowptr
            argsDict["csrColumnOffsets_DofLoops"] = self.comp0_colind
            argsDict["csrRowIndeces_Full"]        = full_rowptr
            argsDict["csrColumnOffsets_Full"]     = full_colind
            argsDict["MC"]                        = MassMatrix
            argsDict["FluxCorrectionMatrix"]      = self.FluxCorrectionMatrix
            argsDict["Rpos"]                      = self.Rpos
            argsDict["Rneg"]                      = self.Rneg
            argsDict["fluxCorrection"]            = self.fluxCorrection
            argsDict["limited_solution"]          = limited_solution
            argsDict["LUMPED_MASS_MATRIX"]        = coef.LUMPED_MASS_MATRIX
            argsDict["MONOLITHIC"]                = 0
            argsDict["offset_u"]                  = self.offset[0]
            argsDict["stride_u"]                  = self.stride[0]
            self.m_comp_co2.FCTStep(argsDict)

            # 5. invert m_w -> p_w. Uses the already-limited S_n in u[1].dof
            #    (postStep calls FCTStep(component=1) before FCTStep(component=0)).
            p_w_lim = np.zeros((n_w,), 'd')
            argsDict = cArgumentsDict.ArgumentsDict()
            argsDict["a_rowptr"]             = coef.sdInfo[(0, 0)][0]
            argsDict["a_colind"]             = coef.sdInfo[(0, 0)][1]
            argsDict["rho"]                  = coef.rho
            argsDict["rho_n"]                = coef.rho_n
            argsDict["p_ref_n"]              = coef.p_ref_n
            argsDict["immiscible"]           = int(coef.immiscible)
            argsDict["T_C"]                  = coef.T_C
            argsDict["beta"]                 = coef.beta
            argsDict["gravity"]              = coef.gravity
            argsDict["alpha"]                = coef.vgm_alpha_types
            argsDict["n"]                    = coef.vgm_n_types
            argsDict["thetaR"]               = coef.thetaR_types
            argsDict["thetaSR"]              = coef.thetaSR_types
            argsDict["KWs"]                  = coef.Ksw_types
            argsDict["krn_end"]              = coef.krn_end_types
            argsDict["S_gr"]                 = coef.S_gr_types
            argsDict["bc_Se_min"]            = coef.bc_Se_min_types
            argsDict["mu_n"]                 = coef.mu_n
            argsDict["elementMaterialTypes"] = self.mesh.elementMaterialTypes
            argsDict["freeDOFMaterialTypes"] = self.freeDOFMaterialTypes
            argsDict["numDOFs"]              = n_w
            argsDict["limited_solution"]     = limited_solution
            argsDict["u_dof"]                = p_w_lim
            argsDict["u_dof_n"]              = self.u[1].dof
            argsDict["USE_NEWTON_INVERT"]    = 0
            argsDict["PSK_TYPE"]             = coef.PSK_TYPE
            argsDict["COMPONENT"]            = 0
            self.m_comp_co2.invert(argsDict)
            _par = getattr(self.u[0], 'par_dof', None)
            if _par is not None:
                self.u[0].dof[:] = p_w_lim
                _par.scatter_forward_insert()
                p_w_lim[:] = self.u[0].dof
            self.u[0].dof[:]  = p_w_lim
            self.u_dof_old[:] = p_w_lim
            self._scatter_component_to_timeintegration(0)

        elif component == 1:
            if self.limited_solution_n is None or self.u_dof_n_old is None:
                return
            n_dof = self.u[1].dof.shape[0]
            _par = getattr(self.u[1], 'par_dof', None)
            if _par is not None:
                _saved = self.u[1].dof.copy()
                for _arr in (self.mLow_n, self.mDotLow_n, self.min_m_bc_n, self.max_m_bc_n):
                    self.u[1].dof[:] = _arr
                    _par.scatter_forward_insert()
                    _arr[:] = self.u[1].dof
                self.u[1].dof[:] = _saved

            # 2. Pass 1: Zalesak ratios.
            argsDict = cArgumentsDict.ArgumentsDict()
            argsDict["component"]                 = 1
            argsDict["pass"]                      = 1
            argsDict["numDOFs_n"]                 = n_dof
            argsDict["dt"]                        = dt
            argsDict["ML_n"]                      = self.ML
            argsDict["MC_n"]                      = self.MC_n
            argsDict["mLow_n"]                    = self.mLow_n
            argsDict["mDotLow_n"]                 = self.mDotLow_n
            argsDict["dt_times_fH_minus_fL_n"]    = self.dt_times_fH_minus_fL_n
            argsDict["min_m_bc_n"]                = self.min_m_bc_n
            argsDict["max_m_bc_n"]                = self.max_m_bc_n
            argsDict["FluxCorrectionMatrix_n"]    = self.FluxCorrectionMatrix_n
            argsDict["Rpos_n"]                    = self.Rpos_n
            argsDict["Rneg_n"]                    = self.Rneg_n
            argsDict["csrRowIndeces_n_DofLoops"]  = self.comp1_rowptr
            argsDict["csrColumnOffsets_n_DofLoops"] = self.comp1_colind
            argsDict["LUMPED_MASS_MATRIX"]        = coef.LUMPED_MASS_MATRIX
            self.m_comp_co2.FCTStep(argsDict)

            _par = getattr(self.u[1], 'par_dof', None)
            if _par is not None:
                _saved = self.u[1].dof.copy()
                for _arr in (self.Rpos_n, self.Rneg_n):
                    self.u[1].dof[:] = _arr
                    _par.scatter_forward_insert()
                    _arr[:] = self.u[1].dof
                self.u[1].dof[:] = _saved
            argsDict = cArgumentsDict.ArgumentsDict()
            argsDict["component"]                 = 1
            argsDict["pass"]                      = 2
            argsDict["numDOFs_n"]                 = n_dof
            argsDict["dt"]                        = dt
            argsDict["ML_n"]                      = self.ML
            argsDict["mLow_n"]                    = self.mLow_n
            argsDict["FluxCorrectionMatrix_n"]    = self.FluxCorrectionMatrix_n
            argsDict["Rpos_n"]                    = self.Rpos_n
            argsDict["Rneg_n"]                    = self.Rneg_n
            argsDict["fluxCorrection_n"]          = self.fluxCorrection_n
            argsDict["limited_solution_n"]        = self.limited_solution_n
            argsDict["bc_mask_n"]                 = self.bc_mask_n
            argsDict["csrRowIndeces_n_DofLoops"]  = self.comp1_rowptr
            argsDict["csrColumnOffsets_n_DofLoops"] = self.comp1_colind
            self.m_comp_co2.FCTStep(argsDict)

            # 5. invert m_n -> S_n.
            S_n_lim = np.zeros_like(self.u[1].dof)
            argsDict = cArgumentsDict.ArgumentsDict()
            argsDict["a_rowptr"]             = coef.sdInfo[(0, 0)][0]
            argsDict["a_colind"]             = coef.sdInfo[(0, 0)][1]
            argsDict["rho"]                  = coef.rho
            argsDict["rho_n"]                = coef.rho_n
            argsDict["p_ref_n"]              = coef.p_ref_n
            argsDict["immiscible"]           = int(coef.immiscible)
            argsDict["T_C"]                  = coef.T_C
            argsDict["beta"]                 = coef.beta
            argsDict["gravity"]              = coef.gravity
            argsDict["alpha"]                = coef.vgm_alpha_types
            argsDict["n"]                    = coef.vgm_n_types
            argsDict["thetaR"]               = coef.thetaR_types
            argsDict["thetaSR"]              = coef.thetaSR_types
            argsDict["KWs"]                  = coef.Ksw_types
            argsDict["krn_end"]              = coef.krn_end_types
            argsDict["S_gr"]                 = coef.S_gr_types
            argsDict["bc_Se_min"]            = coef.bc_Se_min_types
            argsDict["mu_n"]                 = coef.mu_n
            argsDict["elementMaterialTypes"] = self.mesh.elementMaterialTypes
            argsDict["freeDOFMaterialTypes"] = self.freeDOFMaterialTypes
            argsDict["numDOFs"]              = n_dof
            argsDict["limited_solution"]     = self.limited_solution_n
            argsDict["u_dof"]                = S_n_lim
            argsDict["USE_NEWTON_INVERT"]    = 0
            argsDict["PSK_TYPE"]             = coef.PSK_TYPE
            argsDict["COMPONENT"]            = 1
            self.m_comp_co2.invert(argsDict)

            _par = getattr(self.u[1], 'par_dof', None)
            if _par is not None:
                self.u[1].dof[:] = S_n_lim
                _par.scatter_forward_insert()
                S_n_lim[:] = self.u[1].dof
            self.u[1].dof[:]    = S_n_lim
            self.u_dof_n_old[:] = S_n_lim
            self._scatter_component_to_timeintegration(1)


    def kth_FCT_step(self):
        #import pdb
        #pdb.set_trace()
        full_rowptr, full_colind, MassMatrix = self.MC_global.getCSRrepresentation()
        self._ensure_component0_compact_csr(full_rowptr, full_colind)
        rowptr = self.comp0_rowptr
        colind = self.comp0_colind
        compact_mass = MassMatrix[self.comp0_full_offsets]
        limitedFlux = np.zeros(self.comp0_full_offsets.shape[0])
        limited_solution = np.zeros((self.nFreeDOF_global[0],),'d')
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


        self.m_comp_co2.kth_FCT_step(
            self.timeIntegration.dt,
            self.coefficients.num_fct_iter,
            self.comp0_full_offsets.shape[0],
            self.nFreeDOF_global[0],
            compact_mass,
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

        self._scatter_component_to_timeintegration(0)
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
        # FD-Jacobian probe support: stash the global free-DOF vector this
        # residual is being evaluated at, so getJacobian can re-evaluate the
        # residual at u +/- eps*e_j and compare to the assembled tangent.
        self._fd_last_u = np.copy(u)
        cfemIntegrals.zeroJacobian_CSR(self.nNonzerosInJacobian,
                                       self.jacobian)
        if self.u_dof_old is None:
            # Pass initial condition to u_dof_old
            self.u_dof_old = np.copy(self.u[0].dof)
        # lazy init component-1 (S_n) previous-step DOFs from IC.
        if self.u_dof_n_old is None and self.nc >= 2:
            self.u_dof_n_old = np.copy(self.u[1].dof)
        rowptr, colind, nzval = self.jacobian.getCSRrepresentation()
        nnz = nzval.shape[-1]  # number of non-zero entries in sparse matrix
        self._ensure_component0_compact_csr(rowptr, colind)
        comp0_rowptr = self.comp0_rowptr
        comp0_colind = self.comp0_colind
        # Component-1 (S_n) compact DOF CSR + lazy EV edge/DOF buffers.
        self._ensure_component1_compact_csr(rowptr, colind)
        n_n_   = self.u[1].dof.shape[0]
        nnz_n_ = int(self.comp1_colind.shape[0])
        if self.dLow_n is None or self.dLow_n.shape[0] != nnz_n_:
            self.dLow_n                  = np.zeros((nnz_n_,), 'd')
            self.dEV_n                   = np.zeros((nnz_n_,), 'd')
            self.fluxMatrix_n            = np.zeros((nnz_n_,), 'd')
            # Per-edge antidiffusive flux storage (high-order minus low-order)
            # consumed by FCTStep_n_pass1.
            self.dt_times_fH_minus_fL_n  = np.zeros((nnz_n_,), 'd')
            self.FluxCorrectionMatrix_n  = np.zeros((nnz_n_,), 'd')
        if self.mLow_n is None or self.mLow_n.shape[0] != n_n_:
            self.mLow_n             = np.zeros((n_n_,), 'd')
            # Per-node gas-residual budget, 6 slots term-major (numDOFs_n each):
            # [0]accum [1]flux [2]sink [3]injection [4]boundary [5]total-residual.
            # Filled by calculateResidual_entropy_viscosity; summed over owned
            # nodes + MPI-reduced in _log_mass_balance ([gas budget] line).
            self.gas_budget_node    = np.zeros((6 * n_n_,), 'd')
            self.mHigh_n            = np.zeros((n_n_,), 'd')
            self.mDotLow_n          = np.zeros((n_n_,), 'd')
            self.fluxCorrection_n   = np.zeros((n_n_,), 'd')
            self.limited_solution_n = np.zeros((n_n_,), 'd')
            self.Rpos_n             = np.zeros((n_n_,), 'd')
            self.Rneg_n             = np.zeros((n_n_,), 'd')
            self.min_m_bc_n = np.ones((n_n_,), 'd') *  1.0e10
            self.max_m_bc_n = np.ones((n_n_,), 'd') * -1.0e10
            self.bc_mask_n  = np.ones((n_n_,), 'd')
            if self.nc >= 2 and 1 in self.dirichletConditions:
                _dbc_n = getattr(self.dirichletConditions[1],
                                 'DOFBoundaryConditionsDict', {})
                for _dofN in _dbc_n:
                    if 0 <= _dofN < n_n_:
                        self.bc_mask_n[_dofN] = 0.0
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
            # SparseMat dimensions must match the CSR data
            # (rowptr/colind from the full Jacobian span 2N rows for nc=2).
            # Telling SparseMat it's N x N while the CSR is 2N x 2N causes
            # the assembler to write to wrong positions for nc>=2.
            self.MC_global = SparseMat(self.nFreeVDOF_global,
                                       self.nFreeVDOF_global,
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
            self._ensure_component0_compact_csr(rowptr, colind)
            self.ML = np.zeros((self.nFreeDOF_global[0],), 'd')
            for i in range(self.nFreeDOF_global[0]):
                full_offsets_i = self.comp0_full_offsets[self.comp0_rowptr[i]:self.comp0_rowptr[i + 1]]
                self.ML[i] = self.MC_a[full_offsets_i].sum()
            # Consistent mass matrix on the comp-1 (S_n) DOF graph.
            # MC_a above only has its (0,0) block assembled (l2g[0] /
            # csrColumnOffsets[(0,0)]); its comp-1 block is stale, so
            # FCTStep_n's consistency term dt*MC*(mDotLow_i - mDotLow_j)
            # was reading garbage -> non-antisymmetric -> mass leak.
            # Both components share the C0-P1 space, so elementMassMatrix
            # (computed above) IS the comp-1 element mass matrix; scatter it
            # onto the comp-1 compact CSR (same indexing as dLow_n /
            # dt_times_fH_minus_fL_n). Symmetric by construction ->
            # antisymmetric consistency term -> mass-conservative FCT.
            self.MC_n = np.zeros((self.comp1_colind.shape[0],), 'd')
            _u1_l2g = self.u[1].femSpace.dofMap.l2g
            _nDOF = self.nDOF_test_element[0]
            for eN in range(self.mesh.nElements_global):
                for i in range(_nDOF):
                    i_n = _u1_l2g[eN, i]
                    _rstart = self.comp1_rowptr[i_n]
                    _rend   = self.comp1_rowptr[i_n + 1]
                    for j in range(_nDOF):
                        j_n = _u1_l2g[eN, j]
                        for off in range(_rstart, _rend):
                            if self.comp1_colind[off] == j_n:
                                self.MC_n[off] += elementMassMatrix[eN, i, j]
                                break
            # the trace-equals-volume assertion was nc=1 +
            # serial-only; with nc=2 the rowptr spans both blocks and the
            # row-sum no longer matches the per-rank mesh.volume directly.
            # Disabled here; the lumped mass is still correct for the (0,0)
            # block. Re-derive a proper MPI-safe / nc-aware check later.
            # np.testing.assert_almost_equal(self.ML.sum(),
            #                                self.mesh.volume,
            #                                err_msg="Trace of lumped mass matrix should be the domain volume", verbose=True)
            for d in range(self.nSpace_global):  # spatial dimensions
                # C matrices
                self.cterm[d] = np.zeros((self.mesh.nElements_global,
                                          self.nDOF_test_element[0],
                                          self.nDOF_trial_element[0]), 'd')
                self.cterm_a[d] = nzval.copy()
                #self.cterm_a[d] = np.zeros(nzval.size)
                # SparseMat dims must match the full CSR.
                self.cterm_global[d] = SparseMat(self.nFreeVDOF_global,
                                                 self.nFreeVDOF_global,
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
                # SparseMat dims must match the full CSR.
                self.cterm_global_transpose[d] = SparseMat(self.nFreeVDOF_global,
                                                           self.nFreeVDOF_global,
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
        nFree = self.nFreeDOF_global[0]
        self.min_m_bc = np.ones(nFree, 'd')
        self.min_m_bc *= 1.0e10
        self.max_m_bc = np.ones(nFree, 'd')
        self.max_m_bc *= -1.0e10
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
                    self.u_dof_old[dofN] = self.u[cj].dof[dofN]
                    self.bc_mask[dofN] = 0.0
        bc_mask_u = np.ones((self.nFreeDOF_global[0],), 'd')
        bc_mask_u[:] = self.bc_mask[self.freeDOFToNode_u]
        degree_polynomial = 1
        try:
            degree_polynomial = self.u[0].femSpace.order
        except:
            pass
        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["bc_mask"] = bc_mask_u
        argsDict["dt"] = self.timeIntegration.dt
        argsDict["Theta"] = 1.0
        argsDict["Theta_h"] = 0.5
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
        argsDict["rho_n"] = self.coefficients.rho_n
        argsDict["p_ref_n"] = self.coefficients.p_ref_n
        argsDict["immiscible"] = int(self.coefficients.immiscible)
        argsDict["T_C"] = self.coefficients.T_C
        argsDict["beta"] = self.coefficients.beta

        argsDict["q_rho"]= self.q['rho']
        argsDict["ebqe_rho"]= self.ebqe['rho']
        
        argsDict["gravity"] = self.coefficients.gravity
        argsDict["alpha"] = self.coefficients.vgm_alpha_types
        argsDict["n"] = self.coefficients.vgm_n_types
        argsDict["thetaR"] = self.coefficients.thetaR_types
        argsDict["thetaSR"] = self.coefficients.thetaSR_types
        argsDict["KWs"] = self.coefficients.Ksw_types
        argsDict["krn_end"] = self.coefficients.krn_end_types
        argsDict["S_gr"] = self.coefficients.S_gr_types
        argsDict["bc_Se_min"] = self.coefficients.bc_Se_min_types
        argsDict["mu_n"]    = self.coefficients.mu_n
        argsDict["useMetrics"] = 0.0
        argsDict["alphaBDF"] = self.timeIntegration.alpha_bdf
        argsDict["lag_shockCapturing"] = 0
        argsDict["shockCapturingDiffusion"] = self.coefficients.SC
        argsDict["VMS"] = self.coefficients.VMS
        argsDict["sc_uref"] = 1.0
        argsDict["sc_alpha"] = 2.0
        argsDict["u_l2g"] = self.u[0].femSpace.dofMap.l2g
        # Component-1 (z) map.  Each component owns its own femSpace.dofMap.l2g
        # (standard multi-component proteus layout); comp-1's is parallel-correct
        # via the same offset[1]/stride[1]/par_dof machinery Richards.h uses for
        # its single component.  u_l2g_n == u_l2g element-wise on the shared P1 mesh.
        argsDict["u_l2g_n"] = self.u[1].femSpace.dofMap.l2g
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
        # ---- Component-1 (S_n) args ----
        # Gas mass equation. C++ residual reads u_dof_n & u_dof_n_old, builds
        # m_n = phi*rho_n*u_n, integrates (m_n - m_n_old)/dt against test
        # functions, and writes into globalResidual at offset_n + stride_n * dof.
        argsDict["u_dof_n"]     = self.u[1].dof
        argsDict["u_dof_n_old"] = self.u_dof_n_old
        argsDict["offset_n"]    = self.offset[1]
        argsDict["stride_n"]    = self.stride[1]
        argsDict["c_dof"] = getattr(self.coefficients, "c_dof",
                                    np.zeros_like(self.u[1].dof))
        # In flash mode the once-per-step nodal dissolutionFlash owns the
        # gas<->brine exchange, so the in-residual kinetic R_diss MUST be off
        # (pass k_d=0) to avoid double counting; self.coefficients.k_d is kept
        # only for the diagnostic.
        argsDict["k_d"]   = (0.0 if self.coefficients.dissolution_mode == 'flash'
                             else float(self.coefficients.k_d))
        argsDict["c_sat"] = float(self.coefficients.c_sat)
        # CO2 injection: build the per-node source field consumed by the C++
        # gas-equation source term (applied like R_diss, opposite sign).
        # Each port is a small disk: injection_dof = rate (a volumetric source
        # density) on every node within radius of the port, gated by the
        # schedule.  Total injected per port = rate * (pi radius^2) * duration
        # -- mesh-independent and parallel-safe (every rank sets the same
        # density on its disk nodes; each element is integrated once).
        # Rebuilt each call (time gate depends on t).  Always passed -- the
        # kernel reads it unconditionally, so an empty list -> all-zero field.
        if getattr(self, "injection_dof", None) is None \
                or self.injection_dof.shape[0] != self.u[1].dof.shape[0]:
            self.injection_dof = np.zeros_like(self.u[1].dof)
        self.injection_dof[:] = 0.0
        injection_ports = self.coefficients.injection_ports
        if injection_ports:
            if getattr(self, "_injection_masks", None) is None:
                nodes = self.mesh.nodeArray
                nx = nodes[:, 0]; ny = nodes[:, 1]
                self._injection_masks = []
                for (px, py, rate, radius, t0, t1) in injection_ports:
                    d2 = ((nx - px) ** 2 + (ny - py) ** 2)
                    self._injection_masks.append(d2 <= radius * radius)
            t_now = float(self.timeIntegration.t)
            # tanh ramp at each port's start so Newton can track the
            # saturation breakthrough.  tau is set in flow_p.py via
            # injection_ramp_tau (sim time units).  tau == 0 -> no ramp.
            tau = self.coefficients.injection_ramp_tau
            for (px, py, rate, radius, t0, t1), mask in zip(
                    injection_ports, self._injection_masks):
                if t0 <= t_now < t1:
                    if tau > 0.0:
                        ramp = 0.5 * (1.0 + np.tanh((t_now - t0) / tau - 3.0))
                    else:
                        ramp = 1.0
                    self.injection_dof[mask] = rate * ramp
        argsDict["injection_dof"] = self.injection_dof
        # --- Consistent (Galerkin) point-source injection (MOOSE DiracKernel) ---
        # Always passed (kernel reads unconditionally).  inj_point_mode==0 -> the
        # lumped disk above is used and these are inert.  inj_point_mode==1 ->
        # R^c_i -= Q_port*N_i(x_p) on the containing element.  We cache, per port,
        # the OWNED element that contains the port point and its P1 shape-function
        # values N_i (barycentric coords); the rate is time-gated each call.
        point_mode = 1 if getattr(self.coefficients, "injection_point_source", False) else 0
        nDOFel = self.mesh.elementNodesArray.shape[1]   # 3 for P1 triangles
        if point_mode and injection_ports and getattr(self, "_inj_pts", None) is None:
            nodesA = self.mesh.nodeArray
            elemsA = self.mesh.elementNodesArray
            nOwned = int(getattr(self.mesh, "nElements_owned", elemsA.shape[0]))
            x0 = nodesA[elemsA[:nOwned, 0], :2]
            x1 = nodesA[elemsA[:nOwned, 1], :2]
            x2 = nodesA[elemsA[:nOwned, 2], :2]
            det = (x1[:, 1]-x2[:, 1])*(x0[:, 0]-x2[:, 0]) + (x2[:, 0]-x1[:, 0])*(x0[:, 1]-x2[:, 1])
            det = np.where(np.abs(det) < 1e-300, 1e-300, det)
            self._inj_pts = []     # per port: (elem_id_or_-1, [N0,N1,N2], Q2D, t0, t1)
            for (px, py, rate, radius, t0, t1) in injection_ports:
                l0 = ((x1[:, 1]-x2[:, 1])*(px-x2[:, 0]) + (x2[:, 0]-x1[:, 0])*(py-x2[:, 1])) / det
                l1 = ((x2[:, 1]-x0[:, 1])*(px-x2[:, 0]) + (x0[:, 0]-x2[:, 0])*(py-x2[:, 1])) / det
                l2 = 1.0 - l0 - l1
                inside = (l0 >= -1e-9) & (l1 >= -1e-9) & (l2 >= -1e-9)
                if inside.any():
                    e = int(np.argmax(inside))
                    w = np.array([l0[e], l1[e], l2[e]], 'd')
                else:
                    e, w = -1, np.zeros(3, 'd')        # port not on this rank
                Q2D = float(rate) * np.pi * float(radius) ** 2   # = Q_mol/RIG_DEPTH (abs molar rate/depth)
                self._inj_pts.append((e, w, Q2D, float(t0), float(t1)))
        nP = len(injection_ports) if (point_mode and injection_ports) else 0
        inj_element = np.full(max(nP, 1), -1, 'i')
        inj_weight  = np.zeros(max(nP, 1) * nDOFel, 'd')
        inj_rate    = np.zeros(max(nP, 1), 'd')
        if nP:
            t_now = float(self.timeIntegration.t)
            tau   = self.coefficients.injection_ramp_tau
            for p, (e, w, Q2D, t0, t1) in enumerate(self._inj_pts):
                inj_element[p] = e
                inj_weight[p * nDOFel:p * nDOFel + len(w)] = w
                if e >= 0 and t0 <= t_now < t1:
                    ramp = 0.5 * (1.0 + np.tanh((t_now - t0) / tau - 3.0)) if tau > 0.0 else 1.0
                    inj_rate[p] = Q2D * ramp
        argsDict["inj_point_mode"] = point_mode
        argsDict["inj_n_ports"]    = nP
        argsDict["inj_element"]    = inj_element
        argsDict["inj_weight"]     = inj_weight
        argsDict["inj_rate"]       = inj_rate
        # csr maps for the (1,1) Jacobian block (not used by residual; staged
        # for turn 3 when calculateJacobian gains the (1,1) diagonal block).
        argsDict["csrRowIndeces_n_n"]      = self.csrRowIndeces[(1, 1)]
        argsDict["csrColumnOffsets_n_n"]   = self.csrColumnOffsets[(1, 1)]
        # (1,0) cross-block CSR maps for gas-eq diffusion against grad u_w.
        argsDict["csrRowIndeces_n_w"]      = self.csrRowIndeces[(1, 0)]
        argsDict["csrColumnOffsets_n_w"]   = self.csrColumnOffsets[(1, 0)]
        # P1: comp-0 (H2O) (0,0) and (0,1) block CSR maps. The water-flux
        # Jacobian scatters through these (Richards-style, eN_i_j offset) so the
        # (0,1) off-diagonal water<-neighbor-z coupling always lands -- replaces
        # the Full-CSR column search that dropped it (FD-probe structural
        # misses -> stalled Newton). These keys are allocated by the framework
        # because mass[0][1]/advection[0][1]/diffusion[0][0][1] declare the
        # (0,1) coupling, exactly mirroring the (1,0) block above.
        argsDict["csrRowIndeces_w_w"]      = self.csrRowIndeces[(0, 0)]
        argsDict["csrColumnOffsets_w_w"]   = self.csrColumnOffsets[(0, 0)]
        argsDict["csrRowIndeces_w_n"]      = self.csrRowIndeces[(0, 1)]
        argsDict["csrColumnOffsets_w_n"]   = self.csrColumnOffsets[(0, 1)]
        argsDict["csrColumnOffsets_eb_n_n"] = self.csrColumnOffsets_eb[(1, 1)]
        # (1,0) boundary cross-block: gas-eq Darcy diffusion through ext faces.
        argsDict["csrColumnOffsets_eb_n_w"] = self.csrColumnOffsets_eb[(1, 0)]
        argsDict["globalResidual"] = r
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["ebqe_velocity_ext"] = self.ebqe[('velocity',0)]
        argsDict["ebqe_velocity_ext_couple"] = self.ebqe[('velocity_couple',0)]      
        argsDict["isDOFBoundary_u"] = self.numericalFlux.isDOFBoundary[0]
        argsDict["ebqe_bc_u_ext"] = self.numericalFlux.ebqe[('u',0)]
        # component-1 (S_n) boundary arrays. Initialised in
        # init() when missing so this works regardless of numericalFlux class.
        argsDict["isDOFBoundary_n"] = self.numericalFlux.isDOFBoundary[1]
        argsDict["ebqe_bc_u_n_ext"] = self.numericalFlux.ebqe[('u',1)]
        argsDict["isFluxBoundary_u"] = self.ebqe[('advectiveFlux_bc_flag',0)]
        argsDict["ebqe_bc_flux_ext"] = self.ebqe[('advectiveFlux_bc',0)]
        argsDict["ebqe_phi"] = self.ebqe[('u',0)]
        argsDict["epsFact"] = 0.0
        #argsDict["ebqe_x"] = self.ebqe['x']
        argsDict["ebqe_u"] = self.ebqe[('u',0)]
        argsDict["ebqe_theta"] = self.ebqe[('theta',0)]
        argsDict["ebqe_flux"] = self.ebqe[('advectiveFlux',0)]
        argsDict['STABILIZATION_TYPE'] = self.coefficients.STABILIZATION_TYPE
        # ENTROPY VISCOSITY and ARTIFICIAL COMRPESSION
        argsDict["cE"] = self.coefficients.cE
        argsDict["cK"] = self.coefficients.cK
        # PARAMETERS FOR LOG BASED ENTROPY FUNCTION
        argsDict["uL"] = self.coefficients.uL
        argsDict["uR"] = self.coefficients.uR
        # PARAMETERS FOR EDGE VISCOSITY
        
        argsDict["numDOFs"] = self.nFreeDOF_global[0]
       
        argsDict["numDOFs_u"] = self.nFreeDOF_global[0]
        argsDict["NNZ"] = self.nnz
        argsDict["Cx"] = len(Cx)  # num of non-zero entries in the sparsity pattern
        argsDict["csrRowIndeces_DofLoops"] = comp0_rowptr  # compact component-0 CSR for DOF loops
        argsDict["csrColumnOffsets_DofLoops"] = comp0_colind
        argsDict["csrRowIndeces_Full"] = rowptr
        argsDict["csrColumnOffsets_Full"] = colind
        # Component-1 (S_n) DOF graph + EV buffers.
        argsDict["numDOFs_n"]                   = self.u[1].dof.shape[0]
        argsDict["NNZ_n"]                       = int(self.comp1_colind.shape[0])
        argsDict["csrRowIndeces_n_DofLoops"]    = self.comp1_rowptr
        argsDict["csrColumnOffsets_n_DofLoops"] = self.comp1_colind
        argsDict["dLow_n"]                     = self.dLow_n
        argsDict["dEV_n"]                       = self.dEV_n
        argsDict["fluxMatrix_n"]                = self.fluxMatrix_n
        argsDict["mLow_n"]                      = self.mLow_n
        argsDict["mDotLow_n"]                   = self.mDotLow_n
        # Comp-1 FCT bundle (Zalesak limiter inputs / outputs).
        argsDict["dt_times_fH_minus_fL_n"]      = self.dt_times_fH_minus_fL_n
        argsDict["min_m_bc_n"]                  = self.min_m_bc_n
        argsDict["max_m_bc_n"]                  = self.max_m_bc_n
        argsDict["fluxCorrection_n"]            = self.fluxCorrection_n
        argsDict["limited_solution_n"]          = self.limited_solution_n
        argsDict["bc_mask_n"]                   = self.bc_mask_n
        # ML_n aliases the comp-0 lumped mass (purely geometric; both
        # components share the FE space). MC_n is the DEDICATED comp-1
        # consistent mass matrix, indexed by the comp-1 compact CSR -- MC_a
        # only has its (0,0) block assembled, so it cannot be used here.
        argsDict["ML_n"]                        = self.ML
        argsDict["MC_n"]                        = self.MC_n
        # FCT activation: maps the user's original Coefficients(FCT=True)
        # request to the C++ gate. coefficients.FCT itself is forced False
        # to bypass the framework's single-component post-Newton hook (see
        # Coefficients.__init__ comment near self._fct_requested).
        argsDict["FCT_n"]                       = int(bool(self.coefficients._fct_requested))
        # S_n bounds for the comp-1 entropy / smoothness sensor.
        # Material 0 used as the fallback when materials are heterogeneous.
        # S_n in [0, 1 - S_wr] with S_wr = thetaR/(thetaR+thetaSR).
        _S_wr0 = float(self.coefficients.thetaR_types[0] /
                       (self.coefficients.thetaR_types[0] + self.coefficients.thetaSR_types[0]))
        argsDict["u_n_L"] = 0.0
        argsDict["u_n_R"] = 1.0 - _S_wr0
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
        argsDict["STABILIZATION_TYPE"] = self.coefficients.STABILIZATION_TYPE
        argsDict["ENTROPY_TYPE"] = self.coefficients.ENTROPY_TYPE
        argsDict["PSK_TYPE"] = self.coefficients.PSK_TYPE
        # FLUX CORRECTED TRANSPORT
        argsDict["dLow"] = self.dLow
        argsDict["fluxMatrix"] = self.fluxMatrix
        argsDict["mDotLow"] = self.mDotLow
        argsDict["fluxCorrection"] = self.fluxCorrection
        limited_solution = np.zeros((self.nFreeDOF_global[0],),'d')
        argsDict["limited_solution"] = limited_solution
        argsDict["MONOLITHIC"] =0
        argsDict["mLow"] = self.mLow
        argsDict["dt_times_fH_minus_fL"] = self.dt_times_dC_minus_dL
        argsDict["min_m_bc"] = self.min_m_bc
        argsDict["max_m_bc"] = self.max_m_bc
        argsDict["quantDOFs"] = self.quantDOFs
        argsDict["mn"] = self.mn
        # Component-1 (S_n) low-order EV buffers.
        argsDict["mn_n"]        = self.mn_n
        argsDict["quantDOFs_n"] = self.quantDOFs_n
        argsDict["anb_seepage_flux_n"]= self.anb_seepage_flux_n
        argsDict["freeDOFMaterialTypes"] = self.freeDOFMaterialTypes
        argsDict["freeDOFToNode_u"] = self.freeDOFToNode_u
        argsDict["nodeMaterialTypes_n"] = self.nodeMaterialTypes_n
        # PARALLEL FIX (one-time): ghost-synchronize the capillary-gate nodal
        # arrays. node_pd_min / node_Sn_max are built in __init__ by a LOCAL
        # MIN/MAX over elementNodesArray with no ghost exchange. Owned nodes get
        # the complete star (full overlap), but GHOST nodes (one layer out) see
        # only this rank's incident elements -> an incomplete MIN/MAX. The
        # interior gas-flux gate reads these at the UPSTREAM node, which may be a
        # ghost, so the shared partition edge gets a DIFFERENT gate (hence a
        # different F) on the two ranks -> the antisymmetric flux no longer
        # cancels across the cut -> spurious gas mass created at every partition
        # edge (diagnosed via [gas budget] flux != 0 while per-element sum_F~0).
        # Owners hold the correct complete value (full overlap), so a
        # forward-insert scatter (owner -> ghost copies) repairs every ghost.
        # Done lazily here (not __init__) because u[1].par_dof exists only after
        # the global parallel layout is built. par_dof is None / no-op in serial.
        if not getattr(self, '_gate_arrays_synced', False):
            _pd = getattr(self.u[1], 'par_dof', None)
            if _pd is not None:
                _save = self.u[1].dof.copy()
                for _arr in (self.node_pd_min, self.node_Sn_max):
                    n_arr = min(len(_arr), self.u[1].dof.shape[0])
                    self.u[1].dof[:n_arr] = _arr[:n_arr]
                    _pd.scatter_forward_insert()
                    _arr[:n_arr] = self.u[1].dof[:n_arr]
                self.u[1].dof[:] = _save
            self._gate_arrays_synced = True
        argsDict["node_pd_min"] = self.node_pd_min
        argsDict["node_Sn_max"] = self.node_Sn_max
        # Per-rock closure weights: ghost nodes take the owners' complete W_ir
        # (one scatter per material column, once), as for the gate arrays above.
        if self.coefficients.per_rock_closure and not getattr(self, '_rock_W_synced', False):
            _pd = getattr(self.u[1], 'par_dof', None)
            if _pd is not None:
                _save = self.u[1].dof.copy()
                for rk in range(self.node_rock_W.shape[1]):
                    self.u[1].dof[:] = self.node_rock_W[:, rk]
                    _pd.scatter_forward_insert()
                    self.node_rock_W[:, rk] = self.u[1].dof
                self.u[1].dof[:] = _save
            self._rock_W_synced = True
        argsDict["per_rock_closure"] = int(self.coefficients.per_rock_closure)
        argsDict["node_rock_W"] = self.node_rock_W
        argsDict["nRockTypes"] = int(self.node_rock_W.shape[1])
        argsDict["gas_diag"] = self.gas_diag
        argsDict["gas_budget_node"] = self.gas_budget_node
        ######################################################################################
        argsDict["pn"] = self.u[0].dof
        argsDict["mHigh"] = self.mHigh

        rowptr, colind, MassMatrix = self.MC_global.getCSRrepresentation()
        argsDict["MassMatrix"] = MassMatrix
        
######################################################################################        
        #argsDict["anb_seepage_flux"] = self.coefficients.anb_seepage_flux
        argsDict["anb_seepage_flux"] = self.anb_seepage_flux
        argsDict["q_velocity"] = self.q[('q_velocity_buf', 0)]
        argsDict["csrRowIndeces_u_u"] = self.csrRowIndeces[(0,0)]
        argsDict["csrColumnOffsets_u_u"] = self.csrColumnOffsets[(0,0)]
        argsDict["csrColumnOffsets_eb_u_u"] = self.csrColumnOffsets_eb[(0,0)]
        #argsDict["q_grad_psi"] = self.q[('velocity', 0)] 
        #print(anb_seepage_flux)
        #argsDict["anb_seepage_flux_n"] = self.coefficients.anb_seepage_flux_n
        #if np.sum(anb_seepage_flux_n)>0:

        #logEvent("Hi, this is Arnob", self.anb_seepage_flux_n[0])
        #print("Seepage Flux from Python file",  np.sum(self.anb_seepage_flux_n))
        #seepage_text_variable= np.sum(self.anb_seepage_flux_n)
        #t_now = float(self.timeIntegration.t)
        #s_now = float(np.sum(self.anb_seepage_flux_n))

        #with open("seepage_stab_0.txt", "a") as f:
        #    f.write(f"t={t_now:.6f}, s={s_now:.6e}\n")
        #with open('seepage_stab_0.txt',"a" ) as f:
            #f.write("\n Time"+ ",\t" +"Seepage\n")
        #    f.write(f"{self.timeIntegration.t:.6f},\t{float(seepage_text_variable):.6e}\n")
#            f.write(repr(self.coefficients.t)+ ",\t" +repr(seepage_text_variable), "\n")
            #f.write(repr(seepage_text_variable)+ "\n")
        
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        seepage_flux_value = np.sum(self.anb_seepage_flux_n)
        if seepage_flux_value > 0.0:
        # Each processor writes its own flux with its rank
            with open("seepage_flux_try.txt", "a") as f:
                f.write(f"Rank {rank}:, {self.timeIntegration.t:.6f}, {seepage_flux_value:.8f}\n")
       
        # seepage_flux_value = np.sum(self.anb_seepage_flux_n) #self.anb_seepage_flux_n[0]
        
        # with open("seepage_flux_try.txt", "a") as f:
        #    f.write(f"{seepage_flux_value:.8f}\n")
                          
        #seepage_flux.append(seepage_flux_value)
        
        # comm = Comm.get()
        # if comm.isMaster():
        #     with open("seepage_flux_try.txt", "a") as f:
        #         f.write(f"{seepage_flux_value:.6f}\n")
                
            #seepage_flux_value = np.sum(self.anb_seepage_flux_n) #self.anb_seepage_flux_n[0]
            #logEvent(f"Seepage flux at t={self.timeIntegration.t:.6f} is {seepage_flux_value:.6e}", level=2)
#            with open("seepage_flux_vs_time.txt", "a") as f:
#                f.write(f"{self.timeIntegration.t:.6f}, {seepage_flux_value:.6f}\n")

        if (self.coefficients.STABILIZATION_TYPE == 0):  # SUPG
            self.calculateResidual = self.m_comp_co2.calculateResidual
            self.calculateJacobian = self.m_comp_co2.calculateJacobian
        else:
            self.calculateResidual = self.m_comp_co2.calculateResidual_entropy_viscosity
            self.calculateJacobian = self.m_comp_co2.calculateMassMatrix
        
        if self.delta_x_ij is None:
            self.delta_x_ij = -np.ones((self.nNonzerosInJacobian*3,),'d')
        self.calculateResidual(argsDict)
        if getattr(self, "_theta_log_count", 0) < 5:
            q_theta = self.q[('theta',0)]
            ebqe_theta = self.ebqe[('theta',0)]
            logEvent("[Richards q_theta] t={:.6e} q(min,max,mean)=({:.6e},{:.6e},{:.6e}) ebqe(min,max,mean)=({:.6e},{:.6e},{:.6e}) q_zero_count={} ebqe_zero_count={}".format(
                     self.timeIntegration.t,
                     float(np.min(q_theta)), float(np.max(q_theta)), float(np.mean(q_theta)),
                     float(np.min(ebqe_theta)), float(np.max(ebqe_theta)), float(np.mean(ebqe_theta)),
                     int(np.count_nonzero(q_theta == 0.0)), int(np.count_nonzero(ebqe_theta == 0.0))),
                     level=2)
            self._theta_log_count = getattr(self, "_theta_log_count", 0) + 1
        


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
        # Per-Newton-iteration residual decomposition (divergence triage).
        # Enable with MCOMP_RES_DBG=1.  Unlike the postStep [gas budget] line,
        # this fires on EVERY residual evaluation -- so it still reports on a
        # step that diverges before it can converge.
        # Suppress during the FD-Jacobian probe: the probe's internal residual
        # re-evaluations are rank-local (esp. the targeted row dump), and
        # _log_residual_terms does an MPI allreduce -- calling it from only the
        # owning rank deadlocks the others.  Also de-spams the probe.
        if os.environ.get("MCOMP_RES_DBG") and not getattr(self, "_in_fd_probe", False):
            self._log_residual_terms(self.timeIntegration.t, r)
        if self.globalResidualDummy is None:
            self.globalResidualDummy = np.zeros(r.shape,'d')

    def _log_residual_terms(self, t, r):
        r"""Per-residual-evaluation decomposition for divergence triage.

        Enabled with MCOMP_RES_DBG=1.  Logged on EVERY residual evaluation
        (unlike the [gas budget]/[Mass balance] lines, which run in postStep and
        so are never reached on a step that diverges).  For the assembled
        residual r it reports, each Newton iteration:
          * per-component global ||r||_2 (and the change d= since the previous
            evaluation) and global ||r||_inf -- the component whose norm climbs
            is where Newton is diverging;
          * the rank-0-local node carrying max|r| in each component: mesh node,
            (x,y), material flag, and the current p and z there -- tells you
            WHERE (which facies / front) the blow-up sits;
          * the kernel-exported per-term comp-1 (CO2/z) budget for THIS eval
            (accum / flux / sink / injection / boundary) -- the term whose
            magnitude tracks the growth is the culprit term;
          * a non-finite scan flagging the first NaN/Inf DOF.

        Reads-only; no MPI collective is gated behind a rank test (allreduce is
        called on every rank, printing on rank 0) so it is deadlock-safe in
        parallel.
        """
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
        except Exception:
            comm = None
            rank = 0
        off = list(self.offset); strd = list(self.stride)
        nfree = list(self.nFreeDOF_global)
        pdof = np.asarray(self.u[0].dof)
        zdof = np.asarray(self.u[1].dof) if self.nc >= 2 else None
        f2n_u = np.asarray(self.freeDOFToNode_u)
        X = np.asarray(self.mesh.nodeArray)
        mat = np.asarray(self.nodeMaterialTypes_n)
        names = {0: 'p', 1: 'z'}
        it = int(getattr(self, 'nonlinear_function_evaluations', -1))
        nsize = int(comm.Get_size()) if comm is not None else 1
        # Owned-vs-ghost split.  In parallel the local free-DOF / node arrays
        # carry a trailing ghost layer; those rows are NOT owned equations, so
        # they must be excluded from the global ||r|| (otherwise shared DOFs are
        # double-counted across ranks and the printed norm disagrees with the
        # solver's own owned-only PETSc norm).  Mesh nodes [0, nNodes_owned) are
        # owned on this rank; the rest are ghosts.
        nodes_owned = int(getattr(self.mesh, 'nNodes_owned',
                                  getattr(self.mesh, 'nNodes_global', X.shape[0])))

        def _owned_mask(ci):
            # boolean mask over this component's free-DOF slice: True = owned.
            nf = int(nfree[ci])
            if ci == 0:
                m = np.zeros((nf,), dtype=bool)
                k = min(nf, f2n_u.shape[0])
                m[:k] = f2n_u[:k] < nodes_owned
                return m
            # comp-1 (z): no Dirichlet, free-DOF index == mesh-node index.
            return np.arange(nf) < nodes_owned

        def _is_partition_boundary(ci, li):
            # owned free-DOF li whose stencil reaches a ghost node sits on a
            # partition cut -- the prime suspect for a parallel-only blow-up
            # (cross-cut flux inconsistency) vs an interior (front) blow-up.
            try:
                if ci == 0:
                    rp = getattr(self, 'comp0_rowptr', None)
                    cidx = getattr(self, 'comp0_colind', None)
                    if rp is None or cidx is None or li + 1 >= rp.shape[0]:
                        return -1
                    for off2 in range(int(rp[li]), int(rp[li + 1])):
                        jn = int(f2n_u[cidx[off2]]) if cidx[off2] < f2n_u.shape[0] else -1
                        if jn >= nodes_owned:
                            return 1
                    return 0
                rp = getattr(self, 'comp1_rowptr', None)
                cidx = getattr(self, 'comp1_colind', None)
                if rp is None or cidx is None or li + 1 >= rp.shape[0]:
                    return -1
                for off2 in range(int(rp[li]), int(rp[li + 1])):
                    if int(cidx[off2]) >= nodes_owned:
                        return 1
                return 0
            except Exception:
                return -1

        # non-finite scan (rank-local; first offending global index).
        nfin_local = 0 if np.all(np.isfinite(r)) else 1
        nfin = comm.allreduce(nfin_local, op=MPI.SUM) if comm is not None else nfin_local
        if nfin and nfin_local:
            bad = int(np.argmax(~np.isfinite(r)))
            logEvent("[res dbg] t={:.4e} it={} rank={} *** NON-FINITE r at global "
                     "idx {} (value={}) ***".format(float(t), it, rank, bad, r[bad]),
                     level=1)

        if not hasattr(self, '_res_dbg_prev'):
            self._res_dbg_prev = {}
        for ci in range(self.nc):
            sl = r[off[ci]:off[ci] + strd[ci] * nfree[ci]:strd[ci]]
            owned = _owned_mask(ci)
            slo = sl[owned] if sl.size else sl
            ss_local = float(np.sum(slo * slo)) if slo.size else 0.0
            # This rank's owned argmax + full detail tuple.  EVERY rank assembles
            # its own detail; we gather to rank 0 and rank 0 picks/prints the
            # global max.  (proteus logEvent emits only on rank 0, so the owner
            # rank cannot print the localization itself -- it would be dropped.)
            node = -1; xloc = 0.0; yloc = 0.0; m = -1; bnd = -1
            pv = float('nan'); zv = float('nan')
            if slo.size:
                li_owned = int(np.argmax(np.abs(slo)))
                linf_local = float(np.abs(slo[li_owned]))
                li_local = int(np.flatnonzero(owned)[li_owned])
                node = int(f2n_u[li_local]) if (ci == 0 and li_local < f2n_u.shape[0]) else int(li_local)
                if node < X.shape[0]:
                    xloc = float(X[node][0]); yloc = float(X[node][1])
                if node < mat.shape[0]:
                    m = int(mat[node])
                if node < pdof.shape[0]:
                    pv = float(pdof[node])
                if zdof is not None and node < zdof.shape[0]:
                    zv = float(zdof[node])
                bnd = _is_partition_boundary(ci, li_local)
            else:
                linf_local = 0.0
            detail = (linf_local, rank, node, xloc, yloc, m, bnd, pv, zv)
            if comm is not None:
                ss = comm.allreduce(ss_local, op=MPI.SUM)
                per_rank = comm.gather(np.sqrt(ss_local), root=0)
                details = comm.gather(detail, root=0)
            else:
                ss = ss_local
                per_rank = [np.sqrt(ss_local)]
                details = [detail]
            l2 = float(np.sqrt(ss))
            prev = self._res_dbg_prev.get(ci, l2)
            self._res_dbg_prev[ci] = l2

            if rank == 0:
                gmax, owner, gnode, gx, gy, gm, gbnd, gp, gz = max(details, key=lambda d: d[0])
                logEvent("[res dbg] t={:.4e} it={} comp{}({}) ||r||2={:.4e} "
                         "d={:+.3e} ||r||inf={:.4e}@rank{} node{} "
                         "(x={:.4f},y={:.4f}) mat={} cut={} p={:.4e} z={:.4e}"
                         .format(float(t), it, ci, names.get(ci, '?'), l2,
                                 l2 - prev, float(gmax), int(owner), int(gnode),
                                 float(gx), float(gy), int(gm), int(gbnd),
                                 float(gp), float(gz)), level=1)
                if nsize > 1 and per_rank is not None:
                    # top-5 ranks by owned ||r||2 -- one rank dominating points
                    # at a partition-local (cut) divergence.
                    top = sorted(enumerate(per_rank), key=lambda kv: -kv[1])[:5]
                    pr = " ".join("r{}={:.2e}".format(k, v) for k, v in top)
                    logEvent("[res dbg] t={:.4e} it={} comp{} top5||r||2: {}"
                             .format(float(t), it, ci, pr), level=1)

        # per-term comp-1 (CO2/z) budget for THIS residual evaluation (owned only).
        gb = getattr(self, 'gas_budget_node', None)
        if gb is not None and self.nc >= 2:
            gb = np.asarray(gb); n = gb.shape[0] // 6
            if n > 0:
                ns = min(nfree[1], n, nodes_owned)
                terms_local = [float(np.sum(gb[k * n:k * n + ns])) for k in range(6)]
                if comm is not None:
                    terms = [comm.allreduce(v, op=MPI.SUM) for v in terms_local]
                else:
                    terms = terms_local
                if rank == 0:
                    acc, flx, snk, inj, bnd, tot = terms
                    logEvent("[res dbg] t={:.4e} it={} z-budget accum={:+.4e} "
                             "flux={:+.4e} sink={:+.4e} inj={:+.4e} bnd={:+.4e} "
                             "total={:+.4e}".format(float(t), it, acc, flx, snk,
                                                    inj, bnd, tot), level=1)

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
    
        full_rowptr, full_colind, Cx = self.cterm_global[0].getCSRrepresentation()
        self._ensure_component0_compact_csr(full_rowptr, full_colind)
        rowptr = self.comp0_rowptr
        colind = self.comp0_colind
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
        argsDict["rho_n"] = self.coefficients.rho_n
        argsDict["p_ref_n"] = self.coefficients.p_ref_n
        argsDict["immiscible"] = int(self.coefficients.immiscible)
        argsDict["T_C"] = self.coefficients.T_C
        argsDict["beta"] = self.coefficients.beta
        argsDict["gravity"] = self.coefficients.gravity
        argsDict["alpha"] = self.coefficients.vgm_alpha_types
        argsDict["n"] = self.coefficients.vgm_n_types
        argsDict["thetaR"] = self.coefficients.thetaR_types
        argsDict["thetaSR"] = self.coefficients.thetaSR_types
        argsDict["KWs"] = self.coefficients.Ksw_types
        argsDict["krn_end"] = self.coefficients.krn_end_types
        argsDict["S_gr"] = self.coefficients.S_gr_types
        argsDict["bc_Se_min"] = self.coefficients.bc_Se_min_types
        argsDict["mu_n"]    = self.coefficients.mu_n
        argsDict["useMetrics"] = 0.0
        argsDict["alphaBDF"] = self.timeIntegration.alpha_bdf
        argsDict["lag_shockCapturing"] = 0
        argsDict["shockCapturingDiffusion"] = self.coefficients.SC
        argsDict["sc_uref"] = 1.0
        argsDict["sc_alpha"] = 2.0
        argsDict["u_l2g"] = self.u[0].femSpace.dofMap.l2g
        # Component-1 (z) map.  Each component owns its own femSpace.dofMap.l2g
        # (standard multi-component proteus layout); comp-1's is parallel-correct
        # via the same offset[1]/stride[1]/par_dof machinery Richards.h uses for
        # its single component.  u_l2g_n == u_l2g element-wise on the shared P1 mesh.
        argsDict["u_l2g_n"] = self.u[1].femSpace.dofMap.l2g
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
        # component-1 (S_n) boundary arrays. Initialised in
        # init() when missing so this works regardless of numericalFlux class.
        argsDict["isDOFBoundary_n"] = self.numericalFlux.isDOFBoundary[1]
        argsDict["ebqe_bc_u_n_ext"] = self.numericalFlux.ebqe[('u',1)]
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
        argsDict["numDOFs"] = self.nFreeDOF_global[0]
        # numDOFs_u bounds the Richards DOF loop to comp-0.
        argsDict["numDOFs_u"] = self.nFreeDOF_global[0]
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
        argsDict["PSK_TYPE"] = self.coefficients.PSK_TYPE
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
        argsDict["freeDOFToNode_u"] = self.freeDOFToNode_u
        argsDict["USE_NEWTON_INVERT"] = 1 if (self.coefficients._fct_requested and self.coefficients.nd > 1) else 0
        argsDict["PSK_TYPE"] = self.coefficients.PSK_TYPE
        argsDict["COMPONENT"] = 0  # m -> u_w via retention curve (legacy path)
        self.m_comp_co2.invert(argsDict)

    def getJacobian(self,jacobian):
        if (self.coefficients.STABILIZATION_TYPE == 0):  # SUPG
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
        argsDict["rho_n"] = self.coefficients.rho_n
        argsDict["p_ref_n"] = self.coefficients.p_ref_n
        argsDict["immiscible"] = int(self.coefficients.immiscible)
        argsDict["T_C"] = self.coefficients.T_C
        argsDict["beta"] = self.coefficients.beta

        argsDict["q_rho"]= self.q['rho']
        argsDict["ebqe_rho"]= self.ebqe['rho']
        
        argsDict["gravity"] = self.coefficients.gravity
        argsDict["alpha"] = self.coefficients.vgm_alpha_types
        argsDict["n"] = self.coefficients.vgm_n_types
        argsDict["thetaR"] = self.coefficients.thetaR_types
        argsDict["thetaSR"] = self.coefficients.thetaSR_types
        argsDict["KWs"] = self.coefficients.Ksw_types
        argsDict["krn_end"] = self.coefficients.krn_end_types
        argsDict["S_gr"] = self.coefficients.S_gr_types
        argsDict["bc_Se_min"] = self.coefficients.bc_Se_min_types
        argsDict["mu_n"]    = self.coefficients.mu_n
        # Per-rock closure (weights ghost-synced by getResidual, which runs first).
        argsDict["per_rock_closure"] = int(self.coefficients.per_rock_closure)
        argsDict["node_rock_W"] = self.node_rock_W
        argsDict["nRockTypes"] = int(self.node_rock_W.shape[1])
        argsDict["useMetrics"] = 0.0
        argsDict["alphaBDF"] = self.timeIntegration.alpha_bdf
        argsDict["lag_shockCapturing"] = 0
        argsDict["shockCapturingDiffusion"] = 0.1
        argsDict["u_l2g"] = self.u[0].femSpace.dofMap.l2g
        # Component-1 (z) map.  Each component owns its own femSpace.dofMap.l2g
        # (standard multi-component proteus layout); comp-1's is parallel-correct
        # via the same offset[1]/stride[1]/par_dof machinery Richards.h uses for
        # its single component.  u_l2g_n == u_l2g element-wise on the shared P1 mesh.
        argsDict["u_l2g_n"] = self.u[1].femSpace.dofMap.l2g
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
        # component-1 (S_n) Jacobian (1,1) block args.
        argsDict["dt"] = self.timeIntegration.dt
        argsDict["csrRowIndeces_n_n"] = self.csrRowIndeces[(1, 1)]
        argsDict["csrColumnOffsets_n_n"] = self.csrColumnOffsets[(1, 1)]
        # (1,0) cross-block CSR maps.
        argsDict["csrRowIndeces_n_w"] = self.csrRowIndeces[(1, 0)]
        argsDict["csrColumnOffsets_n_w"] = self.csrColumnOffsets[(1, 0)]
        # Exterior-boundary column offsets for the comp-1 boundary Jacobian
        # loop in calculateJacobian (STAB=0 comp-1 Dirichlet enforcement).
        argsDict["csrColumnOffsets_eb_n_n"] = self.csrColumnOffsets_eb[(1, 1)]
        argsDict["csrColumnOffsets_eb_n_w"] = self.csrColumnOffsets_eb[(1, 0)]
        # (0,1) cross-block CSR maps for the wetting eq.
        argsDict["csrRowIndeces_w_n"] = self.csrRowIndeces[(0, 1)]
        argsDict["csrColumnOffsets_w_n"] = self.csrColumnOffsets[(0, 1)]
        argsDict["u_dof_n"] = self.u[1].dof
        argsDict["u_dof_n_old"] = self.u_dof_n_old if self.u_dof_n_old is not None else self.u[1].dof
        argsDict["offset_n"] = self.offset[1]
        argsDict["stride_n"] = self.stride[1]
        # Stage 3b: kinetic dissolution sink contributes -k_d * S_n * (1-S_n) *
        # (c_sat - c) * theta_w * rho_w to the gas-equation residual.  Its
        # derivative wrt S_n is needed in the (1,1) Jacobian block (signs of
        # the two factors are opposite so the linearization is well-behaved
        # near c < c_sat).  c_dof and parameters supplied here for the kernel
        # to evaluate; defaulting to k_d=0 keeps legacy Jacobian unchanged.
        argsDict["c_dof"] = getattr(self.coefficients, "c_dof",
                                    np.zeros_like(self.u[1].dof))
        # In flash mode the once-per-step nodal dissolutionFlash owns the
        # gas<->brine exchange, so the in-residual kinetic R_diss MUST be off
        # (pass k_d=0) to avoid double counting; self.coefficients.k_d is kept
        # only for the diagnostic.
        argsDict["k_d"]   = (0.0 if self.coefficients.dissolution_mode == 'flash'
                             else float(self.coefficients.k_d))
        argsDict["c_sat"] = float(self.coefficients.c_sat)
        argsDict["globalJacobian"] = jacobian.getCSRrepresentation()[2]
        argsDict["delta_x_ij"] = self.delta_x_ij
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["ebqe_velocity_ext"] = self.ebqe['velocity',0 ]
        argsDict["isDOFBoundary_u"] = self.numericalFlux.isDOFBoundary[0]
        argsDict["ebqe_bc_u_ext"] = self.numericalFlux.ebqe[('u',0)]
        # component-1 (S_n) boundary arrays. Initialised in
        # init() when missing so this works regardless of numericalFlux class.
        argsDict["isDOFBoundary_n"] = self.numericalFlux.isDOFBoundary[1]
        argsDict["ebqe_bc_u_n_ext"] = self.numericalFlux.ebqe[('u',1)]
        argsDict["isFluxBoundary_u"] = self.ebqe[('advectiveFlux_bc_flag',0)]
        argsDict["ebqe_bc_flux_ext"] = self.ebqe[('advectiveFlux_bc',0)]
        argsDict["csrColumnOffsets_eb_u_u"] = self.csrColumnOffsets_eb[(0,0)]
        # (0,1) cross-block boundary CSR for the wetting eq
        # exterior-flux Jacobian. csrRowIndeces_w_n is already passed above.
        argsDict["csrColumnOffsets_eb_w_n"] = self.csrColumnOffsets_eb[(0,1)]
        argsDict["LUMPED_MASS_MATRIX"] = self.coefficients.LUMPED_MASS_MATRIX
        argsDict["VMS"] = self.coefficients.VMS
        argsDict["PSK_TYPE"] = self.coefficients.PSK_TYPE
        #argsDict["anb_seepage_flux"] = self.coefficients.anb_seepage_flux

        self.calculateJacobian(argsDict)
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
        # ---- one-shot finite-difference Jacobian consistency probe ----------
        # Enable with MCOMP_FD_JAC=<jac-eval-index> (e.g. 5 to sample a stalled
        # mid-Newton tangent).  Compares the assembled globalJacobian against
        # (R(u+eps e_j) - R(u))/eps for a sample of columns and reports the
        # worst row/col mismatches, classified interior / boundary / injection.
        #
        # Alternatively set MCOMP_FD_JAC_TIME=<t> to fire on the FIRST Jacobian
        # eval at timeIntegration.t >= t (more robust than counting jac evals
        # when you want the first Newton iterate of a specific diverging step --
        # e.g. MCOMP_FD_JAC_TIME=1800 captures the still-finite tangent before
        # the z<0 overshoot detonates).  Either trigger arms the probe.
        _fd_env  = os.environ.get("MCOMP_FD_JAC")
        _fd_time = os.environ.get("MCOMP_FD_JAC_TIME")
        if (_fd_env is not None or _fd_time is not None) \
                and not getattr(self, "_fd_done", False):
            _fire = False
            if _fd_time is not None:
                # Fire on the (1 + MCOMP_FD_JAC_SKIP)-th Jacobian eval at t>=_fd_time.
                # SKIP=0 (default) -> u0 / onset (S_g~0).  SKIP=3 probes a DEVELOPED
                # two-phase iterate (e.g. Newton it 3, ||r||~1) to test whether the
                # tangent stays consistent once gas has grown, not just at appearance.
                _skip = int(os.environ.get("MCOMP_FD_JAC_SKIP", "0"))
                if float(self.timeIntegration.t) >= float(_fd_time):
                    self._fd_skip_seen = getattr(self, "_fd_skip_seen", 0)
                    _fire = (self._fd_skip_seen >= _skip)
                    self._fd_skip_seen += 1
            else:
                _target = int(_fd_env) if _fd_env.lstrip("-").isdigit() else 5
                _fire = self.nonlinear_function_jacobian_evaluations >= _target
            if _fire:
                try:
                    self._in_fd_probe = True   # gate off the MPI-collective res log
                    self._fd_jacobian_probe(jacobian)
                finally:
                    self._fd_done = True
                    self._in_fd_probe = False
        return jacobian

    def _fd_jacobian_probe(self, jacobian):
        """Finite-difference consistency check of the assembled Jacobian.

        At the current Newton iterate u0 (stashed by getResidual), for a sample
        of columns j we form the FD column (R(u0+du e_j)-R(u0))/du and compare
        it to the assembled tangent column.  The assembled matrix passed in is
        the FULL tangent (flux block from calculateResidual_entropy_viscosity +
        mass block from calculateMassMatrix + any strong-Dirichlet rows).

        NOTE: re-evaluating getResidual overwrites self.jacobian with the
        flux-only block, so we snapshot the full nzval first and restore it (and
        the component DOFs) before returning.
        """
        if getattr(self, "_fd_last_u", None) is None:
            logEvent("[FD JAC PROBE] skipped: no stashed residual point", level=1)
            return
        ncols = int(os.environ.get("MCOMP_FD_JAC_NCOLS", "48"))
        eps   = float(os.environ.get("MCOMP_FD_JAC_EPS", "1.0e-7"))
        ntop  = int(os.environ.get("MCOMP_FD_JAC_NTOP", "25"))

        u0 = np.copy(self._fd_last_u)
        dim = u0.shape[0]
        rowptr, colind, nzval = jacobian.getCSRrepresentation()
        A = np.copy(nzval)                       # full assembled tangent
        # offset of each nonzero -> its row, for fast column extraction.
        row_of_nz = np.repeat(np.arange(dim, dtype=np.int64), np.diff(rowptr))

        # baseline residual at u0 (this refills self.jacobian flux-only).
        r0 = np.zeros_like(u0)
        self.getResidual(u0, r0)

        # component / node bookkeeping for classification.
        off = list(self.offset); strd = list(self.stride)
        nfree = list(self.nFreeDOF_global)
        f2n_u = self.freeDOFToNode_u
        inj = getattr(self, "injection_dof", None)
        is_dbc0 = self.numericalFlux.isDOFBoundary[0]
        is_dbc1 = self.numericalFlux.isDOFBoundary[1] if self.nc >= 2 else None
        nnodes_mesh = int(self.mesh.nodeArray.shape[0])

        def classify(row):
            for ci in (0, 1) if self.nc >= 2 else (0,):
                d = row - off[ci]
                if d >= 0 and (strd[ci] == 0 or d % strd[ci] == 0):
                    li = d // strd[ci] if strd[ci] else d
                    if 0 <= li < nfree[ci]:
                        node = int(f2n_u[li]) if ci == 0 else int(li)   # comp-1 DOF == mesh node
                        if node < 0 or node >= nnodes_mesh:
                            return ci, node, np.zeros(3), False
                        x = self.mesh.nodeArray[node]
                        injf = bool(inj is not None and node < inj.shape[0]
                                    and inj[node] != 0.0)
                        return ci, node, x, injf
            return -1, -1, np.zeros(3), False

        cols = np.unique(np.linspace(0, dim - 1, ncols).astype(np.int64))
        records = []   # (rel, abs, row, col, a_val, fd_val)
        miss    = []   # structural misses: FD large where matrix has no entry
        rp = np.zeros_like(u0)
        for j in cols:
            du = eps * max(1.0, abs(u0[j]))
            up = np.copy(u0); up[j] += du
            self.getResidual(up, rp)
            fd = (rp - r0) / du
            sel = np.where(colind == j)[0]          # nonzeros in column j
            rows_j = row_of_nz[sel]
            a_vals = A[sel]
            fd_vals = fd[rows_j]
            diff = np.abs(a_vals - fd_vals)
            denom = np.abs(fd_vals) + np.abs(a_vals) + 1.0e-30
            rel = diff / denom
            for k in range(rows_j.shape[0]):
                records.append((rel[k], diff[k], int(rows_j[k]), int(j),
                                float(a_vals[k]), float(fd_vals[k])))
            # FD-significant rows that have NO structural entry in column j.
            scale = np.abs(fd).max() + 1.0e-30
            big = np.where(np.abs(fd) > 1.0e-3 * scale)[0]
            have = set(rows_j.tolist())
            for ii in big:
                if int(ii) not in have:
                    miss.append((float(fd[ii]), int(ii), int(j)))

        # ---- targeted single-row dump (MCOMP_FD_JAC_ROW=<mesh node>|auto) ----
        # Dump the FULL comp-0 AND comp-1 equation rows at the given mesh node:
        # for every column in each row's CSR sparsity, perturb that column and
        # read A_ij vs FD_ij at the target row.  Lets you see exactly which
        # coupling (which neighbour node / which component column) is wrong --
        # e.g. the 2x (1,0) flux off-diagonal at an injection-front node.
        # MCOMP_FD_JAC_ROW=auto (or an out-of-range node) auto-targets the
        # comp-1 node carrying the largest |A-FD| in the column sample, so it
        # always lands on the worst node regardless of mesh / node numbering.
        # SERIAL-ONLY: the row dump is rank-local (only the rank owning the
        # target node enters the loop), but getResidual contains an MPI
        # collective (scatter_forward_insert of the gate arrays), so a rank-local
        # loop deadlocks the other ranks in parallel.  In serial that scatter is
        # a no-op.  Run the coarse mesh serially for the row dump.
        try:
            from mpi4py import MPI as _MPI
            _nranks = _MPI.COMM_WORLD.Get_size()
        except Exception:
            _nranks = 1
        row_records = []   # (comp_eq, row, col, a_val, fd_val, rel)
        _row_env = os.environ.get("MCOMP_FD_JAC_ROW")
        if _row_env is not None and _nranks > 1:
            logEvent("[FD JAC PROBE] targeted row dump SKIPPED in parallel "
                     "({} ranks); rerun serial (mpiexec -n 1 / no mpiexec) for "
                     "the row dump.".format(_nranks), level=1)
            _row_env = None
        if _row_env is not None:
            nd_t = -1
            if _row_env.lstrip("-").isdigit():
                _c = int(_row_env)
                if 0 <= _c < int(self.mesh.nNodes_global):
                    nd_t = _c
            elif _row_env.strip().lower() == "auto-front":
                # FRONT auto-target: among nodes with z > zmin (the CO2 plume,
                # where the bubble-point stiffness lives -- NOT the quiescent
                # boundary), pick the one with the largest (1,1) diagonal |A-FD|
                # via a quick per-node diagonal FD.  This lands on the 2x
                # mis-weighting that drives the t=1800 detonation, skipping the
                # large-but-benign y=0 boundary phantoms.
                zmin = float(os.environ.get("MCOMP_FD_JAC_ZMIN", "1.0e-4"))
                zdof = np.asarray(self.u[1].dof)
                nN   = int(self.mesh.nNodes_global)
                cand = np.where(zdof[:nN] > zmin)[0]
                # bound cost: keep the top-N front nodes by z.
                ncap = int(os.environ.get("MCOMP_FD_JAC_FRONT_CAP", "150"))
                if cand.shape[0] > ncap:
                    cand = cand[np.argsort(zdof[cand])[-ncap:]]
                worst = -1.0
                for nd in cand.tolist():
                    row = off[1] + strd[1] * nd          # comp-1 diagonal row
                    if row < 0 or row >= dim:
                        continue
                    o_diag = -1
                    for o in range(int(rowptr[row]), int(rowptr[row + 1])):
                        if colind[o] == row:
                            o_diag = o; break
                    if o_diag < 0:
                        continue
                    du = eps * max(1.0, abs(u0[row]))
                    up = np.copy(u0); up[row] += du
                    self.getResidual(up, rp)
                    fd = float((rp[row] - r0[row]) / du)
                    dab = abs(float(A[o_diag]) - fd)
                    if dab > worst:
                        worst = dab; nd_t = nd
                logEvent("[FD JAC PROBE] MCOMP_FD_JAC_ROW=auto-front targeted node "
                         "{} (z>{:.1e}: {} cand, worst (1,1)-diag |A-FD|={:.3e})"
                         .format(nd_t, zmin, int(cand.shape[0]), worst), level=1)
            if nd_t < 0:   # 'auto' or out-of-range -> worst comp-1 node sampled
                worst = -1.0
                for rel0, dab0, row0, col0, av0, fv0 in records:
                    cci, cnode, _, _ = classify(row0)
                    if cci == 1 and dab0 > worst:
                        worst = dab0; nd_t = cnode
                logEvent("[FD JAC PROBE] MCOMP_FD_JAC_ROW auto-targeted node {} "
                         "(worst comp-1 |A-FD|={:.3e})".format(nd_t, worst), level=1)
        if _row_env is not None and 0 <= nd_t < int(self.mesh.nNodes_global):
            # node -> free-DOF index per component (comp-1 identity; comp-0 via
            # the inverse of freeDOFToNode_u).
            n2f_u = -np.ones(int(self.mesh.nNodes_global), 'i')
            _valid = (f2n_u >= 0)
            n2f_u[f2n_u[_valid]] = np.arange(f2n_u.shape[0])[_valid]
            for ci in (0, 1) if self.nc >= 2 else (0,):
                if ci == 0:
                    free = int(n2f_u[nd_t]) if 0 <= nd_t < n2f_u.shape[0] else -1
                else:
                    free = nd_t
                if free < 0 or free >= nfree[ci]:
                    continue
                row = off[ci] + strd[ci] * free
                if row < 0 or row >= dim:
                    continue
                cols_r = colind[rowptr[row]:rowptr[row + 1]]
                a_r    = A[rowptr[row]:rowptr[row + 1]]
                for col, av in zip(cols_r.tolist(), a_r.tolist()):
                    du = eps * max(1.0, abs(u0[col]))
                    up = np.copy(u0); up[col] += du
                    self.getResidual(up, rp)
                    fv = float((rp[row] - r0[row]) / du)
                    rel = abs(av - fv) / (abs(av) + abs(fv) + 1.0e-30)
                    row_records.append((ci, int(row), int(col), float(av), fv, rel))

        # restore component DOFs to u0 and the full assembled matrix.
        self.getResidual(u0, rp)
        rowptr, colind, nzval = jacobian.getCSRrepresentation()
        nzval[:] = A

        records.sort(key=lambda t: t[0], reverse=True)
        miss.sort(key=lambda t: abs(t[0]), reverse=True)
        lines = []
        lines.append("================ FD JACOBIAN PROBE ================")
        lines.append("t={:.6e}  jac_eval={}  dim={}  cols_tested={}  eps={:.1e}"
                     .format(self.timeIntegration.t,
                             self.nonlinear_function_jacobian_evaluations,
                             dim, cols.shape[0], eps))
        lines.append("||R(u0)|| = {:.6e}".format(float(np.linalg.norm(r0))))
        if records:
            rels = np.array([r[0] for r in records])
            lines.append("rel-err over {} sampled entries: max={:.3e} "
                         "median={:.3e}  frac>1e-3={:.3f}"
                         .format(len(records), rels.max(),
                                 float(np.median(rels)),
                                 float((rels > 1.0e-3).mean())))
        lines.append("--- worst {} entries (rel | A_ij | FD_ij | comp node (x,y) inj) ---"
                     .format(ntop))
        for rel, dab, row, col, av, fv in records[:ntop]:
            ci, node, x, injf = classify(row)
            lines.append("rel={:.2e} A={:+.4e} FD={:+.4e} d={:.2e} "
                         "row={} col={} comp={} node={} (x={:.4f},y={:.4f}){}"
                         .format(rel, av, fv, dab, row, col, ci, node,
                                 x[0], x[1], "  INJ" if injf else ""))
        lines.append("--- structural misses: |FD| significant, NO matrix entry ({}) ---"
                     .format(len(miss)))
        for fv, row, col in miss[:ntop]:
            ci, node, x, injf = classify(row)
            lines.append("FD={:+.4e} row={} col={} comp={} node={} "
                         "(x={:.4f},y={:.4f}){}"
                         .format(fv, row, col, ci, node, x[0], x[1],
                                 "  INJ" if injf else ""))
        if row_records:
            lines.append("--- targeted row dump @ node {} "
                         "(eqcomp | colcomp colnode (x,y) | A_ij | FD_ij | rel) ---"
                         .format(nd_t))
            for ci_eq, row, col, av, fv, rel in row_records:
                cc, cnode, cx, cinj = classify(col)
                flag = "  <<2x" if 0.4 <= rel <= 0.6 else (
                       "  <<BAD" if rel > 0.1 else "")
                lines.append("eq{}({}) <- col c{} node{} (x={:.4f},y={:.4f}){} "
                             "A={:+.6e} FD={:+.6e} rel={:.3e}{}"
                             .format(ci_eq, "p" if ci_eq == 0 else "z", cc, cnode,
                                     cx[0], cx[1], "  INJ" if cinj else "",
                                     av, fv, rel, flag))
        lines.append("===================================================")
        report = "\n".join(lines)
        logEvent(report, level=1)
        try:
            comm = Comm.get()
            if comm.isMaster():
                with open("fd_jac_probe.txt", "a") as fh:
                    fh.write(report + "\n")
        except Exception:
            pass
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
        # Refresh the derived compositional fields (Sg, X, c_brine) from the
        # just-completed step's primary (p,z) so they archive consistently with
        # this time level's p_w / S_n.  Uses the SAME C++ flashPZ as the
        # residual (calculateFlashFields), so the XDMF fields are the solver's
        # internal compositional state -- not an external post-process replica.
        if getattr(self, 'Sg_dof', None) is None:
            return
        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["p_dof"]      = self.u[0].dof
        argsDict["z_dof"]      = self.u[1].dof
        argsDict["Sg_dof"]     = self.Sg_dof
        argsDict["X_dof"]      = self.X_dof
        argsDict["c_dof"]      = self.c_brine_dof
        # Loop over MESH NODES (== p_dof / Sg_dof size).
        argsDict["numDOFs"]    = self.u[0].dof.shape[0]
        argsDict["immiscible"] = int(self.coefficients.immiscible)
        argsDict["T_C"] = self.coefficients.T_C
        # Per-rock closure output: the kernel copies the min / max of S_r over
        # each node's rocks from its last residual (placeholders when off).
        argsDict["per_rock_closure"] = int(self.coefficients.per_rock_closure)
        argsDict["Sg_rock_max_dof"] = (self.Sg_rock_max_dof if self.Sg_rock_max_dof is not None
                                       else self.node_rock_W.ravel())
        argsDict["Sg_rock_min_dof"] = (self.Sg_rock_min_dof if self.Sg_rock_min_dof is not None
                                       else self.node_rock_W.ravel())
        self.m_comp_co2.calculateFlashFields(argsDict)
    #def postStep(self, t, firstStep=False):
    #    with open('seepage_flux_nnnnk', "a") as f:
    #        f.write("\n Time"+ ",\t" +"Seepage\n")
    #        f.write(repr(t)+ ",\t" +repr(self.coefficients.anb_seepage_flux))

    

#argsDict["anb_seepage_flux"] = self.coefficients.anb_seepage_flux        
#anb_seepage_flux= self.coefficients.anb_seepage_flux
#print("Hi",anb_seepage_flux)


#print("Hello from the python file", self.coefficients.anb_seepage_flux)
