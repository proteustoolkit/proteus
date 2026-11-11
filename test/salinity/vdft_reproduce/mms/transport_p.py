"""TADR half of the coupled MMS (mms_coupled_subsection.tex).

The paper configuration: ImplicitEV (STABILIZATION_TYPE=5) advecting with the
flow model's edge fluxes, FCT from MMS_FCT, strong Dirichlet c_e on all four
sides, and the cancelled-form source g added as a lumped nodal term.

MMSMonitorCoefficients logs, after every step:

    [MMS-ERR]  ML-weighted E1, E2 and max-norm errors of c and psi against
               c_e, psi_e at t^{n+1}, over owned nodes
    [MMS-B]    max_i |B_i - ML_i f_w,i| over interior nodes, split into nodes
               next to a Dirichlet node and the rest.  B is TADR's nodal
               boundary flux (a rate); at an interior node it must equal the
               flow source up to the flow model's Newton residual.

There is no solute budget: c is pinned on all four sides, so the solute that
crosses the boundary is not observable from the deck (MMS_PLAN.md).
"""

import numpy as np
from proteus import *
from proteus.default_p import *
from proteus.Comm import globalMax, globalSum
from proteus.mprans import TADR
from proteus.Profiling import logEvent

try:
    from .domain_mms import *
    from . import mms_exact
except ImportError:
    from domain_mms import *
    import mms_exact

checkMass = False
shockCapturingFactor_tadr = 0.0
lag_shockCapturing_tadr = True


class MMSMonitorCoefficients(TADR.Coefficients):
    """TADR coefficients that report the MMS errors after each step."""

    def postStep(self, t, firstStep=False):
        result = TADR.Coefficients.postStep(self, t, firstStep)
        if getattr(self, "vModel", None) is not None and self.vModel.ML is not None:
            self._report_mms(t)
        return result

    def _masks(self):
        """Owned interior nodes, split into Dirichlet-adjacent and deeper ones."""
        m = self.model
        is_dir = np.zeros(m.u[0].dof.shape, dtype=bool)
        is_dir[list(m.dirichletConditionsForceDOF.DOFBoundaryConditionsDict.keys())] = True
        adj = np.zeros_like(is_dir)
        for i in range(len(m.rowptr) - 1):
            adj[i] = is_dir[m.colind[m.rowptr[i]:m.rowptr[i + 1]]].any()
        no = m.mesh.nNodes_owned
        interior = ~is_dir[:no]
        return interior & adj[:no], interior & ~adj[:no]

    def _report_mms(self, t_arg):
        m, v = self.model, self.vModel
        tn = m.timeIntegration.t
        no = m.mesh.nNodes_owned
        X = m.mesh.nodeArray[:no]
        if not hasattr(self, "_mms_masks"):
            self._mms_masks = self._masks()
            logEvent("[MMS-CFG] %s N=%d dt=%.6e nsteps=%d FCT=%d "
                     "(postStep t=%.9e timeIntegration.t=%.9e)"
                     % (mms_exact.describe(), N, DT, NSTEPS, int(FCT), t_arg, tn))
        near, deep = self._mms_masks

        ec = np.asarray(m.u[0].dof[:no]) - mms_exact.c_e(X, tn)
        ep = np.asarray(v.u[0].dof[:no]) - mms_exact.psi_e(X, tn)
        MLc = np.asarray(m.ML[:no])
        MLp = np.asarray(v.ML[:no])
        e1c = globalSum(float((MLc * np.abs(ec)).sum()))
        e2c = globalSum(float((MLc * ec ** 2).sum())) ** 0.5
        eic = globalMax(float(np.abs(ec).max()))
        e1p = globalSum(float((MLp * np.abs(ep)).sum()))
        e2p = globalSum(float((MLp * ep ** 2).sum())) ** 0.5
        eip = globalMax(float(np.abs(ep).max()))
        logEvent("[MMS-ERR] t=%.9e c: E1=%.6e E2=%.6e Einf=%.6e "
                 "psi: E1=%.6e E2=%.6e Einf=%.6e"
                 % (tn, e1c, e2c, eic, e1p, e2p, eip))

        src = MLc * mms_exact.f_w(X, tn)
        dB = np.abs(np.asarray(m.nodalBoundaryFlux[:no]) - src)
        b_near = globalMax(float(dB[near].max()) if near.any() else 0.0)
        b_deep = globalMax(float(dB[deep].max()) if deep.any() else 0.0)
        src_max = globalMax(float(np.abs(src).max()))
        logEvent("[MMS-B] t=%.9e max|B - ML f_w|: next-to-Dirichlet=%.3e interior=%.3e "
                 "(max|ML f_w|=%.3e)" % (tn, b_near, b_deep, src_max))


LevelModelType = TADR.LevelModel

coefficients = MMSMonitorCoefficients(
    aOfX=None,
    nd=nd,
    V_model=0,                 # Richards is model 0 in pnList
    ME_model=1,
    specified_velocity=False,  # velocity from Richards
    checkMass=checkMass,
    alpha_L=mms_exact.ALPHA_L,
    alpha_T=mms_exact.ALPHA_T,
    Dm=mms_exact.DM,
    porosity=1.0,              # overridden per point by Richards' theta
    diagonal_conductivity=True,
    STABILIZATION_TYPE=5,
    ENTROPY_TYPE="POWER",
    LUMPED_MASS_MATRIX=True,
    FCT=FCT,
    forceStrongConditions=True,
    cE=0.1,
    cK=1.0,
    uL=0.0,
    uR=1.0,
    physicalDiffusion=0.0,
    rho_f=1.0,
    rho_s=1.0 + mms_exact.EPS,
    viscosity_omega_max=mms_exact.OMEGA_MAX,
    mms_source=mms_exact.g,
    EDGE_VELOCITY=True,  # F = A - A^T from Richards' edge velocity
)
coefficients.variableNames = ["u"]

initialConditions = {0: FieldIC(mms_exact.c_e)}
dirichletConditions = {0: dirichlet_everywhere(mms_exact.c_e)}
fluxBoundaryConditions = {0: "setFlow"}
advectiveFluxBoundaryConditions = {0: no_flux_condition}
diffusiveFluxBoundaryConditions = {0: {0: no_flux_condition}}
