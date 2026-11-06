"""TADR (solute transport) half of the Simmons benchmark.

Solves Eq. (4) of Guevara Morel & Graf (2023) for the solute mass fraction,
carried NORMALISED as u = omega/omega_max in [0, 1].

WHY NORMALISED.  The entropy-viscosity and FCT machinery bound the solution to
[uL, uR]; keeping u in [0, 1] means those bounds are the physical ones without
per-case rescaling, and it makes the over/undershoot diagnostic the paper
reports in Fig. 3 (black = u > 1, white = u < 0) read directly off the field.
The density law is unaffected: TADR's rho = rho_f (1 + eps u) with
eps = (rho_s - rho_f)/rho_f and rho_s := rho(omega_max) is algebraically
identical to the paper's Eq. (8) in omega.

DISPERSION.  The kernel builds, at TADR.h:155-163,

    a = theta * rho * [ Dm dIJ + alpha_L vhat_I vhat_J |v|/theta
                                + alpha_T (|v|/theta)(dIJ - vhat_I vhat_J) ]

With v the DARCY flux, theta * alpha_L * |v|/theta = alpha_L |v|, so the
mechanical terms are Eq. (5) term for term.  The molecular term maps as
theta * Dm  <->  phi * S_w * tau * D_d, i.e. Dm := tau * D_d.  The paper never
gives tau (it is absent from Table 2), so cases.TAU = 1.

The leading rho is the non-OB departure described in flow_p.py under G3.
"""

import os

import numpy as np
from proteus import *
from proteus.default_p import *
from proteus import Profiling
from proteus.mprans import TADR

try:
    from .domain_simmons import *
    from . import cases
except ImportError:
    from domain_simmons import *
    import cases

logEvent = Profiling.logEvent

checkMass = False
shockCapturingFactor_tadr = 0.0
lag_shockCapturing_tadr = True

# ---------------------------------------------------------------------------
# Flux-corrected transport.  SIMMONS_FCT=1 turns the limiter on.
#
# This is not a tuning knob -- it decides the ORDER OF THE SCHEME, and with it
# how badly the fingers are damped.  Under STABILIZATION_TYPE=5 (ImplicitEV)
# the residual is
#
#     R_i = ML_i (m_i^{n+1} - m_i^n)/dt + sum_j -a_ij (c_j - c_i)
#     a_ij = max(0,-T_ij) rho_up/rho_f + max(0,-D_ij) >= 0     [TADR.h:1777]
#
# i.e. a FIRST-ORDER upwind M-matrix operator.  Despite the name there is no
# entropy-viscosity limiting in that branch at all (psi/dEV live only in the
# explicit STABILIZATION_TYPE=2 branch).  The FCT post-step is the ONLY route
# back to second order: the branch stores dt*(dH - dLow) with dH = 0 precisely
# so the limiter can antidiffuse the whole of the low-order graph viscosity.
# With FCT off, that antidiffusion is never applied.
#
# The cost of running first order here is set by the TRANSVERSE dispersivity,
# which is tiny (alpha_T = 0.12 mm, 10x smaller than alpha_L).  The upwind
# diffusion |v|h/2 lands right at the physical LONGITUDINAL dispersion -- which
# is what the paper's Pe = dx/alpha_L <= 2 criterion is designed for, so the
# front sharpness is fine -- but it is 8-13x the physical TRANSVERSE value, and
# it is isotropic.  Finger width is set by transverse spreading, so the fingers
# come out fat and few.  HGS uses UNSTABILIZED Galerkin for Eq. (4), no
# artificial viscosity at all, which is exactly why the paper's Fig. 3 shows
# over/undershoots (black u>1, white u<0) at spatial level < 4: they buy
# sharpness with boundedness, this deck buys boundedness with diffusion.
#
# Caveat before trusting FCT=1: the STAB=5 + FCT=True path is the known-suspect
# one (it over-diffused on Elder; see tadr_stab5_fct_kuzmin_textbook_form).
# STAB=2 + FCT=True is the configuration the explicit scheme was designed for.
FCT = os.environ.get("SIMMONS_FCT", "0") not in ("0", "false", "False")

#: Variable viscosity, gap G1 (SIMMONS_VISC=1): mu/mu_0 from Eq. (9) at
#: omega = omega_max*u, evaluated by TADR next to rho and applied by Richards as
#: K_s/mu.  Default off, so every result to date reproduces.
VISC = os.environ.get("SIMMONS_VISC", "0") not in ("0", "false", "False")

#: Coupling ablation (paper Table tab:uhd_coupling), see TADR.Coefficients:
#: SIMMONS_VELOCITY_COUPLING=lagged|eos advects with the Galerkin Darcy velocity
#: instead of the edge fluxes; SIMMONS_CONSERVATIVE_FORM=1 (with a velocity
#: coupling, SIMMONS_FCT=0, serial) solves the conservative solute row.
VELOCITY_COUPLING = os.environ.get("SIMMONS_VELOCITY_COUPLING") or None
CONSERVATIVE_FORM = os.environ.get("SIMMONS_CONSERVATIVE_FORM", "0") not in ("0", "false", "False")

LevelModelType = TADR.LevelModel

coefficients = TADR.Coefficients(
    aOfX=None,
    nd=nd,
    V_model=0,               # Richards is model 0 in pnList
    ME_model=1,
    specified_velocity=False,  # take ('velocity_couple',0) from Richards
    checkMass=checkMass,
    alpha_L=cases.ALPHA_L,
    alpha_T=cases.ALPHA_T,
    Dm=cases.DM,
    porosity=cases.POROSITY,   # overridden per point by Richards' theta
    diagonal_conductivity=True,
    STABILIZATION_TYPE=STAB,
    ENTROPY_TYPE="POWER",
    LUMPED_MASS_MATRIX=True,
    # SIMMONS_FCT (see below).  Default preserves the previous behaviour, FCT
    # off, for every stabilization.
    FCT=FCT,
    forceStrongConditions=True,
    cE=0.1,
    cK=1.0,
    uL=0.0,
    uR=1.0,
    physicalDiffusion=0.0,
    rho_f=case.rho_f,          # 1.0        (dimensionless, rho/rho_0)
    rho_s=case.rho_s,          # 1.235 or 1.048
    viscosity_omega_max=case.omega_max if VISC else 0.0,
    EDGE_VELOCITY=True,  # advect with Richards' rebuilt edge velocity
    VELOCITY_COUPLING=VELOCITY_COUPLING,
    CONSERVATIVE_FORM=CONSERVATIVE_FORM,
    diagnostics=('fct',),
)
coefficients.variableNames = ["u"]

# Carried for the diagnostics and for the phase-2 viscosity work; not read by
# the kernel.
coefficients.omega_max = case.omega_max
coefficients.beta_omega = cases.BETA_OMEGA


# ---------------------------------------------------------------------------
# Initial and boundary conditions.  omega = omega_max (u = 1) is imposed on the
# solute source; every other boundary is a zero-dispersive-mass-flux boundary
# ("Conceptual model").  The tank starts free of solute.
# ---------------------------------------------------------------------------
class ConstantIC(object):
    def __init__(self, value=0.0):
        self.value = value

    def uOfXT(self, x, t):
        return self.value

    def uOfX(self, x):
        return self.value


initialConditions = {0: ConstantIC(0.0)}


def getConcentrationDirichletBCs(x, tag):
    if on_source_strip(x, tag):
        return lambda x, t: 1.0          # u = 1  <=>  omega = omega_max


def getZeroDiffusiveFluxBCs(x, tag):
    """Every boundary but the source is a zero-DISPERSIVE-mass-flux boundary."""
    if on_source_strip(x, tag):
        return None                      # strong Dirichlet there instead
    return lambda x, t: 0.0


def getAdvectiveFluxBCs(x, tag):
    """Advective flux is NATURAL wherever water can cross the boundary.

    The paper specifies only that the non-source boundaries carry zero
    *dispersive* mass flux; the advective flux there is outflow, not zero.
    That distinction is what broke the first run of this deck: in the
    saturated cases the whole top is Dirichlet in head (psi = 0.0042 m under
    the source, psi = 0 either side), so water genuinely leaves through the
    top outside the source.  Clamping the advective flux to zero there while
    dispersion kept delivering solute into those nodes piled mass up with
    nowhere to go -- u climbed past 1 within 30 s and saturated near 4.

    Returning None on the whole top is correct in BOTH cases: it is a genuine
    outflow in the saturated cases, and in the unsaturated ones v.n = 0 there
    anyway, so the natural flux integral vanishes on its own.  (It is also
    what the reference deck's explicit zero amounts to there, since its top
    outside the strip carries no water flux either.)

    The bottom is impermeable to flow in every case, so an explicit zero is
    exact there.  The sides are impermeable too UNLESS SIMMONS_SIDE_BC=head,
    which puts a fixed head on them: water then genuinely crosses, so they
    must be natural/outflow -- the same choice the reference deck makes on
    its own head sides.  Clamping the advective flux to zero on a boundary
    water leaves through is exactly the failure described above.
    """
    if on_source_strip(x, tag):
        return None                      # strong Dirichlet there instead
    if tag == boundaryTags["top"]:
        return None                      # natural / outflow
    if SIDE_BC == "head" and tag in (boundaryTags["left"],
                                     boundaryTags["right"]):
        return None                      # natural / outflow
    return lambda x, t: 0.0              # impermeable walls


dirichletConditions = {0: getConcentrationDirichletBCs}
fluxBoundaryConditions = {0: "setFlow"}
advectiveFluxBoundaryConditions = {0: getAdvectiveFluxBCs}
diffusiveFluxBoundaryConditions = {0: {0: getZeroDiffusiveFluxBCs}}

# Conservation, bounds and cost measures of the run (vdft_metrics.json in the run
# directory); run.sh copies the helper next to the deck.  Changes nothing in the solution.
try:
    import vdft_metrics
    vdft_metrics.install(coefficients)
except ImportError:
    pass
