"""Parameter sets for the Guevara Morel & Graf (2023) benchmark.

Hydrogeology Journal 31:1903-1919, doi:10.1007/s10040-023-02673-y.
Physical basis: Simmons, Pierini & Hutson (2002), Transp Porous Media 47:215-244.

Everything here is SI (m, s, kg).  Times in the paper are quoted in minutes;
they are converted at the bottom of this file so the decks never see minutes.

The four cases are

    UHD  unsaturated high-density    (the calibration case)
    SHD  saturated   high-density    (validation)
    UMD  unsaturated medium-density  (validation)
    SMD  saturated   medium-density  (validation)

Concentration is carried NORMALISED, u = omega/omega_max in [0, 1], so that the
entropy-viscosity / FCT bounds uL=0, uR=1 stay valid without rescaling.  The
density law rho = rho_0 (1 + beta_omega (omega - omega_0)) of Eq. (8) then maps
exactly onto TADR's rho = rho_f (1 + eps u) with eps = (rho_s - rho_f)/rho_f and
rho_s := rho(omega_max).  See RHO_MAX below.

PAPER ERRATA (resolved here, see README.md for the derivation)
  * The paper quotes THREE different element sizes and no two of them agree.
    Eqs. (10)/(11) are authoritative and reconcile all three:
      - "Numerical accuracy" says g_4,4 has dx = 0.002142, dz = 0.002356.  That
        is garbled: 0.002356 is Eq. (10)'s dx AT LEVEL 4 and 0.002142 is the
        same equation at level 4.5.  Eqs. (10)/(11) at level 4 give
        dx = 0.01178/5 = 0.002356 and dz = 0.01060/5 = 0.002120, i.e. exactly
        500 x 500 elements -- which is the only reading that reproduces the
        paper's own 250,000 elements / 502,002 nodes (501 x 501 x 2 layers).
      - Table 2 prints dx = 0.002142, dz = 0.002120.  That pair is NOT g_4,4:
        it is Eq. (10) at 4.5 and Eq. (11) at 4, i.e. 550 x 500, the mixed
        level g_4.5,4 that the validation cases (SHD, UMD, SMD) actually ran
        on.  See spatial_level() below.
  * Table 2's last row gives omega*(theta) = 0.01.  Eq. (12) with theta = 4
    gives omega_max/2**4 = 0.2535/16 = 0.0158438, which is the 0.0158 quoted in
    the running text.  Eq. (12) is authoritative.
"""

import numpy as np

MINUTE = 60.0

# ---------------------------------------------------------------------------
# Geometry (Fig. 1).  1.178 m x 1.06 m tank, 0.54 m solute source centred on
# the top boundary with 0.319 m of buffer either side.
# ---------------------------------------------------------------------------
LX = 1.178
LZ = 1.060
SOURCE_WIDTH = 0.54
SOURCE_X0 = 0.5 * (LX - SOURCE_WIDTH)          # 0.319
SOURCE_X1 = 0.5 * (LX + SOURCE_WIDTH)          # 0.859

# ---------------------------------------------------------------------------
# Calibrated material parameters (Table 2, "Calibrated" column).
# ---------------------------------------------------------------------------
K0 = 0.00112          # m/s     equivalent freshwater hydraulic conductivity
K0_UNCALIBRATED = 0.00163
RHO_0 = 1000.0        # kg/m^3
POROSITY = 0.36       # -
MU_0 = 0.001          # kg/(m s)
ALPHA_L = 0.0012      # m       longitudinal dispersivity
ALPHA_T = 0.00012     # m       transverse dispersivity
VG_ALPHA = 11.6       # 1/m     van Genuchten alpha
VG_N = 2.39           # -       van Genuchten n  (m = 1 - 1/n = 0.581590)
S_WR = 0.045          # -       residual saturation
D_D = 1.2e-9          # m^2/s   free-solution diffusion coefficient

# Tortuosity tau never appears in Table 2.  Eq. (5)'s molecular term is
# phi*tau*Dd; TADR's kernel builds theta*Dm, so Dm := tau*Dd and tau = 1 is the
# only assumption consistent with the paper's silence.
TAU = 1.0
DM = TAU * D_D

VG_M = 1.0 - 1.0 / VG_N

# Richards works in water content, not saturation.
THETA_S = POROSITY                      # 0.36
THETA_R = S_WR * POROSITY               # 0.0162
THETA_SR = THETA_S - THETA_R            # 0.3438

# ---------------------------------------------------------------------------
# Equation of state.  Eq. (8) with omega_0 = 0.
#
# High-density   CaCl2: omega_max = 0.2535  -> rho = 1235 kg/m^3
# Medium-density CaCl2: omega_max = 0.05178 -> rho = 1048 kg/m^3
#
# Both give beta_omega = 0.9273, which is the internal consistency check that
# the paper's two omega_max values and Simmons' two densities agree on one
# linear law.
# ---------------------------------------------------------------------------
OMEGA_MAX_HIGH = 0.2535
OMEGA_MAX_MEDIUM = 0.05178
RHO_MAX_HIGH = 1235.0
RHO_MAX_MEDIUM = 1048.0

BETA_OMEGA_HIGH = (RHO_MAX_HIGH / RHO_0 - 1.0) / OMEGA_MAX_HIGH
BETA_OMEGA_MEDIUM = (RHO_MAX_MEDIUM / RHO_0 - 1.0) / OMEGA_MAX_MEDIUM
assert abs(BETA_OMEGA_HIGH - BETA_OMEGA_MEDIUM) < 2.0e-3, (
    "high- and medium-density cases disagree on beta_omega: %r vs %r"
    % (BETA_OMEGA_HIGH, BETA_OMEGA_MEDIUM))
BETA_OMEGA = BETA_OMEGA_HIGH            # 0.92702


def viscosity_ratio(omega):
    """mu_0 / mu(omega) from Eq. (9), Holzbecher (1998).

    This is the mobility multiplier that gap G1 will apply to K in the Richards
    kernel.  Kept here in Python so the C++ change can be checked against it.
    """
    omega = np.asarray(omega, dtype=float)
    return 1.0 / (1.0 + 1.85 * omega - 4.1 * omega ** 2 + 44.5 * omega ** 3)


# ---------------------------------------------------------------------------
# Initial water levels (Fig. 1).  psi_0 = h_ref - z.
# ---------------------------------------------------------------------------
H_REF_SATURATED = 1.060       # water table at the top of the tank
H_REF_UNSATURATED = 0.620     # water table at z = 0.62 m

# Saturated cases: Dirichlet freshwater head under the source (allows brine to
# pond) and zero head on the rest of the top boundary.
PSI_SOURCE_SATURATED = 0.0042
PSI_TOP_SATURATED = 0.0


class Case(object):
    """One of the four benchmark cases."""

    def __init__(self, name, saturated, omega_max, rho_max, sigma_lnK,
                 T_minutes, t_out_minutes):
        self.name = name
        self.saturated = saturated
        self.omega_max = omega_max
        self.rho_max = rho_max
        #: variance of the random ln(K0) field used to perturb the medium
        self.sigma_lnK = sigma_lnK
        self.T = T_minutes * MINUTE
        self.t_out = np.array(t_out_minutes, dtype=float) * MINUTE

    # -- density law, in the normalised variable u = omega/omega_max ---------
    #
    # Density is carried DIMENSIONLESS, rho/rho_0, as in both working decks
    # (Elder and the Liu variable-density case).  Lengths and times stay SI.
    #
    # This is not cosmetic.  Richards only ever uses density as the ratio
    # rho_local/rho_0 (Richards.h:1562) and as an overall factor on m, a and f,
    # so the pressure-head solution and `velocity_couple` are invariant under
    # the rescaling.  TADR's m = theta*rho*u, f and a all carry the same factor,
    # so its equation is invariant too.  What the rescaling DOES fix is that
    # several TADR diagnostics were written assuming rho ~ 1 and are not
    # normalised by it -- most visibly the cell-based CFL, which reads
    # dt*rho*|v|/(theta*h) and came out ~1000x inflated in SI.
    #
    # omega is still recoverable for the phase-2 viscosity work:
    #     omega = (rho/rho_0 - 1) / beta_omega
    # which is exactly what the dimensionless field gives directly.
    @property
    def rho_f(self):
        return 1.0

    @property
    def rho_s(self):
        """rho(omega_max)/rho_0 -- 1.235 (high) or 1.048 (medium)."""
        return self.rho_max / RHO_0

    # -- initial condition ---------------------------------------------------
    @property
    def h_ref(self):
        return H_REF_SATURATED if self.saturated else H_REF_UNSATURATED

    # -- Eq. (12) time-step target ------------------------------------------
    def omega_star(self, theta_level):
        """Maximum allowed change in solute mass fraction over one step.

        Eq. (12): omega*(theta) = omega_max / 2**theta.  Returned in the
        normalised variable, so it is simply 2**-theta.
        """
        return 2.0 ** (-theta_level)

    def __repr__(self):
        return "<Case %s %s omega_max=%g rho_max=%g sigma=%g T=%gs>" % (
            self.name, "SAT" if self.saturated else "UNSAT",
            self.omega_max, self.rho_max, self.sigma_lnK, self.T)


# Output times.  The 13 times in the "Numerical accuracy" section drive the
# grid-convergence functional J; the penetration-height series in Figs. 7 and 9
# runs to the end of each case.
T_OUT_UHD = [6, 9, 12, 15, 18, 20, 26, 32, 38, 40, 44, 46, 48]

CASES = {
    "UHD": Case("UHD", saturated=False, omega_max=OMEGA_MAX_HIGH,
                rho_max=RHO_MAX_HIGH, sigma_lnK=1.0e-2,
                T_minutes=48, t_out_minutes=T_OUT_UHD),
    "SHD": Case("SHD", saturated=True, omega_max=OMEGA_MAX_HIGH,
                rho_max=RHO_MAX_HIGH, sigma_lnK=1.0e-3,
                T_minutes=60, t_out_minutes=list(range(5, 65, 5))),
    "UMD": Case("UMD", saturated=False, omega_max=OMEGA_MAX_MEDIUM,
                rho_max=RHO_MAX_MEDIUM, sigma_lnK=1.0e-1,
                T_minutes=70, t_out_minutes=list(range(5, 75, 5))),
    "SMD": Case("SMD", saturated=True, omega_max=OMEGA_MAX_MEDIUM,
                rho_max=RHO_MAX_MEDIUM, sigma_lnK=1.0e-3,
                T_minutes=140, t_out_minutes=list(range(10, 150, 10))),
}

# ---------------------------------------------------------------------------
# Discretisation ladder, Eqs. (10)/(11).  Level 4 is the converged grid g_4,4,
# which is what the UHD CALIBRATION ran on.  The three VALIDATION cases -- SHD,
# UMD and SMD -- did not: "the same correlation length as for UHD using grid
# g_4.5,4 was used" ("Parameter validation with SHD, UMD, SMD").  Level 4.5 is
# not on the ladder, but Eq. (10) evaluates there anyway and gives
# dx = 0.01178/5.5 = 0.002142 m, which is exactly the dx that Table 2 and the
# "Application of the benchmark problem" procedure both print.  So the paper's
# SMD grid is 550 columns, not 500.
#
# The level is therefore allowed to differ between x and z, because the paper's
# own SMD prescription does: Table 2's dx = 0.002142 is Eq. (10) at 4.5 and its
# dz = 0.002120 is Eq. (11) at 4.  That pair -- 550 x 500 -- is step 2 of the
# application procedure, i.e. the grid the paper tells a code under validation
# to use, and is what SIMMONS_LEVEL=4.5 SIMMONS_LEVEL_Z=4 builds.
# ---------------------------------------------------------------------------
def spatial_level(varsigma, varsigma_z=None):
    """(dx, dz, nx_elements, nz_elements) at spatial discretisation level.

    ``varsigma_z`` defaults to ``varsigma``, which is the ladder proper; pass it
    to build the paper's mixed-level validation grid (see above).
    """
    if varsigma_z is None:
        varsigma_z = varsigma
    dx = 0.01178 / (varsigma + 1)
    dz = 0.01060 / (varsigma_z + 1)
    nx = int(round(LX / dx))
    nz = int(round(LZ / dz))
    return dx, dz, nx, nz


# Plume penetration height is the z of the deepest point at 70% of omega_max,
# i.e. u = 0.7 in the normalised variable ("Definition of the new benchmark
# problem").
PENETRATION_THRESHOLD = 0.70

# Elevation at which the grid-convergence solute mass flux J is sampled
# ("Numerical accuracy").
J_SAMPLE_Z = 0.8480
