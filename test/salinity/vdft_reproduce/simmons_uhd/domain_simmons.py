"""Geometry, mesh and run-configuration for the Simmons benchmark.

Imported by both flow_p.py and transport_p.py, so anything both models must
agree on (domain size, boundary tags, the source strip, the selected case and
discretisation level) lives here and nowhere else.

The mesh is STRUCTURED: with ``domain = None`` and ``polyfile = None``, Proteus
takes the old-style rectangular path in NumericalSolution and builds a regular
triangular grid from (nnx, nny) via
MeshTools.generateTriangularMeshFromRectangularGrid.  That matters for two
reasons:

  * element ordering is deterministic, which is what makes a per-element random
    K0 field reproducible; and
  * the paper's grids are uniform refinements of one rectangle (Eqs. 10/11), so
    an unstructured triangulation would not be the same convergence ladder.

``triangleFlag = 1`` is the union-jack split (alternating diagonals).  A
single-orientation split would put a systematic diagonal bias into a problem
whose entire subject is the direction fingers grow.

Run configuration comes from the environment so the whole ladder runs off one
deck:

    SIMMONS_CASE    UHD | SHD | UMD | SMD             (default SHD)
    SIMMONS_LEVEL   spatial discretisation level 0..5 (default 0, may be
                    fractional: the paper's validation cases ran at 4.5 in x)
    SIMMONS_LEVEL_Z spatial level in z                (default SIMMONS_LEVEL)
    SIMMONS_THETA   temporal discretisation level 0..5 (default 0)
    SIMMONS_SEED    random K0 realisation index       (default 0)

SIMMONS_LEVEL=4.5 SIMMONS_LEVEL_Z=4 is the paper's own SMD grid, 550 x 500; see
cases.spatial_level.
"""

import os

from proteus import *
from proteus.default_p import *

try:
    from . import cases
except ImportError:
    import cases

nd = 2

# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------
CASE_NAME = os.environ.get("SIMMONS_CASE", "SHD").upper()
if CASE_NAME not in cases.CASES:
    raise ValueError("SIMMONS_CASE=%r is not one of %s"
                     % (CASE_NAME, sorted(cases.CASES)))
case = cases.CASES[CASE_NAME]

SPATIAL_LEVEL = float(os.environ.get("SIMMONS_LEVEL", "0"))
SPATIAL_LEVEL_Z = float(os.environ.get("SIMMONS_LEVEL_Z", SPATIAL_LEVEL))
TEMPORAL_LEVEL = int(os.environ.get("SIMMONS_THETA", "0"))
SEED = int(os.environ.get("SIMMONS_SEED", "0"))

#: TADR stabilization.  2 = explicit RKEV entropy viscosity (CFL-bound);
#: 5 = ImplicitEV, backward Euler on the edge-based upwind operator, whose
#: low-order operator is an M-matrix so dt is limited by accuracy and Newton
#: convergence rather than by the CFL condition.  See test/salinity/
#: Elder_problem/stab_5/ for the working configuration this follows.
STAB = int(os.environ.get("SIMMONS_STAB", "5"))

#: Set 0 for the paper's negative control (Fig. 2b,d): a uniform conductivity
#: field, which must NOT produce fingering.  If fingers appear without the
#: random field they are numerical and nothing downstream is meaningful.
RANDOM_K = os.environ.get("SIMMONS_RANDOM_K", "1") not in ("0", "false", "False")

#: One K0 draw per grid RECTANGLE, shared by its two triangles, instead of one
#: per triangle (SIMMONS_K0_PER_CELL=1).  The paper draws one value per HGS
#: brick, i.e. per rectangle of the same Dx x Dz grid; i.i.d. draws per triangle
#: halve the cell area, so the field averaged over a rectangle has 1/sqrt(2) of
#: the paper's standard deviation.  Default off, so every result to date
#: reproduces.
K0_PER_CELL = os.environ.get(
    "SIMMONS_K0_PER_CELL", "0") not in ("0", "false", "False")

#: Variance of ln K0 for the random field (SIMMONS_SIGMA).  Default is the
#: case's own value in cases.py, so every result to date reproduces.
SIGMA_LNK = float(os.environ.get("SIMMONS_SIGMA", case.sigma_lnK))

#: Infiltration record to read instead of data/q_<CASE>.csv (SIMMONS_Q_FILE),
#: for variants of the unrecorded first minutes.  A relative path is taken from
#: the deck directory.  Default unset = data/q_<CASE>.csv.
Q_FILE = os.environ.get("SIMMONS_Q_FILE")

#: Treat the solute source as an INFLOW-ONLY boundary (Signorini) rather than a
#: plain two-way Dirichlet head.  Set 0 to recover the literal reading of the
#: paper's BC, which vents 28-44% of the strip; see RandomFieldCoefficients.preStep.
SOURCE_INFLOW_ONLY = os.environ.get(
    "SIMMONS_SOURCE_INFLOW_ONLY", "1") not in ("0", "false", "False")

#: What the two VERTICAL sides do.
#:
#:   noflow  the benchmark's own closed tank (Fig. 1: left, right and bottom
#:           impermeable).  Default -- every result to date was run this way.
#:   head    the Liu-style fixed water table of
#:           test/salinity/Variable_sat_density_watertable_2m_finer: a constant
#:           total head H = h_ref on both sides, written as psi = h_ref - z,
#:           which is the SAME expression as the initial condition, so the run
#:           starts in equilibrium with its own boundary.  Transport is then
#:           left natural (outflow) on the sides, again as in that deck.
#:
#: This is the only boundary that can hold the water table down once the tank
#: starts filling -- the corner drain cannot (flow_p.py DRAIN puts its capacity
#: at 8.6% of the peak inflow) -- so it is the control for whether the missing
#: fingering is a venting problem.  With 'head' the seepage face is dropped:
#: the sides vent instead, and the reference deck has no seepage nodes.
SIDE_BC = os.environ.get("SIMMONS_SIDE_BC", "noflow").strip().lower()
if SIDE_BC not in ("noflow", "head"):
    raise ValueError("SIMMONS_SIDE_BC must be 'noflow' or 'head'; got %r"
                     % SIDE_BC)

#: Log where along the source strip water actually enters, once per step; see
#: RandomFieldCoefficients.log_source_inflow_profile in flow_p.py.  This is the
#: diagnostic for the empty middle of the plume, and it runs with the Signorini
#: seal either on or off.  Pair it with diagnostics=("boundary_flux",) in Richards and TADR for
#: the boundary-flux decomposition on both sides of the coupling.
SOURCE_PROFILE_DBG = os.environ.get(
    "SIMMONS_SRC_DBG", "0") not in ("0", "false", "False")

dx, dz, nx_elements, nz_elements = cases.spatial_level(SPATIAL_LEVEL,
                                                       SPATIAL_LEVEL_Z)

# ---------------------------------------------------------------------------
# Time-step cap (was gap G7).
#
# Neither model's step controller has an accuracy limit: HeuristicNL_dt_controller
# keys only on the Newton iteration count, and STAB=5 is unconditionally stable,
# so dt doubles until stepExact pins it to the reporting interval.  Measured on
# SMD: dt = 600 s for the whole physical run, i.e. ONE step per reporting time,
# 102 steps total at BOTH level 2 and level 4.
#
# That is a Courant number of 25 (level 2) to 42 (level 4) on the buoyant pore
# velocity, and backward Euler at Courant C smears an advected front with a
# numerical diffusivity
#
#     D_BE = U^2 dt / 2
#
# which is INDEPENDENT OF h -- which is why refining the mesh changed nothing.
# At dt = 600 s, D_BE = 6.7e-6 m^2/s against a physical Dm + alpha_L |q|/theta
# of 1.8e-7, i.e. 37x too diffusive.  The shortest wavelength that can grow
# against a diffusivity D is lambda_c = 2 pi D / U; at D_BE that is 0.28 m,
# larger than the whole plume, so NO finger of any wavelength can form.  The
# paper's Eq. (12) controller is what keeps its dt small enough to finger.
#
# Until Eq. (12) proper is implemented, cap dt at a target Courant number on
# the buoyant pore velocity U = K0 * eps / theta_s.  C = 0.5 puts D_BE at half
# the physical dispersion; C = 1 puts it level with it.
TARGET_COURANT = float(os.environ.get("SIMMONS_COURANT", "0.5"))

#: Buoyant pore velocity scale, m/s.  eps = (rho_s - rho_f)/rho_f.
U_BUOYANT = cases.K0 * (case.rho_s - case.rho_f) / cases.THETA_S

if TARGET_COURANT > 0.0:
    DT_MAX = TARGET_COURANT * dz / U_BUOYANT
else:
    DT_MAX = None                      # 0 or negative disables the cap


from proteus.StepControl import HeuristicNL_dt_controller as _HeuristicNL


class CappedNL_dt_controller(_HeuristicNL):
    """HeuristicNL_dt_controller with an accuracy ceiling on dt.

    The base controller only ever asks "did Newton converge quickly?", which
    under an unconditionally stable scheme is a question with no upper bound.
    Sequential_MinModelStep takes the min of the models' dt_model, so capping
    here caps the system step (SplitOperator.py:317).
    """

    dt_max = DT_MAX

    def choose_dt_model(self):
        _HeuristicNL.choose_dt_model(self)
        if self.dt_max is not None and self.dt_model > self.dt_max:
            self.dt_model = self.dt_max
            self.set_dt_allLevels()
            self.setSubsteps([self.t_model])

# ---------------------------------------------------------------------------
# Domain.  Left as the old-style rectangle so NumericalSolution builds the
# structured mesh; RectangularDomain's 2D tags are bottom/right/top/left =
# 1/2/3/4, which is what the boundary-condition helpers below assume.
# ---------------------------------------------------------------------------
L = (cases.LX, cases.LZ, 1.0)
x0 = (0.0, 0.0, 0.0)
domain = None
polyfile = None

boundaryTags = {"bottom": 1, "right": 2, "top": 3, "left": 4}

# Node counts for the n-files.
nnx = nx_elements + 1
nny = nz_elements + 1
triangleFlag = 1

# Global element count of the structured union-jack grid: two triangles per
# rectangle.  The random K0 field is drawn against this, so it must match what
# the mesher produces.
nElements_global_expected = 2 * nx_elements * nz_elements


def on_source_strip(x, tag, tol=1.0e-8):
    """True on the 0.54 m solute source centred on the top boundary."""
    return (tag == boundaryTags["top"]
            and abs(x[1] - cases.LZ) <= tol
            and cases.SOURCE_X0 - tol <= x[0] <= cases.SOURCE_X1 + tol)


def on_top_outside_source(x, tag, tol=1.0e-8):
    return (tag == boundaryTags["top"]
            and not on_source_strip(x, tag, tol))


# ---------------------------------------------------------------------------
# FEM spaces.  P1 on simplices for both models, and -- critically -- ONE
# quadrature rule shared by both: Richards hands TADR its velocity and water
# content as raw quadrature arrays indexed by eN*nQuad+k, so the two models
# must agree on the rule point-for-point.  TADR.checkCoupledQuadrature enforces
# this at attachModels time.
# ---------------------------------------------------------------------------
femSpaces = {0: C0_AffineLinearOnSimplexWithNodalBasis}
elementQuadrature = SimplexLobattoQuadrature(nd, 1)
elementBoundaryQuadrature = SimplexLobattoQuadrature(nd - 1, 1)
subgridError = None
