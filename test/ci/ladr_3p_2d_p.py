from proteus import *
from proteus.default_p import *
from proteus import ADR, Domain, AnalyticalSolutions
import numpy
import math

LevelModelType = ADR.LevelModel

"""
Three-phase (fluid/fluid/solid) linear advection-diffusion-reaction at equilibrium in 2D.

Unlike ladr_ss_2d_p.py -- which exercises a single fluid-fluid interface (immersedBoundary /
IFEM / gf_f) -- this file drives BOTH cut-cell mechanisms at once on the same domain:

  * immersedBoundary (gf_f, IFEM, two-sided): a flat fluid-fluid interface, the "free surface".
  * embeddedBoundary (gf_s, cutFEM/Nitsche, one-sided): a circular solid "floating object"
    straddling that interface.

This is the case referred to as `test=13.0` in the design notes (see the ifem-guide skill's
three-phase-status.md); it is numbered `test=1.0` here now that it lives in its own p-file.
"""

## \page Tests Test Problems
# \ref ladr_3p_2d_p.py "Three-phase linear advection-diffusion-reaction at steady state"
#

##\ingroup test
#\file ladr_3p_2d_p.py
#
#\brief Three-phase (fluid/fluid/solid) linear ADR at equilibrium in 2D.

opts = Context.Options([
    ("test", 1.0, "which test to run (default is 1.0):\n"
        "  1.0 = piecewise-linear exactness check, Dirichlet condition on Gamma_s\n"
        "  2.0 = trig MMS rate check, Dirichlet condition on Gamma_s\n"
        "  2.1 = exactly test=2.0's TrigMMS solution and geometry, plus a genuine, constant\n"
        "        FLUX jump [beta du/dn] = test21fluxJump across Gamma_f (2.0's own g has\n"
        "        none -- only a gradient kink from beta itself jumping). [u] stays EXACTLY 0\n"
        "        everywhere on Gamma_f (not just at the Gamma_f/Gamma_s triple point), so this\n"
        "        needs no well-posedness bookkeeping at all -- see TrigMMS.fluxJumpConstant.\n"
        "        Same bulk source as 2.0 (interface_d is affine, so it contributes nothing to\n"
        "        the Laplacian). Isolates immersedBoundary_fluxJump specifically.\n"
        "  3.0 = same geometry and exact solution as 2.0 (trig MMS), but with a Robin\n"
        "        (convective) condition on Gamma_s instead of Dirichlet -- phase-dependent\n"
        "        convective coefficients (robin_h_water on the mua side, robin_h_air on the\n"
        "        mub side) meeting exactly at the Gamma_f/Gamma_s triple point. A genuine\n"
        "        rate check, not just an exactness check, since 2.0's solution is\n"
        "        non-polynomial. Best run with solidShape='line' for a single, clean\n"
        "        triply-cut element."),
    ("fluxLift", True, "add the flux-jump lift J_f*psi to the trial function (False: load term only,\n"
        "        the earlier scheme)"),
    ("PG", False, "Petrov-Galerkin: test with ordinary P1 hat functions instead of "
        "the enriched va/vb IFEM basis. Trial/solution side is unchanged."),
    ("unstructured", False, "use unstructured mesh (default is structured)"),
    ("skew", 0.0001, "skew the domain when using unstructured mesh"),
    ("refinement", 0, "number of times to refine the mesh (default is 0)"),
    ("nCells", 0, "cells per side of the square; overrides refinement when > 0 (e.g. 10*2**k\n"
     "        to match the node counts of the fitted and unstructured meshes)"),
    ("qOrder", 6, "order of quadrature (default is 6)"),
    ("pOrder", 1, "order of FEM basis (default is 1)"),
    ("usePetsc", False, "Use PETSc linear solvers"),
    ("name", "ladr_3p_2d", "name of the test (default is ladr_3p_2d)"),
    ("immersedSCIFEM_switch", 0.0, "switch (0/1) for the SCIFEM interface consistency terms"),
    ("immersedSCIFEM_penalty", 0.0, "coefficient gamma of the interior-penalty stabilization"),
    ("interfaceSlope", 1.0, "slope m of the fluid-fluid interface y = m*x + interfaceOffset."),
    ("interfaceOffset", 1.0/3.0, "intercept c of the interface y = m*x + c.\n"
        "Two regimes, and BOTH must work:\n"
        "  * mesh-aligned (e.g. m=0, c=0, or any dyadic c with m=0 or 1): the interface runs\n"
        "    along element edges. gf_f produces no genuinely cut element and instead takes the\n"
        "    degenerate exact.edge/exact.corner branches in ADR.h. This is a real code path that\n"
        "    needs coverage, but it does NOT exercise the two-sided IFEM basis (va/vb are never\n"
        "    built) and the kink is representable by ordinary continuous P1.\n"
        "  * non-aligned (the default m=1, c=1/3): elements are genuinely cut and the IFEM\n"
        "    enrichment is exercised. Note the mesh has nnx-1 = 2^(refinement+2) cells over\n"
        "    [-1,1], so its node lines and 45-degree element diagonals all sit at dyadic\n"
        "    offsets; c must be non-dyadic to stay off them. 1/3 never is.\n"
        "The default is non-aligned because that is the weaker-covered path; run the aligned\n"
        "configuration too."),
    ("forceStrongDirichlet", True, "apply the exterior Dirichlet BC strongly (diagnostic knob)"),
    ("debugExactInit", False,
        "diagnostic only: seed the Newton initial guess with the nodal interpolant of the\n"
        "exact solution. Since Newton converges in one step for this linear problem, the\n"
        "residual reported as 'Newton it 0' is then exactly R(u_exact) -- the consistency\n"
        "residual of the scheme, decoupled from whatever the linear solve converges to."),
    ("mua", 1.0, "diffusion in Omega- (water)"),
    ("mub", 5.0, "diffusion in Omega+ (air)"),
    ("solidShape", "circle", "shape of the embedded solid: circle or square. A square has a\n        straight boundary, which a linear level set represents exactly; a circle is\n        approximated by a chord in every cut element."),
    ("solidSlope", -1.0, "slope of the straight solid boundary when solidShape is 'line'"),
    ("solidOffset", 0.1, "intercept of the straight solid boundary when solidShape is 'line'"),
    ("solidRadius", 0.3, "radius of the embedded (solid) circular object"),
    ("solidCenterY", 1.0/3.0, "y-coordinate of the solid object's center. The solid sits at "
                              "(0, solidCenterY), so setting this equal to interfaceOffset puts its "
                              "centre on the interface and makes it straddle."),
    ("embeddedPenalty", 10.0, "Nitsche Dirichlet penalty on the embedded (solid) boundary. A"
        " penalty this large relative to h grows unboundedly as O(1/h) (see updateEmbeddedBoundaryTerms's"
        " embeddedBoundary_penalty/h_phi), which is required for coercivity but also degrades"
        " conditioning at fine mesh; 100.0 (the old default) visibly softened the L-infinity"
        " rate at the finest meshes tested (down to ~1.7 at h=1/256) without affecting L2 or"
        " Newton's one-step convergence. 10.0 removes that softening in every test checked"
        " (see claude_addition.tex Sec. Numerical validation) while still converging cleanly;"
        " it has not been pushed lower to find where stability actually breaks."),
    ("embeddedGhostPenalty", 0.1, "ghost penalty coefficient on cutfem (solid) cut edges"),
    ("robin_h_water", 600.0, "test=3.0 only: convective heat-transfer coefficient on the water\n"
        "        (mua) side of Gamma_s, W/(m^2 K) -- natural convection in water\n"
        "        (Incropera & DeWitt, Fundamentals of Heat and Mass Transfer, Table 1.1)."),
    ("robin_h_air", 15.0, "test=3.0 only: convective heat-transfer coefficient on the air\n"
        "        (mub) side of Gamma_s, W/(m^2 K) -- natural convection in air, same source.\n"
        "        Water convects roughly 40x more effectively than air, so this pair is a real\n"
        "        physical contrast, not an arbitrary choice of test data."),
    ("test21fluxJump", 0.2, "test=2.1 only: constant flux jump [beta du/dn] = this value\n"
        "        across Gamma_f, added to test=2.0's TrigMMS solution with [u] left exactly at\n"
        "        0. Default 0.2 matches the same order of magnitude as ladr_ss_2d_p.py's own\n"
        "        flux-jump tests (e.g. test=2.0 there uses 2C=0.2)."),
    ("test1fluxJumpConstant", 0.0, "DEBUGGING ONLY, test=1.0: a genuine, constant FLUX jump\n"
        "        [beta du/dn] = this value, with [u] left EXACTLY at 0. Both branches stay\n"
        "        affine (a plain slope change of FloatingObjectExample's outer branch), so this\n"
        "        is a genuine machine-precision exactness check that isolates\n"
        "        immersedBoundary_fluxJump specifically (the Dirac-delta D_f term in ADR.h),\n"
        "        NOT the JA/JB solution-jump machinery -- works with any solidShape, no\n"
        "        triple-point bookkeeping needed, since it never touches the two-branch\n"
        "        basis's own value-continuity pinning at all. Default 0.0 exactly reproduces\n"
        "        test=1.0."),
    ("test21solutionJump", 0.0, "test=2.1 only: a genuine, constant SOLUTION (value) jump\n"
        "        [u] = u_outer - u_inner = this value across Gamma_f, added as a flat offset to\n"
        "        test=2.0's TrigMMS outer branch (u_outer += this value) on top of\n"
        "        test21fluxJump's own flux jump -- independent knobs, since the offset is\n"
        "        spatially constant and so changes neither branch's gradient nor the source\n"
        "        term. Exercises the JA/JB solution-jump machinery specifically (the\n"
        "        branch-basis reconstruction driven by immersedBoundary_solutionJump_nodes in\n"
        "        ADR.h), the complement of test21fluxJump, which isolates the Dirac-delta\n"
        "        flux-jump term instead. Set test21fluxJump=0.0 alongside this to isolate a pure\n"
        "        solution jump. Default 0.0 exactly reproduces test=2.1's existing\n"
        "        (already-validated) [u]=0 baseline."),
    ("test1solutionJump", 0.0, "DEBUGGING ONLY, test=1.0: a genuine, constant SOLUTION jump\n"
        "        [u] = this value, with [beta du/dn] left at whatever test1fluxJumpConstant\n"
        "        already gives (independent knobs). Both branches stay exactly affine (a flat\n"
        "        offset added to FloatingObjectExample's outer branch), so with\n"
        "        debugExactInit=True this is a genuine machine-precision exactness check of the\n"
        "        JA/JB solution-jump machinery specifically -- unlike test21solutionJump, whose\n"
        "        TrigMMS solution is non-polynomial and so can never distinguish real\n"
        "        discretization error from a defect that happens to be small. Run with a real\n"
        "        solidShape (e.g. solidShape='line' solidSlope=-0.5 solidOffset=0.3333333333333333)\n"
        "        to put a masked/triply-cut node in play. Default 0.0 exactly reproduces\n"
        "        test=1.0.")
])

name = opts.name
nd = 2
L=(2.0,2.0)
x0 = (-1.0,-1.0)
if opts.unstructured:
    L=(2.0,2.0+opts.skew)#throw off rectangular domain
domainR = Domain.RectangularDomain(L=L,x=x0,name="adr",units="m")
domainR.writePoly("ladr_3p_2d_p")
domainUS = Domain.PlanarStraightLineGraphDomain("ladr_3p_2d_p")
domainUS.boundaryTags = domainR.boundaryTags
domain = domainR
if opts.unstructured:
    domain = domainUS
a0=1.0
b0 = 0.0
A0_1c={0:numpy.array([[a0,0],[0,a0]])}
B0_1c={0:numpy.array([0.0,b0])}
C0_1c={0:0.0}
M0_1c={0:0.0}

# --- interface geometry -------------------------------------------------------------
# The fluid-fluid interface is the line  y = m*x + c.  Work in the orthonormal rotated
# frame (d, t) attached to it:
#
#     d(x,y) = (y - m*x - c)/sqrt(1+m^2)     signed distance, |grad d| = 1
#     t(x,y) = (x + m*y)/sqrt(1+m^2)         tangential coordinate, |grad t| = 1
#
# grad d . grad t = 0, so the Laplacian is just d_dd + d_tt in these coordinates, which
# is what makes the manufactured source below come out as a single expression.
_m = opts.interfaceSlope
_c = opts.interfaceOffset
_L = math.sqrt(1.0 + _m*_m)

def interface_d(x):
    """Signed distance to the interface. Negative = Omega^- (mua side)."""
    return (x[1] - _m*x[0] - _c)/_L

def interface_t(x):
    """Coordinate along the interface."""
    return (x[0] + _m*(x[1] - _c))/_L

# unit normal, = grad(interface_d), pointing from Omega^- into Omega^+
interface_normal = (-_m/_L, 1.0/_L, 0.0)
# unit tangent, = grad(interface_t); orthogonal to interface_normal by construction.
interface_tangent = (1.0/_L, _m/_L, 0.0)

class FloatingObjectExample(AnalyticalSolutions.SteadyState):
    """Floating object on a flat free surface.

    A linear-in-y solution on each side of the horizontal interface at y = 0:

        u = -y                       for y <= 0   (Omega^-, the denser "water" phase)
        u = -y*betaMinus/betaPlus    for y >  0   (Omega^+, the lighter "air" phase)

    The value is continuous at y = 0 (both branches give 0), and the ratio in the outer
    branch makes the diffusive flux continuous as well:

        betaMinus * d(u^-)/dy = -betaMinus = betaPlus * d(u^+)/dy

    So this is a no-jump interface problem ([u] = 0, [beta du/dn] = 0), exercising the
    two-sided IFEM basis without any of the JA/JB prescribed-jump machinery -- the same
    style as test=8.0 in ladr_ss_2d_p.py. Both branches are linear, so the source term
    vanishes identically (f = 0) in both phases.
    """
    def __init__(self, betaMinus=1.0, betaPlus=5.0, fluxJumpConstant=0.0, solutionJumpConstant=0.0):
        self.betaMinus = betaMinus
        self.betaPlus = betaPlus
        # Debugging-only knob (default 0.0, true no-op): a genuine, constant FLUX jump
        # [beta du/dn] = fluxJumpConstant, with [u] left EXACTLY at 0. Generalizes the outer
        # slope: the original outer branch is -d*betaMinus/betaPlus, chosen so
        # betaPlus*d(u^+)/dd = -betaMinus = betaMinus*d(u^-)/dd (zero flux jump, see class
        # docstring); replacing betaMinus with (betaMinus - fluxJumpConstant) in that same
        # ratio instead gives
        #     betaPlus*d(u^+)/dd - betaMinus*d(u^-)/dd
        #   = -(betaMinus - fluxJumpConstant) - (-betaMinus) = fluxJumpConstant
        # exactly, by construction -- while u^+ is still -d*(...), so u^+(0) = u^-(0) = 0,
        # value continuity is untouched. Both branches remain affine (a constant slope
        # change), so this is tested purely via immersedBoundary_fluxJump -- the Dirac-delta
        # (equivalent-polynomial D_f) term in ADR.h, r += jump_flux*D_f -- NOT via JA/JB at
        # all, isolating that mechanism specifically. On a triply-cut element this D_f is the
        # SAME comp_DA_cache/comp_DB_cache composite Dirac weight the bulk term already uses
        # (see ADR.h's comp_valid/composite::dirac_weights), so this is a direct check of
        # whether that Gamma_s-mask-aware weight is actually being used correctly there.
        self.fluxJumpConstant = fluxJumpConstant
        # Debugging-only knob (default 0.0, true no-op): a genuine, constant SOLUTION jump
        # [u] = solutionJumpConstant, added as a flat offset to the outer branch only (inner
        # untouched) -- same reasoning as TrigMMS's own solutionJumpConstant below. Since both
        # branches here are exactly affine (piecewise-linear MMS, u is exactly representable
        # by P1), this is the sharpest possible test of the JA/JB machinery: with
        # debugExactInit=True, any nonzero "Newton it 0" residual is unambiguously a scheme
        # defect, not approximation error -- unlike test=2.1's TrigMMS, which is
        # non-polynomial and so can never distinguish "small discretization error" from "a
        # real defect that happens to be small" this way.
        self.solutionJumpConstant = solutionJumpConstant
        super(FloatingObjectExample, self).__init__()
    def uOfX_inner(self, x):  # Omega- : d <= 0
        return -interface_d(x)
    def uOfX_outer(self, x):  # Omega+ : d > 0
        outer_slope = (self.betaMinus - self.fluxJumpConstant)/self.betaPlus
        return -interface_d(x)*outer_slope + self.solutionJumpConstant
    def uOfX(self, x):
        if interface_d(x) <= 0.0:
            return self.uOfX_inner(x)
        else:
            return self.uOfX_outer(x)
    def solutionJump(self, x):
        return self.solutionJumpConstant
    def fluxJump(self, x):
        return self.fluxJumpConstant
    def flux_dot_n(self, x, nx, ny):
        """beta*grad(u).n at x, for an arbitrary direction (nx,ny) -- the SAME value on
        either branch, since this solution's flux is continuous across Gamma_f by
        construction (see the class docstring). grad(-d) = -interface_normal, constant."""
        return -self.betaMinus*(interface_normal[0]*nx + interface_normal[1]*ny)

class TrigMMS(AnalyticalSolutions.SteadyState):
    """Manufactured solution for the same 3-phase geometry, non-polynomial so that
    convergence *rates* are measurable (FloatingObjectExample is piecewise linear and
    therefore lies in the P1 space, which makes it an exactness check but useless for
    reading a rate).

    Let g(x,y) = sin(pi*interface_d(x)) * cos(pi*interface_t(x)) and put

        u = g / beta      on each side, i.e. u^- = g/mua, u^+ = g/mub

    (d, t), not (x, y), because the fluid-fluid interface is the general line
    y = interfaceSlope*x + interfaceOffset (diagonal by default -- see the geometry
    comment above), not the coordinate axis y = 0. g must vanish on Gamma_f *however it
    is oriented*, and sin(pi*d) does that unconditionally, since d = 0 is the definition
    of Gamma_f -- sin(pi x)*sin(pi y) would only have done so by coincidence, for the
    special case of an axis-aligned interface. cos(pi*t) supplies genuine variation
    *along* the interface, so the test isn't secretly constant in that direction. Interface
    conditions on Gamma_f are both homogeneous, by construction:

      * [u] = 0 because g = 0 wherever d = 0, i.e. on Gamma_f itself, so both branches
        agree there, for any slope/offset;
      * [beta du/dn] = 0 because beta*grad(u) = grad(g) on *either* side, the beta
        cancelling exactly -- and since (d, t) is an orthonormal rotation of (x, y)
        (|grad d| = |grad t| = 1, grad d . grad t = 0), du/dn along Gamma_f's own normal
        is exactly du/dd, with no extra scaling to account for.

    The solution still has a genuine gradient kink at Gamma_f (du^-/dn = g_d/mua vs
    du^+/dn = g_d/mub), so the two-sided IFEM basis is genuinely exercised -- same
    structure as test=8.0 in ladr_ss_2d_p.py, but generalized to a possibly-diagonal
    interface.

    The source is the same expression on both sides: with beta constant per phase,
    beta grad u = grad g, so -div(beta grad u) = -laplacian(g). Because (d, t) is an
    orthonormal rotation of (x, y), the Laplacian keeps the same form in either
    coordinate pair (d_dd + d_tt = d_xx + d_yy), so -laplacian(g) = 2 pi^2 g exactly as
    it would if g were written directly in x, y.

    Unlike the axis-aligned special case, g is generally *not* zero on the outer boundary
    of [-1,1]^2 -- d and t at the domain edges are not simply +-1 once the interface is
    diagonal, so the exterior Dirichlet data g is a genuine, nonzero function of position,
    not homogeneous.
    """
    def __init__(self, betaMinus=1.0, betaPlus=5.0, fluxJumpConstant=0.0, solutionJumpConstant=0.0):
        self.betaMinus = betaMinus
        self.betaPlus = betaPlus
        # Debugging-only knob (default 0.0, true no-op): a genuine, constant FLUX jump
        # [beta du/dn] = fluxJumpConstant, with [u] left EXACTLY at 0 -- not just at the
        # Gamma_f/Gamma_s triple point, everywhere on Gamma_f, so this needs no triple-point
        # bookkeeping at all (unlike a value jump). Adds fluxJumpConstant*d/betaPlus to the
        # outer branch: interface_d(x) is affine, so it vanishes at d=0 exactly like _g does
        # (keeping [u]=0 identically), while contributing a constant to the outer branch's
        # own normal derivative:
        #     betaPlus*d(u^+)/dd = d(g)/dd + betaPlus*(fluxJumpConstant/betaPlus)
        #                        = d(g)/dd + fluxJumpConstant
        # so [beta du/dn] = betaPlus*d(u^+)/dd - betaMinus*d(u^-)/dd = fluxJumpConstant
        # exactly, the d(g)/dd pieces cancelling as in the zero-jump case. Since d is affine
        # (harmonic, zero Laplacian), the existing source term 2*pi^2*g needs no change either
        # -- see f(x) below.
        self.fluxJumpConstant = fluxJumpConstant
        # Debugging-only knob (default 0.0, true no-op): a genuine, constant SOLUTION jump
        # [u] = solutionJumpConstant, with [beta du/dn] left exactly at whatever
        # fluxJumpConstant already gives (independent knobs -- see below). Added as a flat
        # offset to the outer branch ONLY, after dividing by betaPlus (not folded into g),
        # so the jump value itself is clean and beta-independent:
        #     u_outer(d=0) - u_inner(d=0) = [0 + 0]/betaPlus + solutionJumpConstant - 0
        #                                 = solutionJumpConstant
        # exactly (both g and the fluxJumpConstant*d term already vanish at d=0). A constant
        # offset has zero gradient and zero Laplacian, so neither branch's flux (already
        # handled by fluxJumpConstant above) nor the source term f = 2*pi^2*g needs any
        # change -- this genuinely isolates the JA/JB solution-jump reconstruction in ADR.h
        # from the separate Dirac-delta flux-jump term.
        self.solutionJumpConstant = solutionJumpConstant
        super(TrigMMS, self).__init__()
    def _g(self, x):
        # sin(pi d) vanishes on the interface (d = 0), so [u] = 0 there; the cos(pi t)
        # factor gives variation *along* the interface as well as across it.
        return math.sin(math.pi*interface_d(x))*math.cos(math.pi*interface_t(x))
    def uOfX_inner(self, x):  # Omega- : d <= 0
        return self._g(x)/self.betaMinus
    def uOfX_outer(self, x):  # Omega+ : d > 0
        return (self._g(x) + self.fluxJumpConstant*interface_d(x))/self.betaPlus + self.solutionJumpConstant
    def uOfX(self, x):
        if interface_d(x) <= 0.0:
            return self.uOfX_inner(x)
        else:
            return self.uOfX_outer(x)
    def solutionJump(self, x):
        # [u] = u_outer - u_inner, the sign convention immersedBoundary_solutionJump_nodes
        # expects in ADR.h (cross-checked against ladr_ss_2d_p.py's LevequeLiExample3, whose
        # own solutionJump lambda matches its uOfX_outer - uOfX_inner exactly).
        return self.solutionJumpConstant
    def fluxJump(self, x):
        return self.fluxJumpConstant
    def flux_dot_n(self, x, nx, ny):
        """beta*grad(u).n at x, for an arbitrary direction (nx,ny). Assumes fluxJumpConstant
        == 0 (true whenever this is actually called -- test=3.0's Robin construction always
        builds TrigMMS with the default, and the flux is then the SAME value on either branch,
        beta*grad(u) = grad(g), beta cancelling exactly, see the class docstring), unlike
        FloatingObjectExample's this genuinely varies with x. Using
        grad(g) = pi*cos(pi d)*cos(pi t)*grad(d) - pi*sin(pi d)*sin(pi t)*grad(t) and
        dotting with (nx,ny) via grad(d).( nx,ny)=interface_normal.(nx,ny) and likewise
        for grad(t)/interface_tangent (both unit vectors, not generally (nx,ny) itself,
        e.g. when (nx,ny) is Gamma_s's own normal rather than Gamma_f's)."""
        assert self.fluxJumpConstant == 0.0, "flux_dot_n assumes a single, branch-independent flux"
        d = interface_d(x)
        t = interface_t(x)
        dgdd = math.pi*math.cos(math.pi*d)*math.cos(math.pi*t)
        dgdt = -math.pi*math.sin(math.pi*d)*math.sin(math.pi*t)
        n_dot_normal = interface_normal[0]*nx + interface_normal[1]*ny
        n_dot_tangent = interface_tangent[0]*nx + interface_tangent[1]*ny
        return dgdd*n_dot_normal + dgdt*n_dot_tangent

# Interface material parameters (diffusion on each side of the fluid-fluid interface).
# a(x) below uses these as the source of truth.
mua = opts.mua   # Omega- : water (immersedBoundary sdf < 0)
mub = opts.mub   # Omega+ : air   (immersedBoundary sdf > 0)
jf  = 0.0   # no prescribed solution jump across the interface

if opts.test == 1.0:
    ans = FloatingObjectExample(betaMinus=mua, betaPlus=mub, fluxJumpConstant=opts.test1fluxJumpConstant,
                                solutionJumpConstant=opts.test1solutionJump)
elif opts.test == 2.0:
    ans = TrigMMS(betaMinus=mua, betaPlus=mub)
elif opts.test == 2.1:
    ans = TrigMMS(betaMinus=mua, betaPlus=mub, fluxJumpConstant=opts.test21fluxJump,
                  solutionJumpConstant=opts.test21solutionJump)
elif opts.test == 3.0:
    # Same exact solution as test=2.0 (TrigMMS) -- only Gamma_s's condition type changes,
    # from Dirichlet to Robin. Reusing a solution already known to reproduce correctly
    # isolates "did the Robin implementation break anything" from "is this a new, untested
    # manufactured solution"; using TrigMMS rather than test=1.0's linear solution means
    # this is a genuine convergence-RATE check, not just an exactness sanity check (test=1.0
    # is exact for P1 regardless of whether Robin is implemented correctly).
    ans = TrigMMS(betaMinus=mua, betaPlus=mub)
else:
    assert False, "Unknown test %s" % opts.test

analyticalSolution = {0:ans}
initialConditions = {0: ans} if opts.debugExactInit else None

def a(x):
    # Piecewise-constant diffusion across the free surface. The value inside the solid is
    # irrelevant: the bulk residual there is masked out by H_s.
    if interface_d(x) <= 0.0:
        return numpy.array([[mua,0.0],[0.0,mua]])
    else:
        return numpy.array([[mub,0.0],[0.0,mub]])

def f(x):
    # f = -div(beta grad u), the same expression on both sides of the interface for
    # both cases (see the analytical-solution docstrings).
    if opts.test in (2.0, 2.1, 3.0):
        return 2.0*math.pi**2*ans._g(x)
    # test 1.0: u is linear in each phase, so -div(beta grad u) = 0 off the interface.
    return 0.0

# --- test=3.0: Robin (convective) condition on Gamma_s -----------------------------------
# Physical law:  mu * du/dn = -h * (u - u_infinity)  on Gamma_s, n the fluid's own outward
# normal (out of the fluid, into the solid). test=3.0 reuses test=2.0's TrigMMS solution
# unchanged -- a linear (test=1.0) solution would only give a sanity/exactness check (exact
# for P1 regardless of whether the Robin condition is implemented correctly), not a genuine
# convergence RATE check, which needs a non-polynomial u -- so u_infinity is not free data
# but is BACK-SOLVED from the law so that this particular u satisfies it exactly:
#
#     u_infinity = u + (mu * du/dn) / h
#
# This is the standard method-of-manufactured-solutions move for a mixed/Robin boundary: h
# is the genuine physical parameter (see robin_h_water/robin_h_air above), u_infinity is
# whatever ambient value makes the chosen u consistent with it. TrigMMS's du/dn genuinely
# varies with position (unlike test=1.0's constant-per-phase gradient), so u_infinity is a
# nontrivial function along Gamma_s -- not a single far-field number, but that is normal for
# a manufactured Robin target and does not make it less physical: only the material
# parameters (mua, mub, h_water, h_air) are the modeling choices; u_infinity is derived.
# ans.flux_dot_n(x, nx, ny) supplies mu*grad(u).n generically -- the SAME value on either
# branch for both exact solutions used here (flux is continuous by construction, see each
# class's docstring), so one call covers both embeddedBoundary_robin_uInf_inner/_outer.

def _fluid_outward_normal_on_Gs(x):
    """The fluid's own outward normal on Gamma_s: points OUT of the fluid, INTO the solid.
    solidBoundary_sdf's own normal points the other way (out of the solid, into the fluid --
    see its docstring), so flip it. Works for any solidShape, not just 'line'."""
    _, n = solidBoundary_sdf(x, 0.0)
    return (-n[0], -n[1])

def embeddedBoundary_robin_uInf_inner(x, t):
    nx, ny = _fluid_outward_normal_on_Gs(x)
    flux = ans.flux_dot_n(x, nx, ny)   # mua * du_a/dn
    return ans.uOfX_inner(x) + flux/opts.robin_h_water

def embeddedBoundary_robin_uInf_outer(x, t):
    nx, ny = _fluid_outward_normal_on_Gs(x)
    flux = ans.flux_dot_n(x, nx, ny)   # mub * du_b/dn
    return ans.uOfX_outer(x) + flux/opts.robin_h_air

aOfX = {0:a}; fOfX = {0:f}

solidCenter = (0.0, opts.solidCenterY)

def immersedBoundary_sdf(x,t):
    """Fluid-fluid interface (gf_f): the flat free surface at y = 0.

    Sign convention matches ladr_ss_2d_p.py: sdf < 0 is Omega- (mua, weighted by ImH_f in
    ADR.h) and sdf > 0 is Omega+ (mub, weighted by H_f). The normal is grad(sdf).
    """
    return interface_d(x), interface_normal

def solidBoundary_sdf_line(x,t):
    """Half-plane solid bounded by one straight line: y = solidSlope*x + solidOffset.

    The cleanest possible 3-phase geometry. Both level sets are straight, so the linear
    interpolant the code builds from nodal values represents each of them *exactly* -- no
    curvature error, and no corners either (unlike the square). The two lines cross at a single
    point, so exactly one element in the mesh is triply cut. Any error that survives here is
    scheme error, with geometry ruled out by construction.

    Solid is below the line (sdf < 0), fluid above (sdf > 0), matching ADR.h's convention.
    """
    L = math.sqrt(1.0 + opts.solidSlope**2)
    sdf = (x[1] - opts.solidSlope*x[0] - opts.solidOffset)/L
    n = (-opts.solidSlope/L, 1.0/L, 0.0)
    return sdf, n

def solidBoundary_sdf_square(x,t):
    """Axis-aligned square solid, half-width solidRadius, centred on solidCenter.

    Its boundary is straight, so the linear level set the code builds from nodal values
    represents it *exactly* -- unlike a circle, whose boundary is replaced by a chord in every
    cut element. Useful for separating scheme error from geometry error.
    """
    dx = x[0] - solidCenter[0]
    dy = x[1] - solidCenter[1]
    if abs(dx) >= abs(dy):
        sdf = abs(dx) - opts.solidRadius
        n = (1.0 if dx > 0.0 else -1.0, 0.0, 0.0)
    else:
        sdf = abs(dy) - opts.solidRadius
        n = (0.0, 1.0 if dy > 0.0 else -1.0, 0.0)
    return sdf, n

def solidBoundary_sdf_circle(x,t):
    """Fluid-solid boundary (gf_s): the circular floating object.

    Sign convention: in ADR.h the active-fluid mask H_s is the Heaviside of this level set,
    so sdf > 0 is the *fluid* (assembled) region and sdf < 0 is the solid (masked out).
    The object is the disk, so sdf = r - solidRadius is positive outside it, and the normal
    n = grad(sdf) points out of the solid into the fluid. This is the same orientation as
    the standalone cutFEM test on cekees/add_cutfem_3d, where the fluid is inside the circle
    and sdf = radius - r.
    """
    xr = x[0] - solidCenter[0]
    yr = x[1] - solidCenter[1]
    r = (xr**2 + yr**2)**0.5
    if r > 1.0e-16:
        n = (xr/r,yr/r,0.)
    else:
        n = (1.0,0.0,0.0)
    sdf = r - opts.solidRadius
    return sdf,n

solidBoundary_sdf = {"square": solidBoundary_sdf_square,
                     "line":   solidBoundary_sdf_line}.get(opts.solidShape, solidBoundary_sdf_circle)

def immersedBoundary_u(x,t):
    return ans.uOfX(x)

# test=1.0 (only with test1fluxJumpConstant != 0, a debugging knob) and test=2.1 (its own
# test21fluxJump, nonzero by default, and test21solutionJump, a debugging knob defaulting to
# 0.0): prescribed flux-jump/solution-jump data, see FloatingObjectExample/TrigMMS's own
# fluxJump/solutionJump methods. Every other test leaves both at their ADR.Coefficients
# default of None, i.e. [u] = 0 and [beta du/dn] = 0, exactly as before; test=1.0 at its own
# default (test1fluxJumpConstant=0.0) and test=2.1 at test21solutionJump=0.0 wire the same
# always-zero functions in, which is a true no-op (solutionJump/fluxJump both return 0.0, same
# as the None-case zero-initialized arrays), so the ordinary test=1.0 path and test=2.1's
# existing [u]=0 baseline are unchanged.
immersedBoundary_solutionJump = (lambda x, t: ans.solutionJump(x)) if opts.test in (1.0, 2.1) else None
immersedBoundary_fluxJump = (lambda x, t: ans.fluxJump(x)) if opts.test in (1.0, 2.1) else None

def embeddedBoundary_u(x,t):
    # Manufactured weak-Dirichlet target on the solid surface. Always the true exact
    # solution, test=3.0 included: embeddedBoundary_u_inner/_outer below are also read
    # directly (by name) to pin masked triply-cut DOFs to the true solution, a mechanism
    # unrelated to Gamma_s's boundary condition type -- see the note in ADR.Coefficients.
    # test=3.0's Robin target is carried on its own, separate embeddedBoundary_robin_uInfinity*
    # fields instead of overloading this one.
    return ans.uOfX(x)

def embeddedBoundary_u_inner(x,t):
    # Water-branch target, evaluated by its own formula everywhere (analytic continuation),
    # not just where the water actually is. See the note in ADR.Coefficients.
    return ans.uOfX_inner(x)

def embeddedBoundary_u_outer(x,t):
    # Air-branch target, likewise continued over the whole element.
    return ans.uOfX_outer(x)


coefficients = ADR.Coefficients(aOfX=aOfX,fOfX=fOfX,velocity=B0_1c[0],nc=1,nd=nd,
                                forceStrongDirichlet=opts.forceStrongDirichlet,
                                mua=mua,
                                mub=mub,
                                jf=jf,
                                embeddedBoundary=True,
                                embeddedBoundary_penalty=opts.embeddedPenalty,
                                embeddedBoundary_ghost_penalty=opts.embeddedGhostPenalty,
                                embeddedBoundary_sdf=solidBoundary_sdf,
                                embeddedBoundary_u=embeddedBoundary_u,
                                embeddedBoundary_u_inner=embeddedBoundary_u_inner,
                                embeddedBoundary_u_outer=embeddedBoundary_u_outer,
                                embeddedBoundary_robin=(opts.test == 3.0),
                                embeddedBoundary_robin_h_inner=opts.robin_h_water,
                                embeddedBoundary_robin_h_outer=opts.robin_h_air,
                                embeddedBoundary_robin_uInfinity_inner=embeddedBoundary_robin_uInf_inner,
                                embeddedBoundary_robin_uInfinity_outer=embeddedBoundary_robin_uInf_outer,
                                immersedBoundary=True,
                                immersedBoundary_sdf=immersedBoundary_sdf,
                                immersedBoundary_u=immersedBoundary_u,
                                immersedBoundary_solutionJump=immersedBoundary_solutionJump,
                                immersedBoundary_fluxJump=immersedBoundary_fluxJump,
                                immersedBoundary_fluxJumpLift=opts.fluxLift,
                                immersedBoundary_penalty=0.0,
                                immersedSCIFEM_switch=opts.immersedSCIFEM_switch,
                                immersedSCIFEM_penalty=opts.immersedSCIFEM_penalty,
                                PG=opts.PG,
                                analyticalSolution=analyticalSolution)

def getDBC(x,flag):
    if flag in [domain.boundaryTags['left'], domain.boundaryTags['right'],
                domain.boundaryTags['bottom'], domain.boundaryTags['top']]:
        return lambda x,t: ans.uOfX(x)

dirichletConditions = {0:getDBC}

fluxBoundaryConditions = {0:'noFlow'}

def getFlux(x,flag):
    return lambda x,t: 0.0

advectiveFluxBoundaryConditions =  {0:getFlux}

diffusiveFluxBoundaryConditions = {0:{0:getFlux}}
