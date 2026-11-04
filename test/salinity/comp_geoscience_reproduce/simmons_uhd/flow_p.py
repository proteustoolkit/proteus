"""Richards (variably saturated flow) half of the Simmons benchmark.

Solves Eqs. (1) and (3) of Guevara Morel & Graf (2023) in equivalent freshwater
pressure head psi_0, coupled to the TADR transport model, which supplies the
density field rho(omega) each step.

UNITS are SI throughout (m, s, kg).  This is safe even though the Richards
kernel scales its mass and conductivity by rho, because the field handed to
TADR, ``('velocity_couple', 0)``, is formed as -(a/rho)(grad psi - (rho/rho0)g)
at Richards.h:1562-1573 -- i.e. rho divides straight back out and the coupled
velocity is Eq. (1) exactly, in m/s.

TWO KNOWN DEPARTURES FROM THE PAPER, both scheduled for phase 2:

  G1  Variable viscosity.  Eq. (1) carries a mobility factor mu_0/mu(omega),
      Eq. (9).  Applied with SIMMONS_VISC=1 (transport_p.py): TADR evaluates
      mu/mu_0 next to rho and Richards uses K_s/mu at every quadrature point;
      the default runs at constant mu.  Because mu (like rho) is LAGGED --
      copied in Coefficients.preStep -- the factor needs no Jacobian change.
      cases.viscosity_ratio() is the reference implementation to check the
      kernel against.  The factor mu_0/mu is 0.518 at the high-density
      omega_max = 0.2535 and 0.917 at the medium-density omega_max = 0.05178,
      so it is a first-order effect on finger speed in BOTH, not just the
      high-density cases.  It always slows the brine, so adding it moves
      penetration heights DOWN.

  G3  Equation level.  The paper is at the Oberbeck-Boussinesq level: Eq. (3)
      has no rho in the accumulation.  This kernel uses m = rho*theta_w, and
      TADR uses m = theta*rho*u, i.e. a non-OB form.  Because rho varies in
      space the factor does not cancel; at omega_max it is a 23.5% offset.
      Phase 2 adds an OB_level flag to both kernels.

Until those land, penetration heights from this deck are expected to sit near
but not on the published curves.  That is a known bias, not a bug -- do not
"tune" it away.
"""

import os

import numpy as np
from proteus import *
from proteus import Comm
from proteus.default_p import *
from proteus.Profiling import logEvent
from proteus.richards import Richards

try:
    from .domain_simmons import *
    from . import cases
    from . import k0_field
except ImportError:
    from domain_simmons import *
    import cases
    import k0_field

analyticalSolution = None

#: Where the UNSAT drain sits.  Guevara Morel & Graf Fig. 1 says "seepage nodes
#: are placed at the top domain corners"; the deck's original reading opened the
#: whole top outside the source strip instead.  They are not close: at level 4
#: that is 0.639 m of vent against 0.0047 m, a factor of 136 (149 at level 4.5,
#: but only 27 at level 0 -- the corner drain SHRINKS with the mesh, because it
#: is a fixed number of ELEMENTS, so this choice is not mesh-convergent).
#:
#: Capacity, at level 4, against a peak inflow of 6.45e-5 m^3/s per 1 m of 2D
#: thickness: the whole-top drain passes ~12x the inflow at unit head gradient,
#: the two-corner drain ~8.6% of it.  A corner drain therefore CANNOT hold the
#: water table down -- the tank still fills, only slightly slower.
DRAIN = os.environ.get("SIMMONS_DRAIN", "top").strip().lower()
#: How many element faces at each end count as "the corner".
DRAIN_FACES = int(os.environ.get("SIMMONS_DRAIN_FACES", "1"))
if DRAIN not in ("top", "corners"):
    raise ValueError("SIMMONS_DRAIN must be 'top' or 'corners'; got %r" % DRAIN)

#: Flux-corrected transport on the RICHARDS side (SIMMONS_RICHARDS_FCT=1).
#:
#: This is a different limiter from SIMMONS_FCT, which is TADR's.  Richards runs
#: STABILIZATION_TYPE=2 (explicit entropy viscosity) here, and with FCT off the
#: low-order graph viscosity it adds to keep theta bounded is never antidiffused
#: back out -- so psi, and through it the Darcy velocity that drives the plume,
#: carries more numerical diffusion than the scheme needs.  Only FCT flips;
#: num_fct_iter stays 0 and every other Richards setting is unchanged.
#:
#: Default off, so every result to date reproduces.
RICHARDS_FCT = os.environ.get(
    "SIMMONS_RICHARDS_FCT", "0") not in ("0", "false", "False")

# Gravity is dimensionless in the pressure-head form: the kernel builds
# f = rho*(rho/rho_0)*k_rw*K_s*g, so g carries direction only.
dimensionless_gravity = np.array([0.0, -1.0, 0.0])


class RandomFieldCoefficients(Richards.Coefficients):
    """Richards coefficients with one hydraulic conductivity per element.

    The paper triggers fingering with a random K0-field assigned element by
    element.  The kernel already looks its soil parameters up per element --
    ``KWs.data()[elementMaterialTypes[eN]*nnz]`` at Richards.h:1548 -- so a
    per-element field needs no C++ change at all: make every element its own
    material type and give the type arrays one entry per element.  The type
    index is the element's GLOBAL number, because Richards ghost-scatters the
    per-node material type and a rank-local index does not survive that trip
    (see initializeMesh).

    The mesh does not exist when Coefficients is constructed, so the type
    arrays are placeholders here and rebuilt in initializeMesh, which is the
    first hook that sees a mesh and runs before
    BlockHeterogeneousCoefficients derives the boundary types from
    elementMaterialTypes.
    """

    def __init__(self, *args, **kwargs):
        self.k0_mean = kwargs.pop("k0_mean")
        self.k0_sigma = kwargs.pop("k0_sigma")
        self.k0_seed = kwargs.pop("k0_seed")
        self.k0_random = kwargs.pop("k0_random", True)
        self.source_inflow_only = kwargs.pop("source_inflow_only", True)
        #: filled in by initializeMesh, for the diagnostics to report
        self.k0_realisation = None
        self._source_qp = None
        Richards.Coefficients.__init__(self, *args, **kwargs)

    def preStep(self, t, firstStep=False):
        """Signorini update making the solute source an INFLOW-ONLY boundary.

        A Dirichlet head is inherently two-way: fixing psi_0 makes q.n an
        output, sign included, so the boundary supplies or absorbs whatever the
        interior demands.  The source here is a constant-head reservoir feeding
        a PVC box -- an inflow device -- and 0.0042 m of ponding head cannot
        hold it in inflow: the buoyancy head of the plume, ~eps*d, passes it at
        d = 0.088 m (SMD) and d = 0.018 m (SHD).  Measured without this, 28-44%
        of the strip vents, the convection cell short-circuits out through it,
        and the plume tears off the source (centreline u 1.000, 0.330, 0.046).

        What the source actually satisfies is a complementarity condition,

            psi_0 <= psi_s,   q.n <= 0,   (psi_0 - psi_s)(q.n) = 0

        i.e. Dirichlet where fluid enters, no-flow where it would leave.  That
        is an inequality and so cannot be written as any Dirichlet or Neumann
        condition -- it is a switch between the two, chosen by the solution.

        Solved here as a lagged active set: the set is updated once per step
        from the converged previous state and then held fixed through Newton.
        `numericalFlux.isDOFBoundary[0]` feeds BOTH the residual and the
        Jacobian (Richards.py:1724 and :1956), so freezing it keeps the two
        consistent -- which is the point.  Switching inside Newton instead, as
        the in-kernel seepage face does, leaves the Jacobian blind to the
        switch; that inconsistency is hypothesis H3 for the known STAB=2
        seepage failure, and is what this avoids.

        No kernel change and no rebuild: `isDOFBoundary` is built once in
        NF_base.__init__ and `setDirichletValues` refreshes only the values, so
        edits here persist, and the saturated `getFlowFluxBCs` returns None on
        the top, so the flux-BC reload at Richards.py:1620 never touches these
        points.
        """
        rv = Richards.Coefficients.preStep(self, t, firstStep)
        model = getattr(self, "model", None)
        if firstStep or model is None:
            return rv

        ebqe = model.ebqe
        if self._source_qp is None:
            xq = ebqe['x']
            tol = 1.0e-8
            self._source_qp = (
                (np.abs(xq[..., 1] - cases.LZ) <= tol)
                & (xq[..., 0] >= cases.SOURCE_X0 - tol)
                & (xq[..., 0] <= cases.SOURCE_X1 + tol))

        # Runs in BOTH configurations -- with the Signorini seal off this is the
        # only thing that reports the state of the source, and the seal-off run
        # is exactly the one that showed the empty middle is not the seal's doing.
        self.log_source_inflow_profile(t, model)

        if not self.source_inflow_only:
            return rv

        # ------------------------------------------------------------------
        # BC-VEL, SPLIT BY BOUNDARY.  Richards.preStep's generic [BC-VEL] probe
        # sums the penalty term -- the part of the imposed boundary flux that
        # velocity_couple structurally cannot carry -- over EVERY Dirichlet face,
        # which here is the whole top.  That total is ambiguous: under the source
        # strip TADR's concentration rows are strongly pinned (forceStrongConditions
        # + the u=1 Dirichlet), so whatever the two models disagree about there
        # lands in a row Python overwrites and is inert.  OUTSIDE the source the
        # top is a free outflow for the convection cell and TADR's rows there are
        # live, so the same defect is real.  Only the split says which we have.
        # ------------------------------------------------------------------
        try:
            _pen = ebqe['penalty']
            _psi_bc = model.numericalFlux.ebqe[('u', 0)]
            _isdb = model.numericalFlux.isDOFBoundary[0] == 1
            _gap = np.abs(_pen * (ebqe[('u', 0)] - _psi_bc))
            _flx = np.abs(ebqe[('advectiveFlux', 0)])
            _in_src = self._source_qp & _isdb
            _off_src = (~self._source_qp) & _isdb
            _c = Comm.get().comm.tompi4py() if Comm.get().size() > 1 else None
            def _s(v):
                v = float(v)
                return _c.allreduce(v) if _c is not None else v
            _gs, _fs = _s(_gap[_in_src].sum()), _s(_flx[_in_src].sum())
            _go, _fo = _s(_gap[_off_src].sum()), _s(_flx[_off_src].sum())
            logEvent("[BC-VEL-SPLIT] t=%.6e  SOURCE(pinned, inert): unseen=%.1f%% "
                     "of %.4e | TOP-OUTSIDE(live): unseen=%.1f%% of %.4e"
                     % (t, 100.0 * _gs / max(_fs, 1e-300), _fs,
                        100.0 * _go / max(_fo, 1e-300), _fo), level=2)
        except Exception as _e:
            logEvent("[BC-VEL-SPLIT] probe failed: %r" % (_e,), level=2)

        src = self._source_qp
        active = model.numericalFlux.isDOFBoundary[0]
        flux = ebqe[('advectiveFlux', 0)]      # outward normal, so > 0 is OUT
        psi = ebqe[('u', 0)]                   # interior trace of psi_0

        # TOLERANCE, not a bare sign test.  `flux > 0.0` decides a RATCHET: shut
        # is sticky, because reopening needs the free trace to fall back below
        # PSI_SOURCE_SATURATED, which a brine pond under the strip prevents.  So
        # a single wrong decision is permanent.  And the decision is made exactly
        # where the flux crosses zero, on an ebqe array whose faces are DUPLICATED
        # across ranks (an exterior face of an overlap element is seen by its
        # owner and by every rank holding it as overlap).  Those copies agree only
        # to round-off, so with a bare `> 0` two ranks can latch a shared face
        # opposite ways and stay that way -- a partition-dependent boundary
        # condition, and one that compounds because the sealed area feeds back on
        # the pressure that drives the next decision.
        #
        # The tolerance is relative to the live flux scale on the source, so it
        # scales with the problem and is not a tuned constant; it only suppresses
        # flipping on values that are numerically indistinguishable from zero.
        flux_scale = float(np.abs(flux[src]).max()) if src.any() else 0.0
        flux_scale = max(flux_scale, 1.0e-300)
        shut = src & (active == 1) & (flux > 1.0e-10 * flux_scale)
        open_ = src & (active == 0) & (psi < cases.PSI_SOURCE_SATURATED)
        active[shut] = 0
        active[open_] = 1
        # isDOFBoundary == 0 sends exteriorNumericalFlux down its `flux =
        # bc_flux` branch, so a shut point needs its prescribed flux at zero.
        ebqe[('advectiveFlux_bc', 0)][shut] = 0.0

        n_shut = int((src & (active == 0)).sum())
        n_src = int(src.sum())
        comm = Comm.get()
        if comm.size() > 1:
            # Rank 0 owns only its own slice of the source; a rank-local count
            # would misreport the state of the boundary.
            n_shut = comm.comm.tompi4py().allreduce(n_shut)
            n_src = comm.comm.tompi4py().allreduce(n_src)
        if n_src:
            logEvent("Simmons: source inflow-only -- %d/%d qp shut (%.0f%%)"
                     % (n_shut, n_src, 100.0 * n_shut / n_src), level=2)
        return rv

    def log_source_inflow_profile(self, t, model):
        """SIMMONS_SRC_DBG=1: where along the source strip does water enter?

        The empty middle of the plume is not a transport failure -- it is that
        almost no water crosses the middle of the strip.  The head BC steps from
        PSI_SOURCE_SATURATED to PSI_TOP_SATURATED across ONE element at
        x = SOURCE_X0 and SOURCE_X1, which is a mixed-Dirichlet corner: the
        normal flux there is singular, regularised only by dx, so the inflow
        piles into the two edge elements and the centre is left as the return
        limb of the convection cell.  Measured at level 4, t = 10 min, the outer
        1 cm of each edge (4% of the strip) already carries 32% of the inflow.

        This bins the boundary water flux by DISTANCE FROM THE NEAREST STRIP
        EDGE, which is the coordinate the mechanism lives in, and reports the
        flux Richards actually applies (the penalty piece, ('bflux',0)) next to
        the Darcy consistency piece that is what TADR advects with.  Because the
        concentration on the strip is strongly pinned to u = 1, this profile --
        not the concentration field -- is the thing to watch when the source BC
        is changed.

        Refinement test built in: the edge share must FALL under refinement if
        the edge peak is physical and RISE if it is the discretised singularity.
        It rises (22% at level 2, 32% at level 4).
        """
        if not SOURCE_PROFILE_DBG:
            return
        if getattr(self, "_src_dbg_t", None) == t:
            return
        self._src_dbg_t = t
        try:
            ebqe = model.ebqe
            src = self._source_qp
            dS = ebqe['dS']
            applied = ebqe[('bflux', 0)]        # what the residual integrates
            full = ebqe[('advectiveFlux', 0)]   # applied + Darcy consistency
            consist = full - applied            # = rho * (v_couple . n), TADR's
            x = ebqe['x'][..., 0]
            isdb = model.numericalFlux.isDOFBoundary[0] == 1

            owned = model._ownedExteriorFaceMask()
            keep = np.ones(dS.shape, dtype=bool)
            if owned is not None:
                keep[:] = False
                keep[owned] = True

            # distance from the nearest edge of the strip, 0 at the edge and
            # SOURCE_WIDTH/2 at the centre
            edge_dist = np.minimum(x - cases.SOURCE_X0, cases.SOURCE_X1 - x)
            bands = [(0.0, 0.01), (0.01, 0.03), (0.03, 0.08),
                     (0.08, 0.5 * cases.SOURCE_WIDTH + 1.0)]
            on_src = src & keep
            off = keep & (~src) & isdb

            # ONE collective for the whole probe; see the note on the matching
            # reduction in Richards._logBoundaryFluxSplit.  A reduction per band
            # inside a try/except can desynchronise the ranks and deadlock the
            # job if it ever throws on one of them.
            local = [float(on_src.sum()),
                     (applied[on_src] * dS[on_src]).sum(),
                     (consist[on_src] * dS[on_src]).sum(),
                     (applied[off] * dS[off]).sum(),
                     float((src & keep & isdb).sum())]
            for lo, hi in bands:
                b = on_src & (edge_dist >= lo) & (edge_dist < hi)
                local += [float(b.sum()), (applied[b] * dS[b]).sum()]
            local = np.array(local, dtype='d')
            comm = Comm.get()
            if comm.size() > 1:
                comm.comm.tompi4py().Allreduce(local.copy(), local)

            n_src, total, consist_tot, off_tot, n_open = local[:5]
            denom = total if abs(total) > 1.0e-300 else 1.0
            parts = []
            for i, (lo, hi) in enumerate(bands):
                n, q = local[5 + 2 * i], local[6 + 2 * i]
                # The qp count matters: the strip edge is not on a node at every
                # level (SOURCE_X0/dx is 27.08 at level 0 but 135.4 at level 4),
                # so an empty band is "nothing sampled here", not "no flux here",
                # and the two read identically without n.
                parts.append("%.0f-%.0fcm[n=%d]:%+.3e(%4.1f%%)"
                             % (100 * lo, min(100 * hi, 27), int(n),
                                q, 100.0 * q / denom))
            logEvent(
                "[SRC-PROFILE] t=%.6e  strip inflow by distance from edge  %s  || "
                "strip total=%+.4e  consistency(TADR sees)=%+.4e  top-outside=%+.4e  "
                "|| Dirichlet-open qp %d/%d  [m^3/s, outward normal positive]"
                % (t, "  ".join(parts), total, consist_tot, off_tot,
                   int(n_open), int(n_src)), level=2)
        except Exception as _e:
            logEvent("[SRC-PROFILE] probe failed: %r" % (_e,), level=2)

    def initializeMesh(self, mesh):
        n_local = len(mesh.elementMaterialTypes)

        if not self.k0_random:
            # Negative control: a uniform medium needs ONE material type, not
            # one per element.  Leaving the mesh's material types alone here
            # keeps this path on the stock code path, which makes it a clean
            # control for the per-element machinery as well as for the physics.
            self.k0_realisation = np.full(1, self.k0_mean, "d")
            logEvent("Simmons: uniform K0 = %.4e m/s (negative control)"
                     % self.k0_mean)
            Richards.Coefficients.initializeMesh(self, mesh)
            self._restrict_seepage_face(mesh)
            return

        k_global, idx, n_global = k0_field.global_field_for_mesh(
            mesh, self.k0_mean, self.k0_sigma, self.k0_seed,
            nx_elements, nz_elements, L, x0, per_cell=K0_PER_CELL)
        assert n_global == nElements_global_expected, (
            "k0_field: %d global elements, domain_simmons expects %d"
            % (n_global, nElements_global_expected))
        k = k_global[idx]
        self.k0_realisation = k

        # One material type per element, named by its GLOBAL (and geometric,
        # hence partition-independent) index -- NOT by the rank-local element
        # number.  The local number is not a name: Richards
        # ghost-scatters the per-node material type between ranks
        # (Richards.py:1659, so that two ranks sharing a cut edge pick the same
        # rock and fL_ij = -fL_ji still holds), and that scatter transmits the
        # INDEX.  A rank-local index arrives on the neighbour as some unrelated
        # element's index -- or, where the owner's element count is the larger,
        # as no element at all, and the kernel then reads alpha/n/thetaR/
        # thetaSR/KWs past the end of the tables.  The garbage conductivity
        # overflows pcBar^n in vgm_wetting and the run NaNs.  It only shows up
        # with many ranks: on a few fat ranks the stray index usually lands in
        # range and you get a silently wrong K0 on ghost nodes instead of a
        # crash.
        #
        # So the tables below are indexed by global element number and every
        # rank carries all n_global rows.  At level 4.5 (605k elements) that is
        # ~29 MB per rank, of which 4.8 MB (the field itself) was already being
        # paid; the four uniform columns are the price of a single index space.
        mesh.elementMaterialTypes[:] = idx.astype(
            mesh.elementMaterialTypes.dtype)

        self.Ksw_types = np.column_stack([k_global, k_global]).copy()  # (n_global, nd)
        self.vgm_n_types = np.full(n_global, cases.VG_N, "d")
        self.vgm_alpha_types = np.full(n_global, cases.VG_ALPHA, "d")
        self.thetaR_types = np.full(n_global, cases.THETA_R, "d")
        self.thetaSR_types = np.full(n_global, cases.THETA_SR, "d")
        self.nMaterialTypes = n_global

        logEvent("Simmons: K0 field, %d local of %d global elements, seed=%d "
                 "sigma=%g, one draw per %s: local min=%.4e mean=%.4e max=%.4e m/s "
                 "(material type = geometric global element index)"
                 % (n_local, n_global, self.k0_seed, self.k0_sigma,
                    "rectangle" if K0_PER_CELL else "triangle",
                    k.min(), k.mean(), k.max()))

        Richards.Coefficients.initializeMesh(self, mesh)
        self._restrict_seepage_face(mesh)

    def _restrict_seepage_face(self, mesh):
        """Keep the seepage face OFF the solute source strip.

        `getSeepageFace` is handed nothing but a face MATERIAL TAG
        (Richards.py:536), so it cannot tell the source strip from the rest of
        the top boundary -- it returns 1 for the whole of tag 3, the strip
        included.  That is not merely too generous.  `exteriorNumericalFlux2`
        opens with `if (isSeepageFace || isDOFBoundary)` (Richards.h:314) and
        the ONLY place the prescribed `bc_flux` is applied is that branch's
        `else`.  A seepage flag on the strip therefore makes the kernel
        unreachable for the infiltration condition and q(t) is DISCARDED in
        silence: the unsaturated cases would run with no source at all, and the
        symptom -- a plume that never forms -- looks like a physics result
        rather than a boundary-condition bug.

        The geometry is not known until the mesh exists, so the mask is applied
        here, by face centroid, immediately after the base class has built
        `isSeepageFace` from the tags.
        """
        if getattr(self, "isSeepageFace", None) is None:
            return
        xb = mesh.elementBoundaryBarycentersArray[mesh.exteriorElementBoundariesArray]
        tol = 1.0e-8
        on_top = np.abs(xb[:, 1] - cases.LZ) <= tol
        on_strip = (on_top
                    & (xb[:, 0] >= cases.SOURCE_X0 - tol)
                    & (xb[:, 0] <= cases.SOURCE_X1 + tol))
        n_before = int(self.isSeepageFace.sum())
        self.isSeepageFace[on_strip] = 0

        if DRAIN == "corners":
            # The paper's literal reading: "seepage nodes are placed at the top
            # domain corners".  Keep only the DRAIN_FACES element faces nearest
            # each end of the top.
            #
            # The test is GEOMETRIC, by distance from x = 0 and x = LX, and not
            # "the first and last few faces in x order".  A rank-local sort
            # cannot find the domain's ends: every rank's own slice has a
            # leftmost and a rightmost face, so an ordering test would open a
            # drain along every partition seam.  Distance from the domain edge
            # means the same thing on every rank.
            dx = cases.LX / float(nx_elements)
            edge = DRAIN_FACES * dx + tol
            keep = on_top & ((xb[:, 0] <= edge)
                             | (xb[:, 0] >= cases.LX - edge))
            self.isSeepageFace[~keep] = 0

        # Count GLOBALLY.  logEvent prints rank 0's slice only, and rank 0 owns
        # no top faces at all on 88-90 ranks, so the local number reads "0 drain
        # faces left" on exactly the rank whose output is visible -- which is
        # how a missing drain would hide.  See proteus_parallel_rank0_logevent_trap.
        n_after = int(self.isSeepageFace.sum())
        comm = Comm.get()
        if comm.size() > 1:
            from mpi4py import MPI
            c = comm.comm.tompi4py()
            n_before_g = c.allreduce(n_before, op=MPI.SUM)
            n_after_g = c.allreduce(n_after, op=MPI.SUM)
        else:
            n_before_g, n_after_g = n_before, n_after
        # The sum is over every rank's exterior faces including ghosted ones, so
        # a face on a partition seam can be counted twice and the TOTAL can read
        # one or two high (101 rather than 100 on 4 ranks at level 0).  The
        # drain count itself is what matters here and is exact whenever the
        # drain does not straddle a seam; treat the total as a diagnostic.
        logEvent("Simmons: drain = %s -- %d of %d flagged faces removed, "
                 "%d drain faces left GLOBALLY (%.4f m of vent, %.1f%% of the "
                 "top); rank 0 sees %d"
                 % (DRAIN, n_before_g - n_after_g, n_before_g, n_after_g,
                    n_after_g * cases.LX / float(nx_elements),
                    100.0 * n_after_g / float(nx_elements), n_after))
        if n_after_g == 0 and SIDE_BC != "head":
            raise RuntimeError(
                "Simmons: the UNSAT case has NO drain faces anywhere.  "
                "Infiltration would have no outlet and the tank would fill to "
                "full saturation, where beta=0 makes dm/dpsi vanish and Newton "
                "fails on a singular Jacobian.")


# Placeholder single-type arrays; initializeMesh replaces them with per-element
# arrays once the mesh exists.
_one = np.ones(1, "d")

LevelModelType = Richards.LevelModel
coefficients = RandomFieldCoefficients(
    nd,
    np.column_stack([_one * cases.K0, _one * cases.K0]),   # Ksw_types
    _one * cases.VG_N,                                     # vgm_n_types
    _one * cases.VG_ALPHA,                                 # vgm_alpha_types
    _one * cases.THETA_R,                                  # thetaR_types
    _one * cases.THETA_SR,                                 # thetaSR_types
    gravity=dimensionless_gravity,
    # Dimensionless density, rho/rho_0, matching both working decks (Elder and
    # the Liu variable-density case).  See the note in cases.Case.rho_f: the
    # solution is invariant under the rescaling, but several TADR diagnostics
    # assume rho ~ 1 and are not normalised by it.
    density=1.0,
    beta=0.0,                    # incompressible fluid and matrix (Eq. 3)
    density_model=1,             # index of TADR in pnList -> supplies rho
    diagonal_conductivity=True,
    STABILIZATION_TYPE=2,
    ENTROPY_TYPE=1,
    LUMPED_MASS_MATRIX=False,
    FCT=RICHARDS_FCT, FCT_BND_DIFF=True,  # Barua edge-flux runs: boundary routing on (default off since #173)
    
    num_fct_iter=0,
    cE=1.0,
    uL=0.0,
    uR=1.0,
    cK=1.0,
    outputQuantDOFs=False,
    k0_mean=cases.K0,
    k0_sigma=SIGMA_LNK,
    k0_seed=SEED,
    k0_random=RANDOM_K,
    source_inflow_only=SOURCE_INFLOW_ONLY and case.saturated,
)


# ---------------------------------------------------------------------------
# Initial condition: psi_0 = h_ref - z, with h_ref = 1.06 m (saturated cases)
# or 0.62 m (unsaturated cases).  Fig. 1.
# ---------------------------------------------------------------------------
class HydrostaticIC(object):
    def __init__(self, h_ref):
        self.h_ref = h_ref

    def uOfXT(self, x, t):
        return self.h_ref - x[1]

    def uOfX(self, x):
        return self.h_ref - x[1]


initialConditions = {0: HydrostaticIC(case.h_ref)}


# ---------------------------------------------------------------------------
# Boundary conditions (Fig. 1).  Left, right and bottom are impermeable in
# every case.  The top differs:
#
#   saturated   Dirichlet psi_0 = 0.0042 m under the source (which is what
#               lets brine pond) and psi_0 = 0 on the rest of the top.
#   unsaturated specified flux q(t) under the source, no flow elsewhere, plus
#               seepage nodes at the top corners.
# ---------------------------------------------------------------------------
#: Cached (t, q) record, filled on first use by _load_q_record.
_q_record = None

Q_RECORD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             Q_FILE or os.path.join("data", "q_%s.csv" % case.name))


def _load_q_record():
    """The measured infiltration record for the unsaturated cases.

    GAP G5.  The paper prescribes "a time-varying specified flux (adopted from
    Simmons et al. 2002)" and never tabulates it; Simmons et al. recorded it as
    volumetric loss from the constant-head reservoir.  Nothing here invents a
    number -- inventing one would quietly become "the benchmark" -- but the gap
    is now a DATA slot rather than a code edit: drop the digitised record in as

        data/q_<CASE>.csv     two columns, "t_seconds,q_metres_per_second"

    with q > 0 for infiltration (the sign flip to Richards' outward normal is
    applied in getFlowFluxBCs, not here).  `#` comments and a header line are
    ignored.  Values are linearly interpolated and held constant outside the
    record, so the first row should be t = 0 and the last should reach the
    case duration.
    """
    global _q_record
    if _q_record is not None:
        return _q_record
    if not os.path.exists(Q_RECORD_PATH):
        raise NotImplementedError(
            "Simmons et al. (2002) infiltration record q(t) not digitised for "
            "case %s; see README.md gap G5.  Expected a two-column CSV "
            "'t_seconds,q_m_per_s' at %s.  Only the saturated cases (SHD, SMD) "
            "run without it." % (case.name, Q_RECORD_PATH))
    rec = np.loadtxt(Q_RECORD_PATH, delimiter=",", comments="#", ndmin=2)
    if rec.ndim != 2 or rec.shape[1] != 2:
        raise ValueError("%s must have exactly two columns (t_seconds, "
                         "q_m_per_s); got shape %r" % (Q_RECORD_PATH, rec.shape))
    t_q, q = rec[:, 0], rec[:, 1]
    if not np.all(np.diff(t_q) > 0.0):
        raise ValueError("%s: the time column must be strictly increasing"
                         % Q_RECORD_PATH)
    if np.any(q < 0.0):
        raise ValueError("%s: q < 0 means the source EXTRACTS water; the "
                         "Simmons source is an infiltration device" % Q_RECORD_PATH)
    if t_q[0] > 0.0 or t_q[-1] < case.T:
        logEvent("Simmons: WARNING q(t) record spans [%.1f, %.1f] s but the "
                 "%s case runs [0, %.1f] s -- the end values are held constant "
                 "outside that span" % (t_q[0], t_q[-1], case.name, case.T))
    logEvent("Simmons: loaded q(t) from %s -- %d samples, t in [%.1f, %.1f] s, "
             "q in [%.4e, %.4e] m/s"
             % (Q_RECORD_PATH, len(t_q), t_q[0], t_q[-1], q.min(), q.max()))
    _q_record = (t_q, q)
    return _q_record


def source_flux(t):
    """Infiltration flux under the source, m/s, positive INTO the domain."""
    t_q, q = _load_q_record()
    return float(np.interp(t, t_q, q))


#: The two vertical sides, when SIMMONS_SIDE_BC=head.  Constant total head
#: H = h_ref, written as psi = h_ref - z -- literally HydrostaticIC's own
#: expression, so the sides are in equilibrium with the initial condition and
#: drive no flow until the source perturbs the interior.
SIDE_TAGS = (boundaryTags["left"], boundaryTags["right"])


def on_side(x, tag):
    return SIDE_BC == "head" and tag in SIDE_TAGS


def side_head(x, t):
    return case.h_ref - x[1]


if case.saturated:
    def getFlowDirichletBCs(x, tag):
        if on_source_strip(x, tag):
            return lambda x, t: cases.PSI_SOURCE_SATURATED
        if tag == boundaryTags["top"]:
            return lambda x, t: cases.PSI_TOP_SATURATED
        if on_side(x, tag):
            return side_head

    def getFlowFluxBCs(x, tag):
        # Impermeable sides and base.  The top is fully Dirichlet, so it must
        # NOT also carry a flux condition -- and neither must a head side.
        if on_side(x, tag):
            return None
        if tag in (boundaryTags["bottom"], boundaryTags["left"],
                   boundaryTags["right"]):
            return lambda x, t: 0.0

    getSeepageFace = None

else:
    def getFlowDirichletBCs(x, tag):
        if on_side(x, tag):
            return side_head
        return None

    def getFlowFluxBCs(x, tag):
        if on_source_strip(x, tag):
            # Richards uses the OUTWARD normal flux, so infiltration into the
            # domain is negative.
            # q(t) is a VOLUME flux of brine (Eq. 3 is a volume balance), but the
            # kernel balances m = rho*theta and integrates bc_flux as a MASS flux
            # (v.n = bc_flux/rho, Richards.h), so the brine volume q enters as
            # rho_s*q.  Without the factor only q/rho_s of the volume enters.
            return lambda x, t: -case.rho_s * source_flux(t)
        if on_side(x, tag):
            return None          # Dirichlet head there instead
        return lambda x, t: 0.0

    def getSeepageFace(flag):
        """Drain faces so water can leave if the water table reaches the top
        (Fig. 1, "UNSAT: drain nodes").

        This is handed a face MATERIAL TAG only, so it cannot address geometry:
        it flags the whole of tag 3 and `_restrict_seepage_face` then clears the
        solute source strip, leaving the top OUTSIDE the source as drain.

        OPEN (decide before running UHD/UMD): the paper puts the drain at the
        two top CORNERS ("seepage nodes are placed at the top domain corners"),
        not along the whole top outside the source.  The two differ by ~160x in
        vent area (0.638 m vs one element at each end), which sets how hard the
        water table is pinned once it arrives.  A corner-only drain is also
        exactly the 1/r Dirichlet-corner flux singularity already documented for
        the source strip, so it is not obviously the safer reading.

        G4, corrected 2026-09-22: the active set is NOT lagged.
        `exteriorNumericalFlux2` (Richards.h:325-333) switches `isDOFBoundary`
        INSIDE the flux evaluation and, because the parameter is `int&` bound to
        `numericalFlux.isDOFBoundary[0]` (Richards.py:1789), writes the decision
        back into the shared array every residual call.  The Jacobian reads that
        same array (Richards.py:2127), so it is consistent with the post-switch
        residual -- but it carries no derivative OF the switch, making this a
        semismooth Newton that can chatter at the switching boundary.  Watch the
        Newton convergence history, and see richards_stab2_seepage_failure.
        """
        return 1 if flag == boundaryTags["top"] else 0


if SIDE_BC == "head":
    # The reference deck has no seepage nodes -- the fixed-head sides ARE the
    # vent.  Keeping the drain as well would put a second, mesh-dependent
    # outlet on top of the one being tested, and the two cannot be told apart
    # in the water balance.
    getSeepageFace = None

# getSeepageFace is read by Coefficients.initializeMesh, which runs long after
# construction, so attaching it here is equivalent to passing it to __init__ --
# and lets the boundary conditions be defined below the coefficients.
coefficients.getSeepageFace = getSeepageFace

dirichletConditions = {0: getFlowDirichletBCs}
fluxBoundaryConditions = {0: "setFlow"}
advectiveFluxBoundaryConditions = {0: getFlowFluxBCs}
diffusiveFluxBoundaryConditions = {0: {0: getFlowFluxBCs}}
