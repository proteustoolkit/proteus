# SW2DCV / GN_SW2DCV: no-flux walls by integration by parts, drag and porosity

Working notes for branch `swe-ibp-drag-porosity` (off `1.9.x`). They support
adding Proteus SWE and dispersive-SWE results to the Torino flume paper.

## 1. What the walls do now

The advective (inertial) flux enters the residual as

    R_i^adv = Σ_j (F_j − F_i) · c_ij,      c_ij = ∫ φ_i ∇φ_j   (Cx, Cy)

Because `Σ_j c_ij = ∫ φ_i ∇(Σ_j φ_j) = 0` at every node, this is the Galerkin
projection of `∇·F` with **no boundary term at all**. Summed over nodes,

    Σ_i R_i^adv = ∫_Ω ∇·F_h = ∫_∂Ω F_h · n,

so mass and momentum are conserved only if `F·n = 0` holds on the wall. The
"reflecting" condition approximates that by projecting nodal `(hu, hv)` onto
the tangent after every stage, with nodal normals averaged at corners, and
zeroing those rows (`SW2DCV.py` 1202–1230, 1551, 1684).

Baseline, Torino SWE input (`dswe/torino.py`, S = 0.03, refinement 5, 2-element-wide
channel, 4 s), on ce5a7017:

- `max|hv| / max|hu|` grows from 1e-16 to 0.11 by t ≈ 2.6 s, while the depth stays
  symmetric across the channel: a transverse momentum mode growing from round-off.
- The lumped-mass volume `Σ m_i h_i` drifts by +6.4e-4 (relative) over 4 s.

## 2. Integrating the inertial terms by parts

    ∫ φ_i ∇·F = −∫ F·∇φ_i + ∫_∂Ω φ_i F·n.

On a no-flux wall the advective flux has `F_mass·n = hu·n = 0`. The Galerkin
form is then

    R_i^adv = −Σ_j F_j · c_ji,      c_ji = ∫ φ_j ∇φ_i   (CTx, CTy)

and `Σ_i R_i^adv = −Σ_j F_j · ∫ φ_j ∇(1) = 0`, so the scheme is **exactly
conservative**, with no projection and no nodal normals.

**Interior unchanged:** `c_ij + c_ji = ∫_∂Ω φ_i φ_j n`, so `c'_ij := −c_ji` equals
`c_ij` except when i and j both lie on the boundary. The scheme changes only on
boundary edges and boundary nodes.

**Graph viscosity unchanged:** `d_ij` already takes the max over the
`(c_ij, U_i, U_j)` and `(c_ji, U_j, U_i)` directions. Since
λ_max(−n, U_R, U_L) = λ_max(n, U_L, U_R), using c' gives the same `d_ij`.

**Boundary term:** at a boundary node `Σ_j c'_ij = −∫_∂Ω φ_i n =: −n_i`. So

    R_i^adv = Σ_j (F_j − F_i) · c'_ij − F_i · n_i .

The first part is the usual increment form with its bar states, using c' in
place of c. The second is a boundary flux evaluated with the interior state.
For the invariant-domain property it is written as a wall Riemann bar state
with the mirror state `U_i^r = (h_i, u_i − 2(u_i·n̂_i) n̂_i)`, which has an
exactly zero mass flux:

    F_i · n_i  →  ½ (F(U_i) + F(U_i^r)) · n_i − d_i∂ (U_i^r − U_i),

with `d_i∂ = λ_max(n̂_i, U_i, U_i^r) |n_i|`, so the low-order update stays a convex
combination of `U_i`, the edge bar states, and the wall bar state.

**Pressure stays on c_ij.** The hydrostatic term remains in its non-conservative,
well-balanced form `g h_i Σ_j (h_j + Z_j) c_ij` (extended source, eq. 6.13 of
the friction paper). Because `Σ_j c_ij = 0` it has no wall term, and
lake-at-rest balance is untouched.

**To check:** bar states on boundary edges then use c' for the advective part
and c for the pressure. Test whether that preserves IDP and well-balancing:
- lake at rest over the sloping bed, which must stay exactly at rest;
- exact volume conservation in a closed tank;
- a 1D dam break against the exact solution;
- the Torino transverse metric, `hv` at round-off;
- `test_SWFlow.py`. Baselines will change for cases that use walls.

## 3. Drag and porosity (after section 2)

Per unit area: Darcy–Forchheimer drag `(α + β‖u‖) h u`, with `β = ½ C_D m D` and
α = 0 for the paper (Buono et al.). It is treated semi-implicitly like the
Manning term so positivity is kept.

Porosity `φ = 1 − m π D²/4`, from depth-integrating the cylinder array.
Storage and fluxes use the pore volume (φh, φhu): for the same volume of water
the free surface stands higher in the canopy, and for the same discharge the
pore velocity is faster. φ, α and β are per-node fields from the input file
(canopy region). Air entrainment would alter φ and C_D; the RANS2P runs can
supply that if it matters.

## 4. The 1.9.0 drag bug (fixed in ce5a7017)

f0f3d712 gated a Torino canopy drag on `(xi − 5.173) > 0`. `xi` is the scheme
constant 10, so the drag applied everywhere in every SW2DCV and GN_SW2DCV run.
24c3313c's regenerated SWFlow baselines encoded it. With the drag removed, all
10 tests match the earlier baselines (c75eb6c9), which are restored.
