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

**Boundary term:** at a boundary node `Σ_j c'_ij = −∫_∂Ω φ_i n =: −n_i`, so

    R_i^adv = Σ_j (F_j − F_i) · c'_ij − F_i · n_i .

The first part is the usual increment form with its bar states, using c' in
place of c. For the advective flux `F_adv(U)·n = (u·n) U`, so the boundary term
is `−(u_i·n_i) U_i`: it only rescales `U_i`. The low-order update stays a
positive combination of `U_i` and the bar states provided

    dt (Σ_j 2 d_ij − u_i·n_i) ≤ m_i ,

which is the extra margin added to `calculateEdgeBasedCFL` on wall rows. Velocity
is unchanged by the term, and at rest it vanishes, so well-balancing is kept. (An
earlier draft used a mirror-state wall Riemann problem; it is not needed.)

**Pressure stays on c_ij.** The hydrostatic term remains in its non-conservative,
well-balanced form `g h_i Σ_j (h_j + Z_j) c_ij` (extended source, eq. 6.13 of
the friction paper). That form supplies the wall's pressure reaction itself.
Bar states on boundary edges therefore use c' for the advective part and c for
the pressure.

**Which rows:** `wall_dof = 1` on every exterior-boundary node when
`reflecting_BCs=True`, otherwise on nodes with `nodeMaterialTypes == 99`. Other
rows, open boundaries included, keep c_ij, so transmissive outflow is unchanged.
`INTEGRATE_BY_PARTS_WALLS` (default 1) selects this; 0 restores the nodal
projection and row zeroing.

### Two conservation details found in testing

1. **μ_ij must be symmetric.** On a wall–wall edge `c'_ij ≠ −c'_ji`, so the
   well-balancing viscosity `μ_ij = max(−u_i·c'_ij, u_j·c'_ij)` computed per row
   differs from μ_ji, and `Σ_j μ_ij (U_j − U_i)` leaks mass (1e-7 in 4 s).
   On edges touching a wall row, μ is now the max over both orientations.
2. **The limiter must be symmetric.** The kinetic-energy checks in
   `convexLimiting` use row-i data only, so `l_ij ≠ l_ji` can occur and the
   limited correction `Σ_j l_ij A_ij` stops being conservative (a one-off 7.8e-7
   volume jump when the bore hit the end wall). After the limiter loop,
   `l_ij = l_ji = min(l_ij, l_ji)`. This also changes the legacy path slightly
   (dam3Bumps max |Δh| 2.8e-3, obstacle_flow 5e-4).

### Results (ceds-model, refinement 5, S = 0.03)

Closed tank `tank.py` (scratchpad), all walls:

| case | legacy walls | IBP walls |
|---|---|---|
| lake at rest, 4 s: max |η−η0|, max |hu| | 2.6e-14, 2.4e-14 | 2.4e-14, 1.3e-14 |
| wet dam break, 4 s: relative volume change | +3 to +7e-7, drifting | −6.7e-14 |
| dry-bed dam break, 0.51 m wide (7 elements): max cross-channel Δh at t = 1.0 / 2.0 / 2.2 s | 6.4e-3 / 1.8e-2 / 1.95e-2 (growing) | 9.6e-4 / 3.9e-4 / 4.8e-4 (decaying) |

- In the wide channel, the legacy walls grow a transverse depth mode while the
  front runs down the slope. With IBP walls the mesh-induced asymmetry at the
  dam decays. After the end-wall impact, IBP is 0.7–0.9e-2 vs legacy 1.3e-2 at 4 s.
- With IBP the nodal `hv` at the side walls is O(1e-3) near the front: the weak
  no-penetration condition. It is exactly antisymmetric across the channel, and
  the discrete wall mass flux is zero.
- In the original 2-element-wide Torino input, the large `hv` in **both**
  treatments sits in the downstream end-wall corners (flow along the end wall).
  It is not a transverse mode, so `max|hv|/max|hu|` is a poor metric there.
- Dry-bed runs still lose about 3e-5 (IBP) or 7e-5 (legacy) volume at the
  wet/dry front. That is separate from the walls.

`test_SWFlow.py`: dam3Bumps and obstacle_flow baselines regenerated (IBP vs old:
max |Δh| 6.7e-2 of 1.9 and 1.4e-3 of 1.0); the other 8 are unchanged.

**Still to do:** GN_SW2DCV (same changes); rerun the Torino input with a
realistic channel width.

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
