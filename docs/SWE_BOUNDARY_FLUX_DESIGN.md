# SW2DCV boundary fluxes: integrating the whole flux by parts, wall Riemann flux, tailwater

Design note for branch `swe-ibp-drag-porosity`. Status: wall case prototyped and
tested (`WALL_RIEMANN`, default off); tailwater case designed, not implemented, with
three decisions needed (section 5).

## 1. What the scheme does at a wall now

On a wall row (`wall_dof = 1`, `SW2DCV.h` "Walls:" comment) the advective terms are
integrated by parts with zero flux, `R_i = Σ_j (F_j − F_i)·c'_ij + F_i·n_i − F_b·n`, with
`c'_ij = −c_ji`, `n_i = Σ_j c_ji = ∫_∂Ω φ_i ν` and `F_b = 0`. The pressure stays in the
non-conservative, well-balanced form `φ_i (p_j − p_i) c_ij`. The wall's pressure
reaction is therefore carried by `c_ij` itself and there is no explicit boundary
pressure term.

## 2. Integrating the pressure by parts too (the proposal)

For P1 elements `c_ij + c_ji = n_ij := ∫_∂Ω φ_i φ_j ν` and `Σ_j c_ij = 0`. Hence, for
nodal pressure `p`,

    Σ_j p_j c_ij  =  −Σ_j p_j c_ji + Σ_j n_ij p_j .

The left side is what the scheme computes. The right side is the integrated-by-parts
form with the boundary integral `∮ φ_i p ν` evaluated with the interpolated pressure.
**They are identical, not just equivalent.** Replacing `Σ_j n_ij p_j` by the lumped
`p_i n_i` is not equivalent: it differs at every boundary row wherever `p` varies along
the wall, i.e. exactly on a sloped bed, and would break the lake at rest.

Checked numerically on a perturbed P1 mesh (`docs/check_ibp_identity.py`):

    max |c_ij + c_ji − n_ij|          5e-17
    max |G − I_exact|                 6e-17
    max |G − I_lumped|                1.7e-2   (|G| max 3.3e-2; 40 of 40 boundary rows)

Consequences:

- The explicit-flux form costs nothing: the boundary matrix `n_ij` is `Cx + CTx`,
  `Cy + CTy`, already available.
- The boundary flux can then be replaced by any numerical flux (section 3) without
  touching the interior, and the integrated momentum balance equals the boundary pressure
  force exactly.
- Any boundary flux must use the interpolated form `Σ_j n_ij (·)_j` for the part that
  must balance the bed source; a lumped `F_b(U_i)·n_i` is only safe for terms that vanish
  at rest (the impedance part below).

## 3. Boundary flux from a ghost state

At a boundary node add one extra neighbour with state `U_b` and coefficient `n_i`:

    F_b·ν = ½ (f(U_i) + f(U_b))·ν − ½ λ (U_b − U_i),   λ ≥ λ_max(U_i, U_b; ν)

This is an ordinary graph-viscosity edge, so the invariant-domain argument for the
low-order update holds for any `U_b`, and the existing local bounds (which include the
low-order state) contain it. High-order: the same boundary flux is used in both updates
(no antidiffusive boundary flux, so the boundary layer is first-order; see section 6).

**Wall:** `U_b` is the mirror state (`q_b = q − 2 (q·ν) ν`, same `h`). The mass flux is
exactly zero and the momentum flux is `(p + h u_n² + λ h u_n) ν`. Everything except
`dp = h u_n (u_n + λ)` is already in the scheme, so the prototype adds only
`dp · n_i` to the low-order and high-order updates and a damping-rate margin to the
edge-based CFL (`wallRiemann` in `SW2DCV.h`). It raises the wall pressure when the flow
runs into the wall and lowers it when the flow leaves (the impedance of the exact wall
Riemann problem). It vanishes at rest. Switch: `WALL_RIEMANN=1` (`wall_dof = 2`),
`torino.py` option `wall_riemann`. Corners use the nodal normal `n_i/|n_i|`, which is
only approximately right at a 90° corner (see section 6).

## 4. Wall tests (ceds-model)

**Regression.** `WALL_RIEMANN=0` reproduces the cE = 5 S3H15 run (`open_cE5/A`) bit for
bit (gauge series, max difference 0).

**Torino S3H15, closure A (refinement 5, open end).** Central gauges are unchanged
(depth at the three rows at t = 2, 3, 5, 8 s agrees to 1e-3 m; the 0.03 m maximum
difference in the series is a front-timing shift from the slightly smaller `dt`). The
cross-channel anomaly `η − ⟨η⟩_y` at the walls is already tiny with the current scheme
and is not changed: rms 0.061 / 0.053 / 0.040 mm (off) vs 0.065 / 0.052 / 0.038 mm (on) at
t = 2 / 5 / 8 s.

**Bore reflecting off a closed end** (S = 0, no canopy, no friction, uniform `h0 = 0.1 m`,
`u0 = 1 m/s`, Fr = 1.01; exact reflected-bore depth `h1 = 0.21824 m`, bore speed
0.8457 m/s; mean depth over `t_arr + 0.3 ≤ t ≤ 2.5 s`):

| refinement | wall node, off | wall node, on | 0.3 m from wall, off | on |
|---|---|---|---|---|
| 4 | −0.67 % | −1.17 % | −1.05 % | −1.04 % |
| 5 | −0.17 % | −0.19 % | −0.04 % | −0.02 % |
| 6 | −0.04 % | −0.04 % | +0.00 % | −0.00 % |

**Result.** The current wall treatment already reproduces the exact reflected bore and
converges at about second order. The Rusanov impedance term does not improve it: the
extra pressure acts only while the wall-normal velocity is non-zero, which is a few
steps, and the wall node's normal velocity is already ~1e-3 of `u0`. On the closed-end dam break
(S = 0, no canopy, depth at the end wall against the refinement-6 run with the switch off)
the switch is slightly worse, not better: rms 0.0246 vs 0.0233 m at refinement 4 and
0.0147 vs 0.0125 m at refinement 5 (the reference itself has the switch off, so this
biases toward off, but there is no sign of a gain). **The wall case is therefore a consistency result, not
an accuracy gain: the weak mirror-state wall is as good as the existing wall, and the
value of the ghost-state form is the boundaries below.** `WALL_RIEMANN` stays in as an
experimental switch, default off.

## 5. Tailwater (partially open) boundary, design

Use the same ghost-state edge with `U_b` the downstream state and no wall projection.
Properties of the Riemann solution between `U_i` (interior) and `U_b`:

- Supercritical outflow (both characteristics leave): the interface flux is `f(U_i)`,
  so the data are ignored automatically, which is the wanted behaviour with strong
  flow from upstream.
- Subcritical outflow: one characteristic enters from outside, so the downstream stage
  matters.
- A high tailwater blocks or reflects flow, a low one lets it drain.

The current open end (no boundary term) is the special case `U_b = U_i`.

Decisions needed:

1. **Ghost velocity.** (a) A pool, `u_b = 0`, `h_b = h_tw`: the physically natural
   large-basin interpretation, but it launches a wave into a flume that discharges at
   `u ≠ 0` when `h_tw` is below the interior depth. (b) A characteristic ghost,
   `h_b = h_tw`, `u_b = u_i + 2(√(g h_i) − √(g h_tw))` (outgoing invariant kept):
   transmits cleanly, only imposes the stage. (c) Both selectable. Recommendation: (b)
   as the default with (a) available.
2. **Data source.** A constant `h_tw`, or a callable `h_tw(x, y, t)` (a rating curve or
   a recorded stage). Recommendation: callable, constant as a special case.
3. **Weir.** The Torino flume discharges over a weir. A rating `q = C h^{3/2}` imposed
   on the mass flux is a separate, simpler option from the Riemann flux. Should it be
   included?

## 6. Known limitations and open items

- Corners use the nodal normal. Per-facet normals (boundary facet loop instead of the
  nodal `n_i`) are needed for exact corner treatment.
- No antidiffusive flux on the boundary edge: first-order boundary layer. A limited
  boundary antidiffusive flux is possible.
- Dry boundary nodes: the prototype skips `h ≤ hEps`.
- Porosity: `phi_i` multiplies the wall flux, as for the advective boundary term. At a
  porosity jump on the boundary this should be revisited.
- GN_SW2DCV has its own wall setup (`wallCoefficients`) and is not changed.
