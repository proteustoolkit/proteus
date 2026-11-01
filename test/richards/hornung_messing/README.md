# Hornung–Messing manufactured problem (2-D, no gravity)

The accuracy test of Ait Hammou Oulhaj, Cancès & Chainais-Hillairet (2018), in the form used by
Dielmann, Kuzmin & Kees (JNMA, in preparation), §7.1:

- Domain Ω = (0,1)², T = 0.05, no gravity (hydraulic head = pressure head p).
- Constitutive laws: s(p) = π²/2 − 2 arctan²(p) and η(p) = 2/(1+p²) for p < 0; s = π²/2 and η = 2 for p ≥ 0.
- `Richards.Coefficients(..., PSK_type='HornungMessing')` with θ_R = 0, θ_S − θ_R = π²/2, K_s = 2.
- Exact solution: p = −ξ/2 for ξ < 0 (saturated) and p = −tan((e^ξ−1)/(e^ξ+1)) for ξ ≥ 0, with ξ = x − y − t.
  It satisfies the PDE on both branches and has continuous flux across ξ = 0.
- Dirichlet data p_ex on ∂Ω, initial data p_ex(·,0).

## Files
- `hm_2d_p.py`, `hm_2d_n.py`: the decks. Context options: `STABILIZATION_TYPE`, `FCT`, `nn`
  (nodes per side), `nDT` (fixed steps), `T`, `theta_time` (1 = backward Euler, 0.5 = Crank–Nicolson,
  through `Richards.ThetaScheme`).
- `hm_l2_error.py`: L² error at t = T. P1 interpolation, Dunavant degree-8 rule on 4 subtriangles per element.
- `run_hm_convergence.sh`: h = 1/2…1/64. θ = 1 uses Δt = 0.2h²; θ = ½ uses Δt = 0.1h, as in the JNMA tables.
  Writes `convergence_theta1.0.txt` and `convergence_theta0.5.txt`. A full serial sweep takes about 3 min.

```bash
./run_hm_convergence.sh
```

## Results (structured right-triangle P1 mesh, serial LU, Newton tolerance 1e-10, strong Dirichlet)

Full tables: `convergence_theta1.0.txt` and `convergence_theta0.5.txt`. L² error at h = 1/64 and observed
order on the finest meshes:

| scheme | θ = 1 (Δt = 0.2h²) | θ = ½ (Δt = 0.1h) |
|---|---|---|
| low order (`STABILIZATION_TYPE=2 FCT=False`) | 6.99e-5, 0.99 | — (always BE) |
| Galerkin (`STABILIZATION_TYPE=0`) | 2.29e-7, ≈ 1.9–2.2 | 1.51e-7, ≈ 2.4–2.7 |
| WRR FCT (`STABILIZATION_TYPE=2 FCT=True`) | 1.55e-6, ≈ 2.6–2.7 | 2.69e-3, ≈ 0.9 |

The low order has the same non-monotone coarse-mesh regime (h = 1/2 → 1/4) as the JNMA Table 1.

### Implicit (JNMA) FCT: `STABILIZATION_TYPE=2 FCT=True fct_type='implicit'` (scheme `IFCT`)
`Coefficients(FCT_TYPE='implicit')`. Each step: Galerkin θ-scheme predictor pᴴ, raw fluxes at pᴴ, nodal-variation
prelimiter (ψ(r) = r³, `FCT_PRELIMITER_POWER`), Zalesak with global bounds [θ_R, θ_S] relative to θⁿ, then an implicit
low-order solve from the corrected datum θ^{n,FCT}. No θ → p inversion. Q1 L² errors, h = 1/2 … 1/64:

| | proteus | JNMA MFEM |
|---|---|---|
| θ = 1 | 9.79e-4, 6.66e-4, 1.98e-4, 5.23e-5, 1.42e-5, 3.43e-6 | 9.78e-4, 6.74e-4, 1.99e-4, 5.23e-5, 1.42e-5, 3.44e-6 |
| θ = ½ | 7.30e-4, 4.99e-4, 1.57e-4, 4.42e-5, 1.27e-5, 3.24e-6 | 9.86e-4, 5.66e-4, 1.71e-4, 4.63e-5, 1.31e-5, 3.28e-6 |

On P1 at h = 1/64: 7.55e-7 (θ = 1) and 4.90e-7 (θ = ½). Without the prelimiter (`fct_prelimiter=0`), Q1 at θ = 1 gives
1.82e-5, 5.20e-6 and 1.01e-6 at h = 1/16, 1/32 and 1/64. On this smooth problem the prelimiter, not the bounds,
causes most of the gap to Galerkin (2.34e-7).

On Q1 (`MESH=q1`, tables `convergence_q1_theta*.txt`) at h = 1/64, θ = 1: low order 1.15e-4, Galerkin
2.34e-7, WRR FCT 1.94e-6 (order ≈ 2.5). The JNMA MFEM values are 1.15e-4, 1.01e-6 and 3.44e-6 (implicit FCT).

### What the FCT needed
The kernel sets ṁᴸ at strong-Dirichlet DOFs to (m(gⁿ⁺¹) − m(gⁿ))/Δt. ṁᴸ enters the mass antidiffusion
M_C,ij(ṁᴸ_i − ṁᴸ_j), and at a masked row the low-order flux divergence is not a time derivative. Without this
the FCT was first order on Q1 (1.61e-4 at h = 1/64). The error sat one cell from the boundary, on the
unsaturated side of the interface, where s′(p) is small. P1 also improved slightly (1.58e-6 → 1.55e-6).

The deck also sets two `Coefficients` options explicitly. `forceStrongConditions` is off by default; `FCT_SATURATED_LOW_ORDER` has been on by default since it was added. Both leave the WRR 1-D results unchanged:
- `strong_bc` → `forceStrongConditions=True`. With weak (IIPG) Dirichlet the boundary DOFs are free, and the
  FCT correction moved them by up to 1.5e-3, which dominated the error. (`Coefficients` previously ignored
  this argument and always used weak enforcement.)
- `fct_sat_lo` → `FCT_SATURATED_LOW_ORDER=True`. In the saturated zone (p > 0) θ is pinned at θ_S. The local
  bounds let the correction pull θ below θ_S, and the θ → p recovery then returned p ≈ 0⁻ instead of the
  positive head (710 step/node cases at h = 1/16; max error 2.5e-2 vs 3.9e-5 for the low-order predictor).
  The option sets Q_i^± = 0 where the predictor is saturated, so every flux touching such a DOF is limited
  to zero. The correction stays conservative, and the DOF keeps the low-order head.

Before these two options, FCT θ = 1 gave 4.9e-5 at h = 1/64 with erratic orders (7.5e-3 at h = 1/8).

### WRR FCT with θ = ½ does not work
The correction then contains the old-time half of the high-order flux evaluated about the low-order
predictor. That correction is O(Δt), so the local low-order bounds clip it, giving first order. Unlimited,
it does not converge (L² 2e-2–9e-2 at h = 1/16–1/64). This is the time-step restriction of semi-implicit
FCT noted in the WRR text, and it contrasts with the JNMA implicit FCT, which corrects s^n and re-solves
implicitly.
