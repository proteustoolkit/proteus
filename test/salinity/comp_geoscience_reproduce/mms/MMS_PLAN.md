# MMS verification of the coupled Richards-TADR scheme: plan

Companion to the write-up subsection "Verification of the coupled scheme with
manufactured solutions" (`mms_coupled_subsection.tex`, references in
`mms_coupled_refs.bib`). This file is the implementation and run plan; the
subsection is what goes in the paper.

## Goal and success criteria

Measure the convergence of the coupled discretization in the configuration
used in the paper, and separate flow error, transport error and coupling error.

The study is done when:

1. T0-T3 have been run on 5 meshes along refinement path (a), low-order and
   FCT, and `tab:mms_t3` is filled.
2. T3 has been run along path (b) with `dt` and `dt/4` on each mesh.
3. The coupling sweep (`tab:mms_sweep`) shows `e_c(0) = 0` to solver
   tolerance, `e_c` linear in `eps`, and `e_c` halving with `dt`.
4. Every run keeps the fluid continuity defect (`[TADR-CONT2]`) and the
   interior `[MMS-B]` mismatch at the level of the nonlinear-solver tolerance.
   No solute budget: c is pinned on all four sides, so the boundary solute
   exchange is not observable from the deck (decided 2026-11-02).
5. T1 reproduces the standalone flow run to solver tolerance, with `c = c0`
   to round-off.

## Manufactured fields (as in the subsection)

Unit square, `L = T = 1`, z up. Parameters from `cases.py` (UHD sand):
`alpha = 11.6 1/m`, `n = 2.39`, `thetaR = 0.0162`, `thetaSR = 0.3438`,
`K0 = 0.00112 m/s`.

- `psi_e = psi0 (1 + A_psi (t/T) b(x,z))`, `b = 16 x(1-x) z(1-z)` (L = 1),
  `psi0 = -1/alpha`, `A_psi = 1`. Radu & Wang (2014) / Suciu et al. (2021)
  is `psi0 = -1`, `A_psi = 1/16`.
- `c_e = cbar + A_c tanh(exp(-omega t) xi / delta)`,
  `xi = z - (z0 - U t) + a_f cos(2 pi x)`. Shunn et al. (2012) Problem 2,
  adapted. `cbar = 0.5`, `A_c = 0.3`, `z0 = 0.75`, `U T = 0.5` (`MMS_DZF`),
  `delta = 0.15`, `a_f = 0.05`, `omega = ln 2 / T`.
- `T` defaults to one transit `theta_min L / max|v|` of the T3 reference
  configuration (front, `eps = 0.235`, constant mu): 873 s. It, and hence dt,
  is the same for every test and every sweep point.
- Cross-check field: `c_b = 0.2 + 0.6 (t/T) b(x,z)` (Suciu's `c_m`, rescaled).
- Steady variant (path b): `t/T -> 1` in `psi_e`; `z_f = z0`, `omega = 0` in
  `c_e`.

Checked numerically on a 201x201x41 grid (`python -I mms_exact.py` self-test): `psi in [-0.172, -0.086] m`,
`Se in [0.35, 0.67]`, `kr in [0.0055, 0.090]`,
`min(dpsi/dz + rho/rho0) = 0.72 > 0` (so `v_z < 0` everywhere, `eps = 0.235`),
`c in [0.200, 0.797]`, `min dc/dz = 1.9e-4 > 0`.

## Configuration under test (the paper config)

| Item | Setting |
|---|---|
| Flow | Richards `STABILIZATION_TYPE=2`, FCT on, `Theta=1`, lumped storage (`ML`), consistent-mass term in FCT (`LUMPED_MASS_MATRIX=False`), weak IIPG Dirichlet |
| Transport | TADR `STABILIZATION_TYPE=5` (ImplicitEV, BE + Newton), FCT on, `forceStrongConditions=True` |
| Coupling | `RICHARDS_EDGE_VELOCITY=1` (F = A - A^T from Richards' edge velocity), carried M |
| Splitting | `Sequential_MinModelStep`, Richards then TADR, rho and mu lagged one step |
| Low-order runs | same, with FCT off in BOTH models |

Note: with Richards FCT off, `FCT_BND_DIFF` is off too, so the weak-Dirichlet
block is folded into `T` (with the `fmax` clip) instead of getting its own
graph viscosity. The low-order vs FCT comparison therefore also changes the
boundary operator. Record this with the results.

Simplifications:
- `beta = 0`, so TADR's `rho_f (1 + eps c)` and Richards' `a_s` are the same
  function.
- `physicalDiffusion = 0` (the edge-flux branch still scales it in the w-form).
- Pass 1: `D = Dm I`, `alpha_L = alpha_T = 0` (Suciu's `D = 1` case). Pass 2:
  mechanical dispersion on.

## Code facts the symbolic generator must match

- **Units:** the decks are dimensionless in density. Richards `density = 1.0`,
  `gravity = (0, -1, 0)`; TADR `rho_f = 1`, `rho_s = 1 + eps`. So
  `rho = 1 + eps c` and `f_w`, `g` carry `rho/rho0`, not kg/m^3.
- **Viscosity:** `mu/mu0 = 1 + 1.85 w - 4.1 w^2 + 44.5 w^3`, `w = omega_max c`
  (TADR.h, `q_mu`); Richards divides `K_s` by it.
- **Darcy flux:** `v = -kr (K_s/mu) (grad psi - (rho/rho0) g)`; mass flux
  `rho v`. The edge loop uses `rho_bar = (rho_i + rho_j)/2` in the jump
  (second order; kept on purpose, see the 11-02 discussion).
- **Retention:** VGM with `m = 1 - 1/n`; the code's
  `kr = sqrt(Se) (1 - (alpha psiC)^(n-1) Se)^2` is the Mualem form exactly; no
  clamp is active in our range.
- **Dispersion:** `a = theta rho D`,
  `D = Dm I + alpha_L |v_p| vhat vhat + alpha_T |v_p| (I - vhat vhat)`,
  `v_p = v/theta` (TADR.h:149-177).
- **Storage:** Richards `rho(c^n) theta^{n+1} - rho(c^{n-1}) theta^n`, so the
  target is `f_w = d/dt(rho theta) + div(rho v)` (O(dt) lag).
- **Transport source:** `g = f_s - c f_w` (cancelled form).
- **Time level:** both `preStep(t)` hooks receive `t^{n+1}`.

## Phase 1: kernel source hooks -- DONE 2026-10-29

`Coefficients(mms_source=callable(x, t))` in Richards (into `ith_flux_term`,
so FCT sees it) and TADR (cancelled form, edge-flux branch only). Verified on
UHD L0: hooks off are unchanged (psi identical, u within 1-2 ulp), the water
source shifts `sum B` by `s_w V` within 0.6%, and `preStep` gets `t^{n+1}`.

Boundary graph-viscosity export fixed 2026-11-02 (Richards.h, `fG_b` in
`edgeFluxLow`): UHD L0 post-flood `max|B_int|` 3.4e-5 -> 1.1e-7. This was
diagnostic D1; it is now a regression check (`[MMS-B]` below).

## Phase 2: generator and deck (`test/salinity/mms_coupled/`)

**Status: DONE 2026-11-02.** Each file's docstring lists its env knobs.

| File | Content |
|---|---|
| `mms_exact.py` | SymPy fields and code-matched coefficients; numpy callables `psi_e, c_e, f_w, g` on `(..., 3)` points; `T_END`, `VMAX`, `THETA_MIN`. `python -I mms_exact.py` runs the self-tests: `g = f_s - c f_w` at random points, `f_w = 0` for a hydrostatic head, the field checks above. |
| `domain_mms.py` | Unit square, union-jack mesh (`MMS_N`), shared P1 space and Lobatto quadrature, `dt` from `MMS_CO` (or `MMS_DT`), `tnList` with the two 1e-8 s startup steps, `FixedDT_controller`, IC/BC helpers. |
| `flow_p.py` / `flow_n.py` | Plain `Richards.Coefficients`, STAB=2, FCT from `MMS_FCT`, weak Dirichlet `psi_e` on all sides, `mms_source=f_w`. `MMS_FLOW_ONLY=1` with `MMS_RHO_REF` gives T0 and the T1 reference. |
| `transport_p.py` / `transport_n.py` | TADR STAB=5, strong Dirichlet `c_e`, `mms_source=g`, and the monitor: `[MMS-CFG]` once, `[MMS-ERR]` and `[MMS-B]` every step. |
| `mms_so.py` | `Sequential_MinModelStep`; run name carries the configuration. |
| `run_mms.sh` | `run_mms.sh <run dir> VAR=value ...`: one foreground run with the `tadr-check` build; detach long runs with `setsid nohup`. |
| `mms_table.py` | `ladder` / `sweep` modes: orders, `e_c`, LaTeX rows for `tab:mms_t3` / `tab:mms_sweep`. |

Two deck details that the first runs forced:
- Newton stops at `MMS_NL_RTOL * |r0| + MMS_NL_ATOL` (1e-5, 1e-12). A purely
  absolute 1e-10 cannot converge the 1e-8 s startup steps: their storage term
  `ML m / dt` is ~3e5, so round-off holds the residual at ~3e-10.
- `FixedDT_controller` resets dt to `DT` before every step. Plain `FixedStep`
  keeps the clipped 1e-8 s startup step for the whole run.

Verified (N=10 unless stated, run dirs in the 2026-11-02 session scratchpad):
- T3 front, both FCTs: 22 steps, no failures, 2 s wall. `[MMS-B]` at the
  Dirichlet-adjacent nodes 5.4e-9 against `max|ML f_w|` = 1.5e-5.
- T1: coupled psi equals the flow-only reference (`MMS_RHO_REF = 1.1175`) to
  1.1e-16 (FCT off) and 1.8e-16 (FCT on) at every step; `c = c0` exactly.
- T0 low-order, N = 10..80: max psi error 3.9e-2, 2.5e-2, 1.4e-2, 7.6e-3,
  rates 0.68, 0.80, 0.90 (rms 0.88, 0.93, 0.96). First order, so `f_w` is right.
- T2 low-order, N = 10, 20: `E2(c)` 1.66e-2 -> 1.01e-2, `E2(psi)` 1.80e-2 ->
  1.03e-2; psi errors identical to T0, as they must be at `eps = 0`.

**RESOLVED 2026-11-02: the cause was Step B's boundary graph viscosity d^G**,
removed from Richards.h (TB still separate, now in fL and fH alike; Steps A/C
kept). Flow-only MMS with the unchanged explicit FCT, E_inf(psi) at T, N =
10..80: 3.88e-2, 2.08e-2, 9.27e-3, 3.39e-3 (rates 0.90, 1.17, 1.45; low-order
7.6e-3 at N=80), FCT-CONS <= 1e-17, boundary error 1.8e-6 .. 2e-8. An
emulated implicit iterated correction gave the same final error (3.28e-3 at
N=80), so it is not needed. UHD L0 48-min regression: no failures, B_int,
continuity defect and solute mass unchanged (6e-7), c in [0, 1]. Richards
tests: test_1[FCT] and test_3[FCT] now pass; test_2_HYDRUS[FCT] needs its
stored values regenerated (2/101 nodes off 4.4e-6).

**TADR FCT max-norm stall, fixed 2026-11-02.** Phase 3 pilot (T3 coupled,
both FCTs, Co 0.5) had E_inf(c) stall at N=40 (4.9e-2, 2.2e-2, 2.0e-2; worse
than low-order 1.55e-2), growing late near the bottom-left corner. Cause:
TADR's ImplicitEV antidiffusion of dLow differenced c^n although dLow is
applied implicitly to c^{n+1}; unlimited, that correction is unstable (c to
0.93 against an exact maximum of 0.80). TADR.h now differences uLow in
edge-flux mode. After the fix: E_inf(c) 5.27e-2, 2.11e-2, 1.27e-2 (rates
1.32, 0.73), E2(c) rates 1.06, 1.05, continuity defect <= 7e-19, c in
[0.27, 0.80]. TADR tests 8/8, Richards 7/7. UHD L0: no failures, c in [0, 1],
plume depth (c > 0.1) at 48 min 0.986 -> 1.007 m.

The original finding, kept for the record: **Richards FCT did not hold the
weak Dirichlet head.** T0 with Richards FCT on, N = 10..80: max psi error 7.6e-2, 6.3e-2,
4.8e-2, 3.2e-2 (order ~0.5), 2-5x WORSE than low-order, and the error at the
boundary nodes is 2.6e-2 .. 7.3e-3 against 1e-5 .. 1e-6 low-order. It appears
in the first real step: at t = 43.6 s the boundary head is off by 1.75e-2
(20% of the psi range) while `psi_e` is constant on the boundary. On Tracy
(FCT_WEAK_DIRICHLET_AND_ORDER.md) FCT was more accurate than low-order, so
this is new. Candidates, not yet tested: the antidiffusion of the boundary
graph viscosity `-d^G dphi` at weak-Dirichlet rows (`bc_mask = 1` there, and
the penalty acts only in the Newton solve, not in the correction); the
`min_m_bc`/`max_m_bc` bounds; the m -> psi inversion at those nodes.

## Phase 3: pilot (fixes parameters before the ladder)

1. `T = theta_min L / max|v|` = 873 s is in place. Check whether
   `U T = MMS_DZF` should match the pore velocity, so the advective part of `g`
   is small, and write `T`, `U` back into the subsection.
2. Peclet: with `Dm` only, `Pe_h = |v| h/(theta Dm)`. Pick `Dm` so `Pe_h > 1`
   on the coarse meshes and say where it crosses 1. For pass 2,
   `alpha_L = 0.001 L` keeps `h/alpha_L > 1` up to N=160 (0.01 L crosses at
   N=100).
3. `N = 10, 20, 40` at `Co = 0.5`: if N=10 is not asymptotic (the front is
   1.5 cells wide), shift to `N = 20..320`.
4. Check stability: `c_e` puts dense fluid above light. If T3 errors grow in
   time at fixed N, reduce `eps` for the ladder (the sweep still goes to 0.4).
5. Check the Richards FCT never vetoes its limiter on these fields
   (`m = rho theta` is monotone in x and z).

## Phase 4: test ladder

| Test | Setup | Pass criterion |
|---|---|---|
| T0 | Richards alone, `rho = 1`, `f_w` only | Rates recorded |
| T1 | Coupled, `MMS_C=const` (`c = c0 = 0.5`), TADR source 0 | `max|c - c0|` at round-off; psi equals a standalone run with Richards `density = 1 + eps c0` and gravity times `1 + eps c0` to solver tolerance (PASSED at N=10, 2026-11-02) |
| T2 | `eps = 0`, `omega_max = 0`, both sources | Rates; flow independent of c |
| T3 | Full, `eps = 0.235` (UHD) | Rates |
| T3-bubble | T3 with `MMS_C=bubble`, FCT only, path (a) | Report the `Einf` loss at the peak |

T0-T3: low-order and FCT, path (a) on 5 meshes, pass 1 (`Dm` only) then
pass 2 (dispersion). T3 also along path (b), each mesh with `dt` and `dt/4`.

Expected: path (a) `r ~ 1` (BE plus one outer iteration, Shunn et al. 2012
sec. 5.3). Path (b) `r ~ 1` low-order, `1 < r <= 2` in E1/E2 with FCT.

## Phase 5: coupling sweep

Fixed mesh `N = 40` (repeat at 80 if time allows), FCT on, path (a) `dt` and
`dt/2`.

| Sweep | Values | Pass criterion |
|---|---|---|
| S-eps | `eps in {0, 0.1, 0.235, 0.4}`, `omega_max = 0` | `e_c = E(T3) - E(T2)`: 0 at `eps = 0` to solver tolerance (T2 is this point); linear in `eps`; halves with `dt` |
| S-mu | `omega_max in {0, 0.1, 0.25}`, `eps = 0.235` | Same scaling in the viscosity contrast |

A coupling bug (wrong or stale density or viscosity on one side) shows as an
offset at `eps = 0` or as non-linear growth in `eps`, even when the mesh rates
look right.

## Phase 6: diagnostics (internal unless they fail)

| ID | Question | Run | Signal |
|---|---|---|---|
| D1 | Boundary graph-viscosity export (fixed 11-02) still clean? | Every FCT run | `[MMS-B]` at boundary-adjacent nodes at Newton level |
| D2 | Is edge-velocity mode partition-independent? | T3 FCT, N=80, np1 vs np4; edge-flux mode as control | `E(c)` np1 vs np4 agree to solver tolerance |
| D3 | Edge velocity vs edge flux accuracy | T3, both modes, all N | Error ratio per mesh; rates |
| D4 | Is the steady FCT state dt-dependent? | Path (b), `dt` vs `dt/4` | Steady `E(c)` difference |

## Phase 7: write-up

`mms_table.py` -> `tab:mms_t3`, `tab:mms_sweep`, plot. Replace the expected
rates with the observed ones; write the pilot values of `T`, `U`, `Dm`,
`alpha_L` into the subsection; add one sentence on the T3-bubble result.

## Risks

- Weak IIPG penalty `C/h^p`: record `C` and `p`; the Tracy deck notes the
  default `C = 2` sits at the stability threshold for `a <= K_s`.
- Richards FCT bounds near Dirichlet nodes (`min_m_bc`/`max_m_bc`).
- Unstable stratification (Phase 3 item 4).
- Sources and BCs must both be at `t^{n+1}` (verified for the sources).

## Rough effort

| Phase | Time |
|---|---|
| 2 | 1 day |
| 3 | 0.5 day |
| 4-6 | serial runs, minutes each up to N=160; ladder + sweep + diagnostics in about a day of wall time |
| 7 | 0.5 day |
