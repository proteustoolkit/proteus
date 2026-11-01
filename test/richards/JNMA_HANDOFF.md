# Handoff: Richards FCT in proteus (JNMA implicit FCT and WRR semi-implicit FCT)

Written 20 Oct 2026 at the end of a Claude Code session with C. Kees; updated 21 Oct and 27 Oct 2026 (§11, §12). The session worked in a local
folder `~/Richards_JNMA`, which holds the JNMA manuscript (not in this repo) with this repo cloned
inside it. Everything needed to continue is in this repo on branch **`jnma-fct-1.9.x`** of
`https://github.com/cekees/proteus` (head `82806fe9` or later). This file is self-contained.
The branch is cut from `cekees/1.9.x` (bda13572). Pull requests: cekees/proteus#123 (into `1.9.x`) and
cekees/proteus#124 (cherry-pick onto `main`, branch `jnma-fct-main`); see §12.

## Priorities for the next session
1. ~~Fix the accuracy loss of the WRR semi-implicit FCT on Q1 meshes~~ **Done** (§4).
2. ~~Implement the fully implicit JNMA FCT in proteus~~ **Done** (§5).
3. ~~Match the co-authors' 1-D refinement studies and their solver settings~~ **Done** (§11).
4. Open (§6, §11.6): the cheaper first Galerkin step in proteus versus MFEM (initial guess?), the Δt
   window of Galerkin + Newton, the WRR text-vs-code mismatch, bio2d reruns, and the manuscript changes
   that follow from §11.

## 1. Background
- **WRR paper** (Barua, Kees & Kuzmin, *Bound-preserving FCT methods for Richards' equation*, WRR, under
  review; arXiv:2604.14107). Its *semi-implicit* FCT is implemented here as `STABILIZATION_TYPE=2 FCT=True`.
  Input decks are in `test/richards/WRR_input_files/`.
- **JNMA manuscript** (Dielmann, Kuzmin & Kees, *An implicit flux-corrected transport method for Richards'
  equation*, in preparation; results computed with MFEM). Its *implicit* FCT is specified in §5 below.
  The goal is to reproduce the JNMA results with proteus, and then to implement the JNMA method here.

## 2. Environment and build (macOS arm64, as used)
```bash
mamba env create -n proteus-jnma-190 -f environment-openmpi-dev.yml
```
```bash
conda activate proteus-jnma-190 && pip install --no-build-isolation --no-cache-dir -v -e .
```
After any C++ change in `proteus/richards/Richards.h` or `proteus/pskRelations.h`, rerun the `pip install`
line; Python-only changes need no rebuild. Regression check (about 20 s, must stay 7 passed):
```bash
cd test/richards && python -m pytest -q test_richards.py
```

## 3. What this branch adds (on top of `cekees/1.9.x`)
| commit | content |
|---|---|
| 563c8a4b | `Richards.ThetaScheme` θ option (`timeIntegrationTheta` in the numerics file). Galerkin: full θ-method, Jacobian checked against finite differences. Edge-based schemes: low order stays BE, θ only on the FCT high-order flux (`Theta_h`). `PSK_type='HornungMessing'` (4) with inverses. Fix: NaN residual in saturated regions without gravity (metric τ = 1/√0). Newton line search now halves on a non-finite residual. New example `test/richards/hornung_messing/`. |
| 4915ad6e | `Coefficients(FCT_SATURATED_LOW_ORDER=...)`: Zalesak Q_i^± = 0 at DOFs whose low-order predictor is saturated (not for Gardner). `forceStrongConditions` is now honoured (it was silently forced False). With strong BCs, `u_dof_old` at Dirichlet DOFs is set to g(tⁿ) (was g(tⁿ⁺¹), which corrupted the (1−θ) terms). |
| 5b9f4f23 | `FCT_SATURATED_LOW_ORDER=True` is the default. |
| 78b151c2 | `hornung_messing`: `quad=True` (Q1, 2×2 Gauss, the JNMA element), Q1 L² error integration, `MESH=q1` in the driver. |
| f4d847ad, b3513fd9 | This handoff document. |
| c777e21f | WRR FCT: `mDotLow` at strong-Dirichlet DOFs from the imposed data. Q1 Hornung–Messing becomes about second order (§4). |
| 3dad551c | Fully implicit JNMA FCT, `FCT_TYPE='implicit'` (C++ `ImplicitFCTStep`, §5). |
| 0720e829 | Both FCTs refuse parallel layouts in which owned DOFs miss part of their element star (`checkFCTParallelLayout`, §9). |
| 4fa8f106 | `LSCHEME_L`: L-scheme Jacobian option; deck options for the implicit-FCT tests; `galerkin = False` in WRR test_1. |
| e353c693 | `FCT_PREDICTOR_FALLBACK` (default True): low-order predictor when Newton fails on the Galerkin predictor; JNMA L-scheme defect-correction matrix. |
| 6abc1f1e | `NONLINEAR_SOLVER` = `'newton'` / `'lscheme'` / `'hybrid'` and the solver study (§10). |
| 43af6813 | Newton `convergenceTest='du'` / `'r_and_du'` (relative increment), `unorm='L2'` / `'L2_lumped'` (finite element norms, `Richards.femL2Norm`); numerics options `nonlinearSolverUNorm`, `nl_rtol_du`, `nl_atol_du` (§11.2). |
| 69e41a37 | `FCT_PRELIMITER_VARIABLE` = `'pressure'` / `'hydraulic_head'` (default `'pressure'`, as in MFEM, §11.3). |
| 82806fe9 | Fix: `LinearAlgebraTools.wl2Norm` reduced twice in parallel (off by √nranks; §11.5). |

Example (Hornung–Messing, unit square, no gravity, T = 0.05, exact solution; see its README):
```bash
cd test/richards/hornung_messing && MESH=q1 ./run_hm_convergence.sh
```
This takes about 3 min serial and writes `convergence_q1_theta1.0.txt` / `convergence_q1_theta0.5.txt`. Without
`MESH=q1` it runs P1 triangles and writes `convergence_theta*.txt`.
Context options: `STABILIZATION_TYPE` (0 Galerkin, 2 edge-based low order), `FCT`, `nn`, `nDT`,
`theta_time`, `quad`, `strong_bc`, `fct_sat_lo`.

## 4. Priority 1 (RESOLVED): WRR FCT was only first order on Q1
**Fix:** `calculateResidual_entropy_viscosity` now sets `mDotLow` at strong-Dirichlet DOFs (bc_mask = 0) to
(m(gⁿ⁺¹) − m(gⁿ))/Δt instead of the masked row's flux divergence (candidate 1 below). Q1 FCT at θ = 1, h = 1/2 … 1/64:
1.79e-4, 3.11e-4, 2.02e-4, 6.83e-5, 1.11e-5, **1.94e-6** (orders 2.6, 2.5), below the JNMA implicit FCT (3.44e-6).
P1 is slightly better at every level (1.575e-6 → 1.546e-6 at h = 1/64). θ = ½ is unchanged (still first order).
`test_richards.py`: 7 passed. A fully unlimited correction (no bounds, no saturated rule) is unusable on
both meshes (6e-2 Q1, 4e-2 P1 at h = 1/32), so the limiter and the saturated-DOF rule are essential. The
notes below are the original diagnosis.

L² errors at h = 1/64, θ = 1, Δt = 0.2h² (the JNMA convergence setup):

| | low order | Galerkin | WRR FCT |
|---|---|---|---|
| proteus Q1 | 1.154e-4 | 2.340e-7 | **1.613e-4 (order ≈ 1.1)** |
| proteus P1 | 6.989e-5 | 2.292e-7 | 1.575e-6 (order ≈ 2.6) |
| JNMA MFEM (Q1, implicit FCT) | 1.15e-4 | 1.01e-6 | 3.44e-6 |

What is known:
- The Q1 low order reproduces the JNMA MFEM low-order column to every printed digit at all six levels
  (1.78e-4 … 1.15e-4). Problem, element, Δt and low-order operator are therefore identical.
- At h = 1/32 the error does not come from the limiter: unlimited (bounds widened to ±1e10) gives the same
  3.5e-4. It is not on the boundary (Dirichlet DOFs exact to 1e-14 with `strong_bc`), and it is not
  in the saturated zone (max 4e-4 there).
- It is concentrated **one cell from the boundary, on the unsaturated side of the interface ξ = x−y−t ∈
  [0, 0.05)**. The worst node, at (0.094, 0.031), has p_ex = −0.0062 and p_h = −0.0136. There s′(p) ≈ 0.025,
  so a mass error is amplified about 40× by the θ → p recovery.
- On P1 the same scheme is second order, so the difference is Q1-specific. Q1 couples diagonal neighbours
  in both M_C and the stiffness. Candidates to test:
  1. The mass-antidiffusion term M_C,ij(ṁᴸ_i − ṁᴸ_j) in `FCTStep` (`Richards.h`), with ṁᴸ = low-order flux
     divergence / m_i. At DOFs next to strong-Dirichlet nodes ṁᴸ of the Dirichlet neighbour is not a time
     derivative (its row is masked). Try ṁ_j = (m_j(u_j) − m_j(u_jⁿ))/Δt there, or exclude edges to masked
     DOFs.
  2. Near-saturation conditioning: extend the saturated-node rule to DOFs with small s′(pᴸ), or limit in p
     instead of θ there.
  3. Whether the correction should use θᴴ from a Galerkin predictor (as the WRR *text* says) instead of
     the low-order state (what the code does; see §6).
- Diagnostic scripts used: monkeypatch `Richards.LevelModel.FCTStep` to record `u[0].dof` before and after,
  then bin |p_h − p_ex| by distance to the boundary and by ξ. They are easy to recreate from
  `hm_l2_error.py` (`p_exact`, `integrate`).
- Done when: WRR FCT on Q1 converges at about second order at θ = 1; `test_richards.py` still passes;
  P1 results are unchanged or better.

## 5. Priority 2 (DONE): the fully implicit JNMA FCT
**Implemented** as `Coefficients(FCT=True, STABILIZATION_TYPE=2, FCT_TYPE='implicit', FCT_PRELIMITER_POWER=3.0)`.
Hornung–Messing context: `fct_type='implicit'`; sweep scheme `IFCT`.
- `LevelModel.fct_phase` ∈ {predictor, flux, final} selects the kernel through `kernelStabilizationType()`:
  Galerkin (0) for the predictor, the edge-based low order (2) to evaluate the fluxes at pᴴ, and Implicit_FCT (3)
  for the final solve. Type 3 is the low order plus `fluxCorrection` = −(m_i/Δt)(s^{n,FCT}_i − sⁿ_i).
- `Newton.solve` calls `FCTStep` → `ImplicitFCTStep` after the predictor converges. That evaluates the fluxes,
  runs the C++ `ImplicitFCTStep` (steps 2–5; three passes with ghost scatters in parallel), and sets
  `fct_resolve`. Newton then re-solves from pᴴ. `Coefficients.preStep` rearms the predictor each step.
- Q1 errors match the JNMA tables to about 1% at θ = 1 (all levels) and within 5% at θ = ½ for h ≤ 1/16 (see the
  hornung_messing README and `convergence_q1_theta*.txt`).
- Finding: without the prelimiter, Q1 θ = 1 at h = 1/64 gives 1.01e-6 instead of 3.43e-6. The FCT error of the
  JNMA tables is therefore set mostly by the prelimiter, not by the Galerkin target (§7).
- Tested since (21 Oct 2026): parallel, β ≠ 0, and 1-D infiltration with gravity. See §9.

### Original specification
Notation: p pressure head, u = p + z hydraulic head, s(p) water content, m_ij consistent mass, m_i lumped
mass, a_ij Galerkin stiffness (∫ η(p_h)∇φ_j·∇φ_i), q_ij = (∇φ_j,∇φ_i), and the low-order upwind operator
ã_ij = η_ij q_ij, with η_ij = η(p_i) if q_ij(u_i−u_j) < 0, η(p_j) if > 0, and the average if equal. The diagonal
is set so rows sum to zero. This low order **is** proteus `STABILIZATION_TYPE=2 FCT=False` (verified above). The
tie rule and M-matrix clipping `fmax(0,-T_ij)` differ only formally.

Per time step:
1. **High-order predictor pᴴ**: solve the Galerkin θ-scheme
   M(s(p)−s(pⁿ))/Δt + θA(p)u(p) + (1−θ)A(pⁿ)u(pⁿ) = 0.
   JNMA applies M to the nodal interpolant of s; proteus Galerkin evaluates s at quadrature points.
2. **Raw fluxes** at pᴴ (θ-method high order, backward-Euler low order):
   f_ij = m_ij(ṡᴴ_i − ṡᴴ_j) − ãᴴ_ij(uᴴ_i−uᴴ_j) + θ aᴴ_ij(uᴴ_i−uᴴ_j) + (1−θ) aⁿ_ij(uⁿ_i−uⁿ_j),
   with ṡᴴ_i = (s(pᴴ_i) − s(pⁿ_i))/Δt.
3. **Nodal-variation prelimiter**: α_i^nv = 1 − ψ(|Σ_j (uᴴ_i−uᴴ_j)| / Σ_j |uᴴ_i−uᴴ_j|) with ψ(r) = r³
   (α_i^nv = 1 when the denominator is 0), over j ∈ N_i* = {j ≠ i : m_ij > 0}. Then
   f̄_ij = min(α_i^nv, α_j^nv) f_ij.
4. **Zalesak with GLOBAL bounds** [s_min, s_max] = [θ_R, θ_S] (Hornung–Messing: [0, π²/2]), relative to sⁿ:
   P_i^± = Σ max/min(0, f̄_ij), Q_i^± = (m_i/Δt)(s_max/min − sⁿ_i), R_i^± = min(1, Q_i^±/P_i^±) (1 if P = 0),
   α_ij = min(R_i^+, R_j^−) if f̄_ij > 0, min(R_i^−, R_j^+) if f̄_ij < 0.
5. **Corrected datum**: s_i^{n,FCT} = sⁿ_i + (Δt/m_i) Σ_j α_ij f̄_ij.
6. **Final implicit low-order solve** for pⁿ⁺¹: M_L(s(p) − s^{n,FCT})/Δt + Ã(p)u(p) = 0. Fluxes and factors stay
   frozen at pᴴ; there is no θ → p inversion.

JNMA solves all nonlinear systems with an L-scheme, J = (L/Δt)M_L + Ã(pᵏ) with L ≥ sup|s′|. Newton is fine
for a first implementation.

Mapping to proteus:
- Step 6 is the existing low-order residual with `mn` (= sⁿ) replaced by s^{n,FCT}. The kernel already adds
  `fluxCorrection` to the residual under `STABILIZATION_TYPE=Implicit_FCT` (=3); see the end of
  `calculateResidual_entropy_viscosity`. That hook, or simply overwriting `mn`, gives the final solve.
- Step 1 needs a Galerkin solve in the same time step: either a second model object, or a two-pass
  `NonlinearSolvers` driver that toggles the residual between `calculateResidual` and
  `calculateResidual_entropy_viscosity`.
- Steps 2–5 are a new limiter (global bounds relative to sⁿ, prelimiter, fluxes at pᴴ) next to `FCTStep`.
  The existing `FCTStep` uses local bounds relative to mᴸ, has no prelimiter, and uses fluxes at pᴸ.
- Validation: the JNMA Tables 1–2 (§7 below) and `MESH=q1 ./run_hm_convergence.sh`. The low-order column
  already matches.

## 6. Other open items
- **WRR text vs. code.** The WRR text writes the raw flux with θᴴ, ψᴴ (a high-order solution). The code
  (`calculateResidual_entropy_viscosity` → `dt_times_fH_minus_fL`, and `FCTStep`) builds it entirely at the
  low-order state ψᴸ, uses ṁᴸ in the mass term, and uses local bounds from mᴸ. One of them should be
  corrected before the WRR revision.
- **WRR FCT with θ = ½** is not a usable scheme: first order when limited (4.9e-3 at h = 1/64 on Q1) and
  divergent when unlimited.
- **Galerkin + Newton robustness** on sharp 1-D infiltration fronts (Celia, Szymkiewicz): it works only in a
  Δt window. If Δt is too large it collapses; if too small it stalls on the first step from the initial
  jump. The co-authors report the same window for their L-scheme, and chose Δt by it (§11.1). With
  `nl_solver='lscheme'` and the JNMA settings, proteus completes every level without fallbacks (§11.4).
- **`WRR_input_files/bio2d`** passes `forceStrongConditions=True`, which the code now honours. Its published
  results were computed with weak BCs, so a rerun may differ. Ponded 2-D runs may also differ under the
  new default `FCT_SATURATED_LOW_ORDER=True`.
- The digitized Celia data (`data_celia*.csv`) that `WRR_input_files/test_1/plot_test_1.py` reads is not
  committed. (`galerkin = False` was added to the deck in 4fa8f106.)
- The VG–Mualem law hard-codes pore connectivity ℓ = ½. (The co-authors have not stated ℓ.)

## 7. Highlighted finding: MFEM Galerkin accuracy
With identical inputs (Q1, same h and Δt), **proteus Galerkin errors are 2.4–6× smaller than the JNMA MFEM
Galerkin errors on the finer meshes (h ≤ 1/16), for both θ = 1 and θ = ½**. On the coarsest meshes MFEM is more
accurate (h = 1/2 at θ = 1: 4.25e-4 vs 1.00e-3; h ≤ 1/4 at θ = ½). Full tables: `hornung_messing/convergence_q1_theta*.txt`.

| h | MFEM θ=1 | proteus θ=1 | MFEM θ=½ | proteus θ=½ |
|---|---|---|---|---|
| 1/16 | 1.84e-5 | 4.15e-6 | 1.33e-5 | 5.52e-6 |
| 1/32 | 5.24e-6 | 8.64e-7 | 4.15e-6 | 1.07e-6 |
| 1/64 | 1.01e-6 | 2.34e-7 | 7.54e-7 | 2.32e-7 |

Both converge at about second order. A known difference is the storage term (MFEM: consistent mass on the
nodal interpolant of s; proteus: s(p_h) at the quadrature points). Testing it is cheap: add an option to
proteus Galerkin that uses Σ_j m_ij s(p_j) and rerun the Q1 sweep. If it explains the factor, the JNMA
draft should say so, because the JNMA FCT errors (3.44e-6 at h = 1/64) are bounded by this Galerkin target.

Also for the JNMA authors: their Celia figure (labelled 24 h, 1 m column) is not reproducible from the stated
inputs. The same low order fits best at about 1.5 h in a 0.2 m column, and the front shape still differs. The
proteus 24 h result agrees with an independent Picard solver (front 0.565 m below the top). Their
Szymkiewicz figures are consistent.

## 8. Reference values from the JNMA draft (MFEM, Hornung–Messing)
θ = 1 (Δt = 0.2h²), h = 1/2 … 1/64:
- LO: 1.78e-4, 1.10e-3, 7.87e-4, 4.34e-4, 2.27e-4, 1.15e-4
- Galerkin: 4.25e-4, 2.18e-4, 7.98e-5, 1.84e-5, 5.24e-6, 1.01e-6
- FCT: 9.78e-4, 6.74e-4, 1.99e-4, 5.23e-5, 1.42e-5, 3.44e-6

θ = ½ (Δt = 0.1h):
- Galerkin: 5.05e-4, 1.22e-4, 5.72e-5, 1.33e-5, 4.15e-6, 7.54e-7
- FCT: 9.86e-4, 5.66e-4, 1.71e-4, 4.63e-5, 1.31e-5, 3.28e-6

## 9. Testing of the implicit FCT (21 Oct 2026)
**Parallel.** Hornung–Messing (`par=True`, `par_part`, `par_overlap` context options) on 2 and 4 ranks matches serial to every
printed digit for Galerkin, the low order, the WRR FCT and the implicit FCT, **provided owned DOFs see their full element
star**: node partitioning (any overlap) or element partitioning with `nLayersOfOverlapForParallel >= 1`. With element
partitioning and overlap 0, rows of ML, MC and the flux matrices are partial at nodes on the cut. That is harmless for the
additive Galerkin/low-order residuals, but both limiters are nonlinear in those rows. Example: WRR FCT on 2 ranks gives 2.8e-3
instead of 6.8e-5 (h = 1/16); implicit FCT gives 5.50e-5 instead of 5.23e-5.
- **This affects the WRR `Tracy_convergence` runs**, which use element partitioning with overlap 0. Tracy ref_2
  (nnx = 41, WRR FCT, 4 ranks): max |ψ − ψ_serial| = 0.40 m, rms 0.018 m. With node partitioning it is 5.5e-14. The parallel
  WRR FCT results should be rerun with `parallelPartitioningType = MeshParallelPartitioningTypes.node`.
- `LevelModel.checkFCTParallelLayout()` now raises a ValueError for such a layout when FCT is on.

**β ≠ 0.** m = ρe^{βψ}θ(ψ) is increasing with range (0, ∞), so any non-negative corrected datum is admissible for the
final solve. The global bounds ρ[θ_R, θ_S] only forbid moving further outside where sⁿ already lies outside them, which is
continuous as β → 0. Hornung–Messing with β = 0.1 (Q1, θ = 1), nodal rms difference from Galerkin at h = 1/128:
implicit FCT 4.3e-5, 1.1e-5, 2.9e-6 at h = 1/16, 1/32, 1/64 (order ≈ 2); WRR FCT 1.8e-5, 3.3e-6, 7.7e-7; Galerkin 3.2e-6,
8.2e-7, 1.7e-7. β = 1e-10 reproduces β = 0 exactly. No change was needed.

**1-D with gravity** (`WRR_input_files/test_1` and `test_3` now take `fct_type`, `nn`, `nDT`, `lscheme_L`; test_1 has
`galerkin = False`). Untracked study scripts: `WRR_input_files/exp_ifct/`.
- Celia (weak BCs, Δt ∝ h). L2 head difference from the WRR FCT at h = 1/640:
  WRR FCT 0.90, 0.27, 0.10, 0.022; implicit FCT 1.10, 0.47, 0.16, 0.039 at h = 1/40, 1/80, 1/160, 1/320. Both converge to the
  same solution (they differ by 5e-3 at h = 1/640). The WRR FCT, with local bounds, is somewhat more accurate on this front.
  The implicit FCT, with global bounds only, undershoots the initial/boundary head slightly: ψ_min = −10.018, −10.009, −10.004,
  −10.002, −10.0007 (no local maximum principle in ψ).
- Szymkiewicz (dry sand, Brooks–Corey): **the implicit FCT fails**. The Galerkin predictor's Newton does not converge from the
  dry initial state; runs abort after 21–34 steps at nn = 11, 41, 81, 161 (only nn = 21 completes). The WRR FCT and the low
  order run without failures. Celia at h = 1/320 also needed one first-step retry.

**Robust predictor (added 21 Oct 2026).**
- `Coefficients(FCT_PREDICTOR_FALLBACK=True)` (the default). When Newton fails on the Galerkin predictor, `Newton.solve` restores
  the step's initial guess and calls `LevelModel.fctPredictorFallback()`. That switches the step to a low-order predictor
  (`fct_phase='low_predictor'`), so the fluxes are evaluated at pᴸ as in the WRR scheme. The limiter and the final implicit
  solve are unchanged. Szymkiewicz now completes on every grid with 1, 0, 2 and 3 fallback steps at nn = 11, 21, 41, 81. L2
  head differences from the nn = 161 WRR FCT: 2.34, 2.03, 1.13, 0.78 (WRR FCT 2.29, 1.79, 1.08, 0.64). The global bounds let ψ
  undershoot the initial −7.5 m (ψ_min −8.0, −7.61, −7.77, −7.68); θ stays within [θ_R, θ_S]. Hornung–Messing (no fallbacks
  triggered) and `test_richards.py` are unchanged.
- **What JNMA actually does** (manuscript §solver): the predictor *is* the Galerkin system, but every nonlinear system is
  solved by defect correction p^{k+1} = p^k − J⁻¹R_*(p^k) with the *low-order* matrix J = (L/Δt)M_L + Ã(p^k), L ≥ sup|s′|.
  `Coefficients(LSCHEME_L=L)` now implements exactly this for Galerkin solves (STABILIZATION_TYPE=0 and the implicit-FCT
  predictor). The edge-based kernel assembles J with dm/dψ → L and without dk_r terms; the Galerkin residual is unchanged.
  - Hornung–Messing h = 1/16: same solution as Newton (4.150e-6) in ~500 iterations per step, vs ~3 for Newton.
  - Szymkiewicz (L = 3.5): the residual decreases monotonically but at ≈ 0.9996 per iteration (134 → 9.2 in 1000), so with
    `nl_atol_res = 1e-8` it would need ~5·10⁴ iterations per step. Not practical without acceleration (Anderson, as the
    manuscript suggests), a looser or relative tolerance, or an L/Newton switch (MitraPop2019).
  - The final low-order FCT solve still uses Newton (exact Jacobian, 2–3 iterations).
- An earlier variant (consistent-mass Galerkin L-scheme matrix) was replaced by this one.

**Open:** Anderson acceleration or an L→Newton switch for the defect correction; whether JNMA's MFEM runs used a relative
tolerance or larger Δt (it would explain their convergence on Szymkiewicz with the plain L-scheme).

## 10. Nonlinear-solver study (21 Oct 2026)
Full write-up: `~/Richards_JNMA/extended_notes.tex` (extended notes in §Nonlinear solver, Table "proteus nonlinear-solver
study"), scripts and results in `~/Richards_JNMA/numerical_tests/solver_study/`.
- `Coefficients(NONLINEAR_SOLVER='newton'|'lscheme'|'hybrid', LSCHEME_L, LSCHEME_RETRY_ITS, LSCHEME_TRIGGER, LSCHEME_THETA_MAX)`.
  'lscheme' = JNMA defect correction for every solve. 'hybrid' = Newton in which a rejected step (no residual decrease, or
  Deuflhard's Θ_k = ‖J_k⁻¹r(u_{k+1})‖/‖δu_k‖ > Θ_max) is undone and replaced by k L-scheme iterations (no line search).
  Implemented in `Newton.lschemeRetry` (NonlinearSolvers.py) with `LevelModel.setLScheme/lschemeOn/nonlinearHybridOptions`.
- 299 runs (3 problems: WRR Szymkiewicz 5 m, Celia 1 m, JNMA Szymkiewicz 0.2 m):
  - Newton never fails on low-order-type solves (1.4·10⁵).
  - All failures are Galerkin-type solves near start-up.
  - Θ-hybrid with k = 2, Θ_max = ½: 21/26 Galerkin-type cases complete (Newton+line search: 9/26).
  - Θ_max = ¼: 17/26.
  - The hybrid with k = 1 broke 5 robust Celia runs.
  - Best: Newton + low-order predictor fallback (the default). It completes all 13 implicit-FCT cases with ≤1% fallback
    steps at 4–6 linear solves per step.
- Kantorovich: h₀ ≤ ½ holds for only ~60% of converging solves (useless as a test). Θ₀ > 0.25 flags all 17 failing
  Galerkin-type solves and only 0.24% of converging ones. Low-order/final-FCT solves have ~10× smaller Θ₀ than Galerkin solves.
- Szymkiewicz overlays: `~/Richards_JNMA/numerical_tests/jnma_figures/` (I1/I05 implicit FCT, H1/H05 hybrid Galerkin).
  The proteus implicit FCT matches the JNMA FCT front closely. proteus Galerkin undershoots to −8.0 m, while MFEM stays
  at −7.5 m (open question).
- Author query 5 (`~/Richards_JNMA/AUTHOR_QUERIES.md`) asks for the L-scheme tolerances, L, Δt, iteration counts and acceleration.
- Next ideas:
  - Θ₀-triggered remedy (fallback or warm-up) instead of waiting for Newton to fail.
  - FAS / nested iteration (the initial guess from the coarse level keeps Θ₀ small).
  - Anderson acceleration of the L-scheme.

## 11. Co-author answers and parity runs (27 Oct 2026)
Sources outside this repo (in `~/Richards_JNMA`): `notes on convergence and mesh sizes` and `Numerical Results/`
(V2 figures) from L. Dielmann, his email of 27 Oct, `COAUTHOR_NOTES_REVIEW.md` (full review), and
`numerical_tests/jnma_1d/` (decks, drivers, results). The 1-D parity decks are not in this repo.

### 11.1 What the co-authors confirmed
- Structure: Galerkin predictor ("ho") → limiter → final upwind solve from s^FCT ("lo"); fluxes frozen at pᴴ.
- **Detector on the pressure head pᴴ**, not uᴴ = pᴴ + z. An earlier MFEM version used pⁿ by mistake (since fixed;
  the V2 plots are post-fix).
- **Pure L-scheme** everywhere; no Anderson acceleration, no switch to Newton.
  - Stopping criterion: ‖p^{k+1} − p^k‖_{L²} < 10⁻¹² ‖p^{k+1}‖_{L²} (FE L² norm), max 5·10⁴ iterations.
  - L = ½ sup|s′| over the attained pressure range for the infiltration tests (Celia 0.1132/2, Szymkiewicz
    0.14207/2) and L = sup|s′| = 1.6478 for Hornung–Messing. The draft's "L ≥ L_s" does not hold for the 1-D runs.
- Strong Dirichlet. p^{n,0} = pⁿ with boundary values g(tⁿ⁺¹). The limiter treats boundary nodes like interior
  ones (proteus `ImplicitFCTStep` already does). Hornung–Messing bounds: [0, π²/2].
- Refinement studies:
  - Celia: 1 m column, T = 1 d; j = 0…5, Δz = 0.025/2^j m, Δt = 0.5/2^j d (j = 0 is two steps).
  - Szymkiewicz (modified, 0.2 m): t = 0.03 h and 0.15 h; j = 0…6, Δz = 0.005/2^j m, Δt = 0.03/2^j h.
  - Δt₀ = the smallest step for which Galerkin + CN converged within 5·10⁴ iterations.
  - The old Celia figure used the wrong mesh; the V2 runs converge to an independent Picard reference
    (front height 0.435 m).

### 11.2 Termination criterion in proteus (43af6813)
- Numerics file: `levelNonlinearSolverConvergenceTest = 'du'` (or `'r_and_du'`), `nonlinearSolverUNorm = 'L2'`,
  `nl_rtol_du = 1e-12`, `nl_atol_du = 0`.
- `'L2'` is √(xᵀM_C x) via `LevelModel.femL2Norm` (Richards only so far). It matches independent quadrature to 13
  digits on P1 and Q1. In parallel it needs complete M_C rows at owned DOFs (`ownedElementStarsIncomplete()`).
- Increment-only termination controls the residual (the discrete water balance) only if the tolerance is tight.
  Celia j = 0 with the L-scheme (contraction ≈ 0.99, so the error is about 100× the last increment):

  | rtol_du | final ‖r‖ | max head error | stored-water error |
  |---|---|---|---|
  | 1e-4 | 1.4e-5 | 0.13 m | 2e-5 |
  | 1e-6 | 1.4e-7 | 1.3e-3 m | 2e-7 |
  | 1e-12 | 1.3e-13 | — | — |

  Use `'r_and_du'` when the tolerance is loose.

### 11.3 Detector variable (69e41a37)
`Coefficients(FCT_PRELIMITER_VARIABLE='pressure')` is the default (MFEM). `'hydraulic_head'` restores the old
behaviour. Without gravity the two coincide; Hornung–Messing is unchanged (Q1 h = 1/16: 5.229e-5).

### 11.4 1-D parity with the MFEM V2 results
Implicit FCT, θ = 1 and ½, both detectors, all co-author refinement levels. MFEM curves were digitized from the
V2 PDFs, which have the same nodes (40·2^j + 1).
- **Fronts match MFEM to about 1e-4 m** at every level. Node-wise rms differences are 1e-3–1e-2 m and fall with
  refinement.
- The detector choice matters only at Celia θ = ½ on coarse levels, where the pressure detector is closer to MFEM
  (rms 1.06e-2 vs 2.06e-2 at j = 1).
- Newton + fallback: one fallback per run, in the first step. On Szymkiewicz j = 0 at 0.03 h (a single step) it
  undershoots to −7.83 m (θ = 1) and −8.15 m (θ = ½); MFEM stays at −7.5 m.
- **L-scheme with the JNMA settings** (`nl_solver='lscheme'`, `lscheme_L` = L_s/2, `nl_test='du'`,
  `nl_unorm='L2'`, `nl_rtol_du=1e-12`, `nl_max_its=50000`): no fallbacks and no undershoot (min −7.500). The
  Szymkiewicz j = 0 rms drops from 0.158 to 1.9e-2.
- Iterations per solve, proteus vs MFEM notes:
  - Low-order solves match: Celia about 1300, Szymkiewicz about 190 per step.
  - **The first Galerkin step is 3–15× cheaper in proteus** (Celia IE 1563 vs 4508; CN 1684 vs 14 596;
    Szymkiewicz CN 380 vs 5869). Likely the initial guess (proteus `useInitialGuessPredictor` vs MFEM pⁿ with
    new boundary values); not checked.

### 11.5 Ancillary fix (82806fe9)
`LinearAlgebraTools.wl2Norm` wrapped `wDot`, which already reduces, in a second `globalSum`. It was off by √nranks
in parallel (checked on 1, 2 and 4 ranks). Its callers are the multigrid residual norms in `LinearSolvers.py`.
`wl1Norm` and `wlInfNorm` were correct.

### 11.6 Open after 27 Oct
- Check the first-step iteration gap (§11.4): run proteus with p^{n,0} = pⁿ and new boundary values.
- Manuscript changes (applied in the JNMA working copy, not here):
  - the detector is pᴴ, and the §5 extremum argument with gravity must be revisited;
  - the L rule and stopping criterion in §6;
  - refinement figures in §7.2–7.3;
  - a reproducibility table.
- Still unanswered by the co-authors: quadrature, linear solver/tolerance, ℓ, the Hornung–Messing time steps
  and criterion, whether Tables 1–2 predate the pⁿ → pᴴ detector fix, and whether the L-scheme theory covers
  L = L_s/2.

## 12. Pull requests (27 Oct 2026)
- **cekees/proteus#123**: `jnma-fct-1.9.x` → `1.9.x`. All branch commits after `bda13572`. It was mergeable when
  opened.
- **cekees/proteus#124**: `jnma-fct-main` → `main`. The same commits cherry-picked onto `cekees/main` @ `21abf268`,
  plus one porting commit.
  - `main` renamed `xt::pyarray` to `proteus::pyarray`. Conflicts that were only this rename were resolved by taking
    the cherry-picked side with the rename, after checking that none of main's lines were lost.
  - The `LSCHEME_L` conflict in `e353c693` was a real change: that commit removes the earlier Galerkin-Jacobian
    L-scheme variant, and its side was taken.
  - The final commit renames the remaining `xt::pyarray` uses from hunks that applied without conflict. The
    intermediate commits therefore do not build on main by themselves; **squash-merge #124** (or rebase) if
    bisectability matters.
- Verified on main in a separate env built from main's own `environment-openmpi-dev.yml` (`proteus-jnma-main`,
  proteus 2.0.0.dev):
  - `test_richards.py` 7 passed; `test_linear_algebra.py` 12 passed.
  - Hornung–Messing Q1, h = 1/16, θ = 1: implicit FCT 5.229e-5, WRR FCT 6.828e-5, Galerkin 4.150e-6, identical to 1.9.x.
- Local layout used: the 1.9.x checkout at `~/Richards_JNMA/proteus` (env `proteus-jnma-190`) and the main worktree
  at `~/Richards_JNMA/proteus-main-jnma` (env `proteus-jnma-main`).
- Changes to the handoff or the code after this point belong on **both** branches: commit on
  `jnma-fct-1.9.x`, then cherry-pick onto `jnma-fct-main`, porting `xt::pyarray` → `proteus::pyarray` in C++.
- The open questions for the co-authors (§11.6) were drafted as an email to L. Dielmann and D. Kuzmin,
  `~/Richards_JNMA/EMAIL_DRAFT_coauthors.md` (not in this repo).

