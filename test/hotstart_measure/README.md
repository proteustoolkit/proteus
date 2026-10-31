# Hotstart measurement (campaign C17)

How far does a hotstarted run drift from the uninterrupted run it continues,
and which state is responsible? These scripts are the measurement and are the
acceptance test for a precise hotstart: **the comparison below should read
~1e-16 on every field when C17 is done.**

Nothing here is collected by pytest (no `test_*.py`); it is a harness.

## Case

`test/levelset/vortex2D`: the oscillating-vortex level-set benchmark, 21 x 21
P1 triangles, four models in sequence (level set `ls`, redistancing `rdls`,
`vof`, mass correction `phicor`), T = 8 in 80 user steps, archive every user
step. ls and vof carry lagged shock capturing (`lag_shockCapturing = True`).
Serial; one run takes about 11 s. The time integrator is switched with an
argument: `be` (BackwardEuler_cfl, first order) or `vbdf` (variable-coefficient
BDF, order ramps 1 -> 2).

Stop at step N = 40 (t = 4), hotstart from that archive, run on to t = 8, and
compare each archived field with the uninterrupted run, step by step.

## Scripts

| script | what it does |
|---|---|
| `run_case.py OUTDIR TI TORDER MODE [N]` | one run. MODE `full` (0 -> T), `stop` (0 -> t_N), `hot` (hotstart at t_N, run on to T) |
| `compare.py A.h5 B.h5 NAME FROM_STEP` | per field, relative L2 and Linf difference at each archived step |
| `run_matrix.sh PY ROOT TI TORDER N` | full, full again (noise floor), stop + hot, compare |
| `oracle.sh PY ROOT TI TORDER N GROUPS...` | hotstart, then overwrite groups of the rebuilt state with the uninterrupted run's values and see which group removes the error |
| `snapshot.py`, `diff_snap.py` | flatten ~1,100 Python-visible state values of a `NumericalSolution`; diff two of them; `restore` writes a saved state back |

Environment variables for `run_case.py`: `HS_SNAP=file` (state snapshot: after
the stop run, or after the hotstart's setup), `HS_ORACLE=groups` with
`HS_TRUE=file` (the oracle), `HS_ENDN=k` (end a hot run at step k),
`HS_SNAP_END`, `HS_SNAP_MID` (state mid-run in a full run), `HS_REC` (record
each model's dofs as its step is taken), `HS_SPIN=ops` (apply the hotstart's
spin-up calls to a live uninterrupted run), `HS_TAG`, `HS_LAG`, `HS_RD`.
(`HS_RD=0` fails in `NS_base` for this case and `HS_LAG=0` blows up the
level-set solve at step 4 and still exits normally; neither is used below.)

```
PY=$CONDA_PREFIX/bin/python
test/hotstart_measure/run_matrix.sh $PY /tmp/hs be 1 40
test/hotstart_measure/oracle.sh $PY /tmp/hs vbdf 2 40 none sc,fixalias ti,fixalias ti,sc,fixalias
```

## Results (2026-10-25, proteus `9a3f1160`, macOS arm64, Python 3.13)

**Noise floor: exactly 0.** Two uninterrupted runs are bit-identical, so every
number below is a real difference.

Plain hotstart at N = 40. Relative L2 / Linf of the difference, first step
after the restart (step 41) and at t = 8 (step 80). `phiCorr` is the mass
correction, near zero, so only Linf means anything.

| integrator | field | step 41 | step 80 |
|---|---|---|---|
| BackwardEuler + lagged shock capturing | ls `u` | 1.9e-3 / 2.2e-3 | 6.0e-2 / 3.2e-2 |
| | `phid` | 2.3e-3 / 2.5e-3 | 6.1e-2 / 3.2e-2 |
| | `vof` | 7.8e-3 / 2.5e-2 | 0.29 / 0.87 |
| | `phiCorr` | Linf 3.5e-4 | Linf 3.8e-4 |
| VBDF (order 2) + lagged shock capturing | ls `u` | 0.20 / 0.17 | 0.56 / 0.26 |
| | `phid` | 1.0 / 0.27 | 1.3 / 0.34 |
| | `vof` | 0.13 / 0.38 | 1.4 / 1.0 |
| | `phiCorr` | Linf 9.9e-2 | Linf 8.8e-2 |

Dependence on where the run is stopped (ls `u`, relative L2):

| N | BE step N+1 | BE step 80 | VBDF2 step N+1 | VBDF2 step 80 |
|---|---|---|---|---|
| 20 | 2.7e-3 | 2.5e-2 | 0.10 | 0.56 |
| 40 | 1.9e-3 | 6.0e-2 | 0.20 | 0.56 |
| 60 | 1.6e-3 | 4.8e-2 | 0.39 | 0.52 |

The vortex deforms the interface, so a 2e-3 first-step error is amplified to
O(0.1) by t = 8. A vof Linf of 1 is a wrong phase fraction at some nodes.

## Which state is responsible (the oracle)

Hotstart, then overwrite one group of the rebuilt state with the uninterrupted
run's value at step N, before step N+1. Relative L2 of ls `u`, step 41 / step 80.

| restored | values | BE + lag | VBDF2 |
|---|---|---|---|
| nothing | 0 | 1.9e-3 / 6.0e-2 | 0.20 / 0.56 |
| dofs, `q`, controllers, coefficients, ebq arrays (each alone) | 12-550 | no change | |
| time-integration history | 101 / 120 | 1.9e-3 / 6.0e-2 | 2.0e-3 / 9.5e-2 |
| shock-capturing and subgrid lag state (`sc`, with the alias rebuilt) | 139 | **8e-17 / 1.7e-16** | 0.17 / 0.55 |
| time-integration history **and** lag state | 258 | | **8e-17 / 2.3e-14** |
| everything (about 1,100 values) | 1,100 | 8e-17 / 1.2e-16 | 8e-17 / 2.3e-14 |

For BE the whole error, in every field, is the lagged shock-capturing state.
For VBDF2 it is two things, the BDF history (100x of the error) and the lag
state (the rest), and both are needed.

## Mechanism

Why the lag state is wrong after a hotstart, found by bisecting the hotstart's
spin-up on a live, uninterrupted run (`HS_SPIN`):

* `lm.getResidual(...)` followed by `lm.initializeTimeHistory()` (what
  `NumericalSolution.calculateSolution` does for every model on a hotstart, and
  on a cold start) copies `numDiff` recomputed *from the current solution* into
  `numDiff_last`. The uninterrupted run's `numDiff_last` is the value from the
  last evaluation inside the previous step. Applied to the uninterrupted run at
  step 40 this reproduces the hotstart error (1.96e-3 vs 1.86e-3) with no
  archive involved; restoring only `sc` afterwards removes it exactly.
* `mprans.NCLS.ShockCapturing` is a state machine, not an array: it starts with
  `lag = False`, `nStepsToDelay = 1`, `numDiff_last` *aliased* to `numDiff`,
  counts steps in `nSteps`, and after the first step flips `lag = True` and
  swaps in a separate copy. A restarted run begins the machine again, so the
  first step after a hotstart is solved unlagged (4 Newton iterations in the
  oracle run with every value restored but the alias; the uninterrupted run
  takes 1). Writing values back into an aliased array cannot
  restore the machine (`fixalias` rebuilds the copy), which is why restoring the
  values alone left ls at 2.6e-3.
* VBDF keeps `timeOrder` (current order, starts at 1 and ramps), `m_history`,
  `mt_history`, `dt_history`, `alpha_bdf`, `beta_bdf`; a restart returns to
  first order with no history.

## Not measured

* `FLCBDF`: the `flcbdfWrappers` extension is commented out in `setup.py`, so
  `TimeIntegration.FLCBDF` and the `FLCBDF_controller*` raise `ImportError`.
  No case can be built until it is restored or the classes are removed.
* The mprans SSP-RK integrators (`NCLS.RKEV`, `VOF.RKEV`: `u_dof_last`,
  `u_dof_stage`, `lstage`, `dtLast`). They need `STABILIZATION_TYPE > 0`, which
  the vortex's `UnitSquareVortex` coefficients do not offer; a VOF-only case is
  the next experiment.
* `VBDF` with `timeOrder` 1 and 3 give identical results, different from 2.
  Not understood.
* Parallel, and the momentum models that keep `dof_last_last` (RANS2P family).

## RANS3PF (2026-10-31): `run_rans3p.py`, `compare_rans3p.py`

`test/HotStart_3P`, the 2D cylinder case with `mprans.RANS3PF` (P1 velocity, single phase, four models:
momentum, pressure increment, pressure, pressure initial), `ns_lag_shockCapturing = ns_lag_subgridError = True`
(RANS3PF switches the lag on after `nStepsToDelay`), BackwardEuler_cfl, `dt_fixed = 0.005`, 40 steps, archive
every step. Same three modes as `run_case.py` (`full`, `stop N`, `hot N`); `HS_DUMP` (stop run) and
`HS_INJECT` + `HS_TRUE` (hot run) are the oracle, by owner prefix (`sc`, `sge`, `ti`, `lm`, `ctl`, `sys`, `q`,
`ebqe`, `all`, a digit narrows to one model) and `HS_QKEYS` to pick `q` arrays.

Noise floor 0. Hot start at N = 20, relative L2 against the uninterrupted run: steps 20 and 21 are
bit-identical, step 22 first differs (u Linf 1.5e-3, p 0.58, pInc 0.28), step 41 u 1.7e-4, v 1.0e-4, p 9.5e-2.
Oracle (`HS_INJECT`, every attribute listed is put back after the spin-up; relative L2 of u at step 40):

| restored | u at step 40 |
|---|---|
| nothing | 1.7e-4 |
| every attribute of every model's shock capturing, subgrid error, time integration, step controller, system controller | 1.7e-4 |
| all `q` arrays | 3.3e-5 |
| `q[('u_last', 0)]` of the pressure model alone (delta-debugging over 67 keys) | 3.3e-5 |
| that and the pressure model's `q_p_sharp`, `ebqe_p_sharp`, `q_grad_p_sharp`, `ebqe_grad_p_sharp` | 1.3e-7 |
| everything above plus all owners and numerical fluxes | 1.3e-7 |

So the drift was the pressure model's lagged pressure (`mprans.Pres`, set in `postStep`) and the sharp pressure it hands
the momentum model: a hot start rebuilt `u_last` and `p_sharp` from the restart state, and they were not archived.
They live on the coefficients object, so the solver_state archive now has a `coefficients` owner, and
`mprans.Pres.Coefficients` declares `state()` / `set_state()`. With it a plain hot start gives u 1.3e-7, p 1.1e-6
(relative L2 at step 40), the same as the oracle row above: 1300x less drift, and the archive reproduces it.
`test/test_hotstart_rans3pf.py` asserts it (and that it fails without the owner, `HS_NO_COEF=1`).

The 1.3e-7 floor was the stepping loop's own `firstStep` flag, which stayed True after a restart: RANS3PF's `preStep`
then reset the entropy residual to 1e15 (a low-order first step) and skipped the extrapolation from the previous
solutions, whatever state had been restored. (The first differing step reads 22, not 21, only because the hot run
archives the restart point again as step 21.) A continue-mode hot start now sets `firstStep = False` when every model
restored its state, and `RANS3PF.Coefficients` and `RANS3PF.ShockCapturing` declare what they need: entropy residual,
the two previous solutions, `dt_last`, the velocity at quadrature points, and `numDiff_last` with its lag machine. The
oracle found that set by delta debugging; the subgrid error needs nothing. A plain hot start is now bit-identical to the
uninterrupted run (`HS_FIRST_FALSE=1` puts the flag back in an oracle run that restores state by hand).
