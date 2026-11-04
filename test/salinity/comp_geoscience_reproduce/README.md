# Computational Geosciences runs

Richards (flow) STABILIZATION_TYPE 2 + FCT, TADR (transport) STABILIZATION_TYPE 5 + FCT.
The low-order runs (Zalesak FCT=False, MMS MMS_FCT=0, henry/low_order/) turn FCT off in both.
Run each command from the folder named in its heading.

## 1. Zalesak's disk (`zalesak/`)

```bash
# FCT
mpiexec -n 64 parun thelper_tadr_p.py thelper_tadr_n.py -l3 \
  -C "problem=1 nd=2 T=1.0 nDTout=1 refinement=4 STABILIZATION_TYPE=5 cfl=0.5 SSPOrder=2 FCT=True parallel=True" \
  -P "-tadr_ksp_type preonly -tadr_pc_type lu -tadr_pc_factor_mat_solver_type superlu_dist"
# low order
mpiexec -n 64 parun thelper_tadr_p.py thelper_tadr_n.py -l3 \
  -C "problem=1 nd=2 T=1.0 nDTout=1 refinement=4 STABILIZATION_TYPE=5 cfl=0.5 SSPOrder=2 FCT=False parallel=True" \
  -P "-tadr_ksp_type preonly -tadr_pc_type lu -tadr_pc_factor_mat_solver_type superlu_dist"
```

`refinement=5` for the finer mesh.

## 2. Manufactured solution (`mms/`)

```bash
RICHARDS_EDGE_VELOCITY=1 MMS_N=40 MMS_FCT=1 mpiexec -np 1 parun mms_so.py -l3 \
  -P "-richards_ksp_type preonly -richards_pc_type lu -richards_pc_factor_mat_solver_type superlu_dist -tadr_ksp_type preonly -tadr_pc_type lu -tadr_pc_factor_mat_solver_type superlu_dist"
```

`MMS_N` = 10, 20, 40, 80, 160, 320; `MMS_FCT=0` for low order; `MMS_DT` for dt/2
(the values are in `runs/jobs_dt2*.txt`).

## 3. Henry (`henry/` and `henry/low_order/`)

```bash
# henry/ (FCT) and henry/low_order/ (low order), the same command in each
RICHARDS_EDGE_VELOCITY=1 mpiexec -np 4 parun ls_CCS_so.py -l3 \
  -P "-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type superlu_dist -tadr_ksp_type preonly -tadr_pc_type lu -tadr_pc_factor_mat_solver_type superlu_dist"
```

## 4. Elder (`elder/`)

```bash
RICHARDS_EDGE_VELOCITY=1 mpiexec -np 64 parun ls_CCS_so.py -l3 \
  -P "-richards_ksp_type preonly -richards_pc_type lu -richards_pc_factor_mat_solver_type superlu_dist -tadr_ksp_type preonly -tadr_pc_type lu -tadr_pc_factor_mat_solver_type superlu_dist"
```

## 5. Bioswale, bio2d set-up (`bioswale/`)

```bash
RICHARDS_EDGE_VELOCITY=1 mpiexec -np 2 parun ls_CCS_so.py -l3 \
  -P "-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type superlu_dist -tadr_ksp_type preonly -tadr_pc_type lu -tadr_pc_factor_mat_solver_type superlu_dist"
```

## 6. Levee (`levee_density/density/` and `levee_density/non_density/`)

```bash
# levee_density/density/ (rho_s = 1.025) and levee_density/non_density/ (passive tracer)
RICHARDS_EDGE_VELOCITY=1 mpiexec -np 64 parun ls_CCS_so.py -l3 \
  -P "-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type superlu_dist -tadr_ksp_type preonly -tadr_pc_type lu -tadr_pc_factor_mat_solver_type superlu_dist"
```

## On the proteus `vdft-manuscript` line (2026-11-04)

The decks, scripts, solver options and input data here are Barua's, from abarua-ce/proteus `VDFT_barua_kees` at
`c3f611c6`, brought over as files rather than by merging (that branch is based on `81662ec7`, 2026-08-29, and predates the
compatible coupling #150/#152). Left out: the generated figures (`*.png`, `*.pdf`), the crops of the published Simmons
figure (`plot_fig8.py` uses them only when present; they are not ours to redistribute) and the manuscript subsection
`.tex`, which lives in Overleaf.

- **Flags.** The edge-flux transport and its Richards FCT boundary block are on main and 1.9.x since cekees/proteus#173 and
  #174, where the boundary block is opt-in. Every deck here with Richards `FCT=True` therefore passes
  `FCT_BND_DIFF=True, FCT_BND_KRW=True`; without them the runs differ from Barua's builds.
- **Reproduced:** Henry (this deck, TADR STAB 5) on the 1.9.x port equals Barua's build at `c3f611c6` bit for bit (205
  outputs, serial). On main it differs by round-off amplified at the early front: 5e-19 at output 2, 6.2e-3 at output 5,
  3.6e-5 at the final time; RMSE against the semi-analytical solution 2.840e-2 on both.
- **Parallel.** Main's TADR is partition independent (C0 rounds 2-3, cekees/proteus#184's regression test); 1.9.x does not
  carry those scatters, so on 1.9.x run the 64/90-rank figures serially or accept the defect Barua measured.

## Transport used by each figure (cekees, 2026-11-04: both transports stay, each figure labelled)

| figure | transport | Richards | TADR |
|---|---|---|---|
| Zalesak disk | TADR alone (no flow model) | – | STAB 5, FCT on/off |
| MMS T0-T3 | **edge-flux** (Barua) | per `mms_so.py`, flags set | per `mms_so.py` |
| Henry (`henry/`, FCT) | **edge-flux** | STAB 2 + FCT, flags set | STAB 5 + FCT |
| Henry (`henry/low_order/`) | **edge-flux** | STAB 2, FCT off | STAB 5, FCT off |
| Elder | **edge-flux** | STAB 2 + FCT, flags set | STAB 5 |
| Simmons UHD | **edge-flux** | per `flow_p.py`, flags set | per `transport_p.py` |
| bioswale (bio2d set-up) | **edge-flux** | STAB 2 + FCT, flags set | STAB 5 |
| levee density / non-density | **edge-flux** | STAB 2 + FCT, flags set | STAB 5 |
| levee Case_A-D, Henry `ref_3` reruns (`test/salinity/...`, 2026-11-02) | **compatible** | STAB 2 + FCT | `'Compatible'` (serial) |

The comparison of the two transports is in `VDFT_COUPLING_COMPARISON_2026-11-04.md` (local note, cekees).
