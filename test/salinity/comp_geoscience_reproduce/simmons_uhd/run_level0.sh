#!/usr/bin/env bash
# usage: run_level0.sh [run dir]   (default: ./run_uhd_l0)
#
# UHD on the coarsest benchmark mesh (level 0: 100 x 100 cells, 10,201 nodes, 432 steps),
# serial: the mesh of the coupling ablation (Section "Compatible versus velocity-based
# coupling").  Same settings as run.sh otherwise.  vdft_metrics.json reports the run
# measures and the predicted defect of an element-conservative (RT0/BDM1) velocity.
# Needs the digitised inflow record: SIMMONS_Q_FILE (default data/q_UHD_ramp.csv, not yet
# committed).  PROTEUS_ENV and PROTEUS_SRC pick the build.
# 2026-11-04: on main (cekees/vdft-manuscript) this case overshoots c <= 1 by 4.2% from
# t = 32 s in the low-order type-5 solution; the 1.9.x port stays in [0, 1].
set -u
DECK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="$(mkdir -p "${1:-$DECK/run_uhd_l0}" && cd "${1:-$DECK/run_uhd_l0}" && pwd)"
cp "$DECK"/{cases,domain_simmons,flow_n,flow_p,k0_field,penetration_height,simmons_so,transport_n,transport_p}.py \
   "$DECK"/petsc.options.superlu "$DECK/../vdft_metrics.py" "$RUN"/
[ -e "$RUN/data" ] || ln -sfn "$DECK/data" "$RUN/data"
cd "$RUN"
source ~/miniforge3/etc/profile.d/conda.sh
set +u; conda activate "${PROTEUS_ENV:-tadr-check}"; set -u
if [ -n "${PROTEUS_SRC:-}" ]; then export PYTHONPATH="$PROTEUS_SRC"; else unset PYTHONPATH; fi
export RICHARDS_EDGE_VELOCITY=1 OMP_NUM_THREADS=1
export SIMMONS_CASE=UHD SIMMONS_LEVEL=0 SIMMONS_LEVEL_Z=0 SIMMONS_THETA=4 SIMMONS_SEED=0
export SIMMONS_STAB=5 SIMMONS_COURANT=0.5 SIMMONS_FCT=1
export SIMMONS_DRAIN=corners SIMMONS_DRAIN_FACES=1 SIMMONS_RICHARDS_FCT=1 SIMMONS_VISC=1
export SIMMONS_Q_FILE="${SIMMONS_Q_FILE:-data/q_UHD_ramp.csv}" SIMMONS_K0_PER_CELL=1
mpiexec -n 1 parun simmons_so.py -l3 -O petsc.options.superlu > run.out 2>&1
echo "exit $?" >> run.out
