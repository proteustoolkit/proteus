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
#
# Ablation rows of Table tab:uhd_coupling (environment):
#   SIMMONS_VELOCITY_COUPLING=eos      velocity-based, storage theta(psi^n) rho(c^n)
#                                   (rows Co = 0.5 and, with SIMMONS_COURANT=0.25,
#                                   Co = 0.25)
#   SIMMONS_VELOCITY_COUPLING=lagged   velocity-based, lagged density
#   SIMMONS_RICHARDS_FCT=0          flow limiter off (caption note)
#   unset                           compatible (edge fluxes, carried fluid mass)
#   SIMMONS_CONSERVATIVE_FORM=1        with a velocity coupling and SIMMONS_FCT=0:
#                                   the conservative solute row instead of the
#                                   cancelled one (low order, serial)
# The solute created by the coupling is the last [TADR-CONT2] line of
# simmons_so.log over the last [TADR-MASS] solute mass (level0/coupling_table.py).
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
export OMP_NUM_THREADS=1
export SIMMONS_CASE=UHD SIMMONS_LEVEL=0 SIMMONS_LEVEL_Z=0 SIMMONS_THETA=4 SIMMONS_SEED=0
export SIMMONS_STAB=5 SIMMONS_COURANT="${SIMMONS_COURANT:-0.5}" SIMMONS_FCT="${SIMMONS_FCT:-1}"
export SIMMONS_DRAIN=corners SIMMONS_DRAIN_FACES=1 SIMMONS_VISC=1
export SIMMONS_RICHARDS_FCT="${SIMMONS_RICHARDS_FCT:-1}"
export SIMMONS_Q_FILE="${SIMMONS_Q_FILE:-data/q_UHD_ramp.csv}" SIMMONS_K0_PER_CELL=1

{
  echo "date: $(date)"
  echo "conda env: $CONDA_DEFAULT_ENV"
  python -c "import proteus,os;print('proteus:', os.path.dirname(proteus.__file__))"
  echo "repo HEAD: $(git -C "$DECK" rev-parse HEAD)"
  echo "SIMMONS_VELOCITY_COUPLING: ${SIMMONS_VELOCITY_COUPLING:-}"
  echo "SIMMONS_CONSERVATIVE_FORM: ${SIMMONS_CONSERVATIVE_FORM:-0}"
  env | grep -E '^(SIMMONS_|RICHARDS_)' | sort
} > provenance.txt
# uncommitted source edits at run time; the installed build may still differ
git -C "$(git -C "$DECK" rev-parse --show-toplevel)" diff HEAD -- proteus > source.diff
# SIMMONS_NP sets the rank count (1); SIMMONS_CONSERVATIVE_FORM=1 is serial only
mpiexec -n "${SIMMONS_NP:-1}" parun simmons_so.py -l3 -O petsc.options.superlu > run.out 2>&1
echo "exit $?" >> run.out
