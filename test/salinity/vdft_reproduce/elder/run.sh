#!/usr/bin/env bash
# usage: run.sh [run dir]   (default: ./run)
#
# Elder problem, 20 years -- the deck of Elder_problem/stab_5/run_uhdcfg_20yr_nocap,
# which wrote elder_u_colour_velocity_20yr.png and elder_isochlors_velocity_20yr.png
# (2026-10-29, 64 ranks, 8 min).
#   mesh   601 x 151 nodes, structured triangles (domain_elder.py), 600 m x 150 m
#   time   T = 7305 d (20 yr), nDTout = 240 (DT = 30.4375 d), no dt cap: both
#          models step on the shared HeuristicNL_dt_controller
#   flow   Richards FCT=True, saturated VG with mvg_alpha = 1e-5 (flow_p.py)
#   trans  TADR FCT=True (tranport_p.py), advected with Richards' edge velocity
#          (EDGE_VELOCITY=True in the transport deck)
#   mu     constant (Voss & Souza 1987), folded into K = 0.4107 m/d
# Plots: 2 x 2 panels at 2, 4, 10, 20 yr (Voss & Souza 1987 Fig. 9a times).
# Detach, which otherwise dies with the session:
#   setsid nohup ./run.sh > run.log 2>&1 < /dev/null & disown
# Log lines to check: "advecting with the flow model's edge fluxes: YES",
# "edge fluxes assembled from Richards' edge velocity: YES", and [Coupling rho q]
# u max staying at 1.0.
set -u
DECK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="${1:-$DECK/run}"
mkdir -p "$RUN"
cp "$DECK"/{domain_elder,flow_n,flow_n_base,flow_p,tranport_n,tranport_n_base,tranport_p,ls_CCS_so,plot_isochlors_velocity}.py \
   "$DECK"/petsc.options.superlu "$DECK/../vdft_metrics.py" "$RUN"/
cd "$RUN"

source ~/miniforge3/etc/profile.d/conda.sh
set +u   # conda activate scripts read unset variables
conda activate "${PROTEUS_ENV:-tadr-check}"
set -u
# parun must import the installed build, not a repo checkout on PYTHONPATH
unset PYTHONPATH
export OMP_NUM_THREADS=1

{
  echo "date: $(date)"
  echo "conda env: $CONDA_DEFAULT_ENV"
  python -c "import proteus,os;print('proteus:', os.path.dirname(proteus.__file__))"
  echo "repo HEAD: $(git -C "$DECK" rev-parse HEAD)"
  env | grep -E '^(RICHARDS_|TADR_)' | sort
} > provenance.txt
# uncommitted source edits at run time; the installed build may still differ
git -C "$(git -C "$DECK" rev-parse --show-toplevel)" diff HEAD -- proteus > source.diff

mpiexec -np "${ELDER_NP:-64}" parun ls_CCS_so.py -l3 -O petsc.options.superlu > run.out 2>&1
echo "exit $?" >> run.out

python plot_isochlors_velocity.py           # elder_isochlors_velocity_20yr.png
python plot_isochlors_velocity.py colour    # elder_u_colour_velocity_20yr.png
