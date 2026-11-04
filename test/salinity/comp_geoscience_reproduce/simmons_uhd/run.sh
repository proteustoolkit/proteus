#!/usr/bin/env bash
# usage: run.sh [run dir]   (default: ./run_uhd)
#
# The run dir name must look like run_uhd*: plot_fig8.py reads the case from the
# second "_" field of the name.
#
# Simmons et al. (2002) UHD tank as set up by Guevara Morel & Graf (2023) -- the
# deck of benchmark_problem/UMD/run_uhd_l4_ramp, which wrote fig8_uhd_colormap.png
# (run 2026-10-27, 90 ranks, 70 min; plotted 2026-10-29).
#   mesh    SIMMONS_LEVEL=4: 500 x 500 elements (cases.spatial_level)
#   dt      SIMMONS_THETA=4, SIMMONS_COURANT=0.5
#   flow    Richards FCT, seepage drains at the two top corners (DRAIN=corners)
#   source  time-varying inflow q(t) from data/q_UHD_ramp.csv: the digitised
#           record data/q_UHD.csv with a linear ramp from 0 over the unrecorded
#           first 3 min (header of each file gives the source)
#   trans   TADR STAB=5 (ImplicitEV) + FCT, advected with Richards' edge velocity
#   mu      variable viscosity (SIMMONS_VISC=1), omega_max = 0.2535 (cases.py)
#   ln K    lognormal K0 field, sigma_lnK = 0.01, seed 0, one value per grid
#           rectangle (K0_PER_CELL=1): k0_field.py; the realisation is cached in
#           data/k0_n250000_sig0.01_seed0_ln.npy and regenerated from the seed
#           if absent
# Reference data in data/: z_hgs_UHD.csv / z_lab_UHD.csv (Guevara Morel & Graf
# 2023 Fig. 7, digitize_fig7.py) and the lab-photo masks at 32 and 40 min
# (digitize_uhd_photos.py).  plot_fig8.py --colormap draws t = 32 and 40 min.
# Detach, which otherwise dies with the session:
#   setsid nohup ./run.sh > run.log 2>&1 < /dev/null & disown
set -u
DECK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="$(mkdir -p "${1:-$DECK/run_uhd}" && cd "${1:-$DECK/run_uhd}" && pwd)"
cp "$DECK"/{cases,domain_simmons,flow_n,flow_p,k0_field,penetration_height,simmons_so,transport_n,transport_p}.py \
   "$DECK"/petsc.options.superlu "$DECK/../vdft_metrics.py" "$RUN"/
ln -sfn "$DECK/data" "$RUN/data"
cd "$RUN"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate tadr-check
# parun must import the installed build, not a repo checkout on PYTHONPATH
unset PYTHONPATH
export SIMMONS_CASE=UHD
export SIMMONS_LEVEL=4 SIMMONS_LEVEL_Z=4 SIMMONS_THETA=4 SIMMONS_SEED=0
export SIMMONS_STAB=5 SIMMONS_COURANT=0.5 SIMMONS_FCT=1
export SIMMONS_DRAIN=corners SIMMONS_DRAIN_FACES=1
export SIMMONS_RICHARDS_FCT=1
export SIMMONS_SRC_DBG=1 RICHARDS_EDGE_DBG=1
export RICHARDS_EDGE_VELOCITY=1
export SIMMONS_VISC=1
export SIMMONS_Q_FILE=data/q_UHD_ramp.csv
export SIMMONS_K0_PER_CELL=1 TADR_FCT_DBG=1 OMP_NUM_THREADS=1

{
  echo "date: $(date)"
  echo "conda env: $CONDA_DEFAULT_ENV"
  python -c "import proteus,os;print('proteus:', os.path.dirname(proteus.__file__))"
  echo "repo HEAD: $(git -C "$DECK" rev-parse HEAD)"
  env | grep -E '^(SIMMONS_|RICHARDS_|TADR_)' | sort
} > provenance.txt
# uncommitted source edits at run time; the installed build may still differ
git -C "$(git -C "$DECK" rev-parse --show-toplevel)" diff HEAD -- proteus > source.diff

mpiexec -n "${SIMMONS_NP:-90}" parun simmons_so.py -l3 -O petsc.options.superlu > run.out 2>&1
echo "exit $?" >> run.out

cd "$DECK"
python plot_fig8.py --colormap "$RUN"    # $RUN/fig8_uhd_colormap.png
