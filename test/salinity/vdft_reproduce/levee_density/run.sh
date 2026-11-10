#!/usr/bin/env bash
# usage: run.sh [run dir]   (default: ./run)
#
# Levee cross-section, x = 0..160 m, z = 0..57 m, homogeneous soil, density-coupled
# vs passive-tracer runs.
#   flow    Richards STAB 2 + FCT, IC psi = 45 - z (water table 45 m), psi = 48 - z on
#           the wetted river face (leftTop), no flow on the rest of the left side, top
#           and bottom, and no condition (zero flux) on the right
#   trans   TADR STABILIZATION_TYPE 5 + FCT, advected with the reconstructed flow flux
#           (EDGE_VELOCITY=True in the transport deck); c_in = 0.7 on the wetted river face,
#           Dm = 1.886e-2 m^2/d, alpha_L = 0.1 m, alpha_T = 0.01 m
#   density density/: rho_s = 1.025 (eps = 0.025); non_density/: rho_s = 1,
#           DENSITY_MODEL=None.  Also different: a0 (unused) and the flow line search
#           (10 vs 0)
#   time    T = 2.0e4 d, nDTout = 801; the figures show archived steps 100, 400
#           and the last: t = 2446.9, 9937.6, 20025.0 d
# Outputs per run: u_three_times_overlay_<v>.png, u_range.txt, vdft_metrics.json.
# The 2026-05 figures were made with TADR STABILIZATION_TYPE 2 (explicit); this script
# replaces them (2026-11-04).  LEVEE_NP sets the rank count (default 64);
# PROTEUS_ENV and PROTEUS_SRC pick the build.
set -u
DECK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="$(mkdir -p "${1:-$DECK/run}" && cd "${1:-$DECK/run}" && pwd)"

source ~/miniforge3/etc/profile.d/conda.sh
set +u   # conda activate scripts read unset variables
conda activate "${PROTEUS_ENV:-tadr-check}"
set -u
# PROTEUS_SRC=<proteus checkout built in place> runs that build; otherwise parun must
# import the env's installed build, not a repo checkout on PYTHONPATH
if [ -n "${PROTEUS_SRC:-}" ]; then export PYTHONPATH="$PROTEUS_SRC"; else unset PYTHONPATH; fi
# The coupling the manuscript describes (Sec. "Compatibility with the flow model"):
# TADR advects with the mass flux Richards reconstructs element by element from its
# realized edge exchanges.  Without it STABILIZATION_TYPE 5 copies E_ij as the edge flux.
export OMP_NUM_THREADS=1

for v in density non_density; do
  (
  mkdir -p "$RUN/$v"
  cp "$DECK/$v"/*.py "$DECK/../vdft_metrics.py" "$RUN/$v"/
  cd "$RUN/$v"
  {
    echo "date: $(date)"
    echo "conda env: $CONDA_DEFAULT_ENV"
    echo "proteus HEAD: $(git -C "$(python -c 'import proteus,os;print(os.path.dirname(proteus.__file__))')" rev-parse HEAD 2>/dev/null)"
    python -c "import proteus,os;print('proteus:', os.path.dirname(proteus.__file__))"
    echo "repo HEAD: $(git -C "$DECK" rev-parse HEAD)"
    echo "np: ${LEVEE_NP:-64}"
  } > provenance.txt
  # uncommitted source edits at run time; the installed build may still differ
  git -C "$(git -C "$DECK" rev-parse --show-toplevel)" diff HEAD -- proteus > source.diff
  mpiexec -np "${LEVEE_NP:-64}" parun ls_CCS_so.py -l3 \
    -P "-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type superlu_dist -tadr_ksp_type preonly -tadr_pc_type lu -tadr_pc_factor_mat_solver_type superlu_dist" > run.out 2>&1
  echo "exit $?" >> run.out
  # u range over every step, from the [Coupling rho q] lines; max must stay <= 0.7
  grep -o 'u (min,max,mean)=([^)]*' ls_CCS_so.log | tr '(,' '  ' | \
    awk '{if (NR==1 || $5<lo) lo=$5; if (NR==1 || $6>hi) hi=$6} END {print "u range over the run:", lo, hi}' > u_range.txt
  python plot_u_three_times_overlay.py && mv u_three_times_overlay.png "u_three_times_overlay_$v.png"
  ) &
done
wait

cd "$RUN"
python "$DECK/plot_domain_levee.py"    # levee_river_side_no_plume.png
