#!/usr/bin/env bash
# usage: run.sh [run dir]   (default: ./run)
#
# Bioswale cross-section, 3 m x 5 m: storage zone, trapezoidal bioswale zone, base soil,
# and a 0.25 m-radius drain pipe whose wall is impermeable (no seepage), domain_rg.py.
#   flow    Richards STAB 2 + FCT, psi = -3 m uniform IC, 0.5 m ponding on the top for
#           1 <= x <= 2 m, every other boundary no-flow
#   trans   TADR STABILIZATION_TYPE 5 + FCT, advected with the reconstructed flow flux
#           (EDGE_VELOCITY=True in the transport deck); c_in = 1, Dm = 1.886e-3 m^2/d, alpha_L = 0.1,
#           alpha_T = 0.01, rho_s = 1.025 in TADR only (the flow deck has no
#           density_model, so the flow sees no density feedback; beta = 1e-4)
#   time    T = 0.5 d, nDTout = 201; the figure shows t = 0.1219, 0.2488, 0.5025 d
# Outputs: u_three_times_bioswale.png, bioswale_contamination_domain.png,
# vdft_metrics.json (conservation, bounds, cost).  The 2026-05 figure was made with
# TADR STABILIZATION_TYPE 2 (explicit); this script replaces it (2026-11-04).
# BIOSWALE_NP sets the rank count (default 2); PROTEUS_ENV and PROTEUS_SRC pick the build.
set -u
DECK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="$(mkdir -p "${1:-$DECK/run}" && cd "${1:-$DECK/run}" && pwd)"
cp "$DECK"/{domain_rg,Raingarden_p,Raingarden_n,tranport_p,tranport_n,ls_CCS_so,plot_u_three_times,plot_u_velocity,plot_domain_bioswale}.py "$DECK/../vdft_metrics.py" "$RUN"/
cd "$RUN"

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

{
  echo "date: $(date)"
  echo "conda env: $CONDA_DEFAULT_ENV"
    echo "proteus HEAD: $(git -C "$(python -c 'import proteus,os;print(os.path.dirname(proteus.__file__))')" rev-parse HEAD 2>/dev/null)"
  python -c "import proteus,os;print('proteus:', os.path.dirname(proteus.__file__))"
  echo "repo HEAD: $(git -C "$DECK" rev-parse HEAD)"
  echo "np: ${BIOSWALE_NP:-2}"
  echo "BIOSWALE_DENSITY: ${BIOSWALE_DENSITY:-0}"
} > provenance.txt
# uncommitted source edits at run time; the installed build may still differ
git -C "$(git -C "$DECK" rev-parse --show-toplevel)" diff HEAD -- proteus > source.diff

mpiexec -np "${BIOSWALE_NP:-2}" parun ls_CCS_so.py -l3 \
  -P "-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type superlu_dist -tadr_ksp_type preonly -tadr_pc_type lu -tadr_pc_factor_mat_solver_type superlu_dist" > run.out 2>&1
echo "exit $?" >> run.out

python plot_u_three_times.py && mv u_three_times.png u_three_times_bioswale.png
python plot_u_velocity.py         # u_colour_velocity_bioswale.png: u, isochlors, Darcy streamlines
python plot_domain_bioswale.py    # bioswale_contamination_domain.png
