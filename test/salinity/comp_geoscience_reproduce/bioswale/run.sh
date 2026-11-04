#!/usr/bin/env bash
# usage: run.sh [run dir]   (default: ./run)
#
# Bioswale cross-section, 3 m x 5 m, with a 0.25 m-radius perforated drain pipe
# (seepage boundary, domain_rg.py).  Deck = Bioswale_contamination/start_new as
# committed in 22e2d8fb (2026-05-05), which wrote u_three_times_bioswale.png;
# rerunning it on 2026-05-06 redrew that figure pixel for pixel.
#   flow    Richards, hydrostatic IC, 0.5 m ponding on the top for
#           L/3 <= x <= 2L/3 (1 <= x <= 2 m), three van Genuchten zones
#   trans   TADR STAB 2 + FCT, c_in = 1, Dm = 1.886e-3, alpha_L = 0.1,
#           alpha_T = 0.01, rho_s = 1.025
#   time    T = 0.5 d, nDTout = 201; the figure shows t = 0.1219, 0.2488, 0.5025 d
# The original ran on 2 ranks (36 s) in the petsc-dev-again env with the solver
# options below on the command line; this script runs it in tadr-check.
set -u
DECK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="$(mkdir -p "${1:-$DECK/run}" && cd "${1:-$DECK/run}" && pwd)"
cp "$DECK"/{domain_rg,Raingarden_p,Raingarden_n,tranport_p,tranport_n,ls_CCS_so,plot_u_three_times,plot_domain_bioswale}.py "$RUN"/
cd "$RUN"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate tadr-check
# parun must import the installed build, not a repo checkout on PYTHONPATH
unset PYTHONPATH
export OMP_NUM_THREADS=1

{
  echo "date: $(date)"
  echo "conda env: $CONDA_DEFAULT_ENV"
  python -c "import proteus,os;print('proteus:', os.path.dirname(proteus.__file__))"
  echo "repo HEAD: $(git -C "$DECK" rev-parse HEAD)"
  echo "np: ${BIOSWALE_NP:-2}"
} > provenance.txt
# uncommitted source edits at run time; the installed build may still differ
git -C "$(git -C "$DECK" rev-parse --show-toplevel)" diff HEAD -- proteus > source.diff

mpiexec -np "${BIOSWALE_NP:-2}" parun ls_CCS_so.py -l3 \
  -ksp_type preonly -pc_type lu -pc_factor_mat_solver_type superlu_dist > run.out 2>&1
echo "exit $?" >> run.out

python plot_u_three_times.py && mv u_three_times.png u_three_times_bioswale.png
python plot_domain_bioswale.py    # bioswale_contamination_domain.png
