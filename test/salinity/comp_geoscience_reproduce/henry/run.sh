#!/usr/bin/env bash
# usage: run.sh [run dir]   (default: ./run)
#
# Henry problem: 2 m x 1 m, nnx x nny = 80 x 40 NODES (3200; 79 x 39 cells),
# T = 3.0e4 s (nDTout = 201).  Flow Richards STAB 2 (+ FCT in fct/); transport TADR
# STABILIZATION_TYPE 5 (backward Euler, cancelled form, + FCT in fct/) advected with the
# reconstructed flow flux (RICHARDS_EDGE_VELOCITY=1).  Two runs, each compared with the
# semi-analytical solution at the end:
#   <run dir>/fct  FCT=True   -> henry_iso_fct.png
#   <run dir>/lo   FCT=False  (low_order/, differs only in FCT) -> henry_iso_lo.png
#   <run dir>/henry_metrics.json   errors of Table tab:henry_results
#   <run dir>/{fct,lo}/vdft_metrics.json   conservation, bounds, cost
# The 2026-05 figures were made with TADR STABILIZATION_TYPE 2 (explicit); these
# decks and this script replace them (2026-11-04).  HENRY_NP sets the rank count
# (default 4, as in the manuscript); PROTEUS_ENV and PROTEUS_SRC pick the build.
# HENRY_VARIANTS picks the runs ("fct lo"; the plots of a variant not run fail harmlessly).
# Detach, which otherwise dies with the session:
#   setsid nohup ./run.sh > run.log 2>&1 < /dev/null & disown
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
export RICHARDS_EDGE_VELOCITY=1

for v in ${HENRY_VARIANTS:-fct lo}; do
  SRC=$DECK; [ $v = lo ] && SRC=$DECK/low_order
  mkdir -p "$RUN/$v"
  cp "$SRC"/{domain_henry,flow_n,flow_p,tranport_n,tranport_p,ls_CCS_so}.py "$DECK/../vdft_metrics.py" "$RUN/$v"/
  cd "$RUN/$v"
  {
    echo "date: $(date)"
    echo "conda env: $CONDA_DEFAULT_ENV"
    echo "RICHARDS_EDGE_VELOCITY: $RICHARDS_EDGE_VELOCITY"
    echo "proteus HEAD: $(git -C "$(python -c 'import proteus,os;print(os.path.dirname(proteus.__file__))')" rev-parse HEAD 2>/dev/null)"
    python -c "import proteus,os;print('proteus:', os.path.dirname(proteus.__file__))"
    echo "repo HEAD: $(git -C "$DECK" rev-parse HEAD)"
    echo "np: ${HENRY_NP:-4}"
  } > provenance.txt
  # uncommitted source edits at run time; the installed build may still differ
  git -C "$(git -C "$DECK" rev-parse --show-toplevel)" diff HEAD -- proteus > source.diff
  mpiexec -np "${HENRY_NP:-4}" parun ls_CCS_so.py -l3 \
    -P "-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type superlu_dist -tadr_ksp_type preonly -tadr_pc_type lu -tadr_pc_factor_mat_solver_type superlu_dist" > run.out 2>&1
  echo "exit $?" >> run.out
done

# Semi-analytical steady solution (Zidane et al. 2012, WRR 48 W06533, App. A):
# Henry case a = 0.263, b = 0.1, xi = 2, Fourier truncation MA/NA/MB/NB = 7/4/4/19
# (the script's defaults), evaluated at the run's nodes.  Both runs share one
# mesh, so one reference serves both.
cd "$DECK"
python henry_semianalytical.py --h5 "$RUN/fct/ls_CCS_so.h5" --nodes-dataset nodesSpatial_Domain0 \
  --case henry --out "$RUN/henry_reference.npz"
REF="$RUN/henry_reference.npz"

# isochlors 0.1/0.5/0.9 of each scheme against the semi-analytical solution
python plot_iso_combined.py --fct_h5 "$RUN/fct/ls_CCS_so.h5" --npz "$REF" \
  --out "$RUN/henry_iso_fct.png" --fct-label FCT
python plot_iso_combined.py --fct_h5 "$RUN/lo/ls_CCS_so.h5" --npz "$REF" \
  --out "$RUN/henry_iso_lo.png" --fct-label "Low order"
# full-field c vs semi-analytical; prints the nodal RMS and max errors
for v in fct lo; do
  python plot_henry_comparison.py --h5 "$RUN/$v/ls_CCS_so.h5" --npz "$REF" \
    --out "$RUN/henry_comparison_$v.png" 2> "$RUN/henry_error_$v.txt"
done
# Table tab:henry_results: lumped-mass L2 (whole domain and x <= 1.8 m), Linf, and the
# bottom intercepts of the 10/50/90 % isochlors against the reference
python henry_metrics.py --fct "$RUN/fct/ls_CCS_so.h5" --lo "$RUN/lo/ls_CCS_so.h5" \
  --npz "$REF" --out "$RUN/henry_metrics.json"
