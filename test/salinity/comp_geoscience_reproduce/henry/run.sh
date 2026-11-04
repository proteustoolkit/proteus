#!/usr/bin/env bash
# usage: run.sh [run dir]   (default: ./run)
#
# Henry problem, right_dbc/ref_1: 2 m x 1 m, nnx x nny = 80 x 40 NODES (3200;
# 79 x 39 cells), T = 3.0e4 s (nDTout = 201), TADR STABILIZATION_TYPE 2.  Two runs,
# each compared with the semi-analytical solution at the end:
#   <run dir>/fct  FCT=True   -> iso_combined_FCT_ref_1.png
#   <run dir>/lo   FCT=False  (low_order/, differs only in FCT) -> iso_combined_Stab_2_lo.png
# The figures in this folder were made on 64 ranks in the petsc-dev-again env with
# the solver options below on the command line (ls_CCS_so.log of ref_1); this
# script runs the same deck in tadr-check.  HENRY_NP overrides the rank count.
# Detach, which otherwise dies with the session:
#   setsid nohup ./run.sh > run.log 2>&1 < /dev/null & disown
set -u
DECK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="$(mkdir -p "${1:-$DECK/run}" && cd "${1:-$DECK/run}" && pwd)"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate tadr-check
# parun must import the installed build, not a repo checkout on PYTHONPATH
unset PYTHONPATH

for v in fct lo; do
  SRC=$DECK; [ $v = lo ] && SRC=$DECK/low_order
  mkdir -p "$RUN/$v"
  cp "$SRC"/{domain_henry,flow_n,flow_p,tranport_n,tranport_p,ls_CCS_so}.py "$RUN/$v"/
  cd "$RUN/$v"
  {
    echo "date: $(date)"
    echo "conda env: $CONDA_DEFAULT_ENV"
    python -c "import proteus,os;print('proteus:', os.path.dirname(proteus.__file__))"
    echo "repo HEAD: $(git -C "$DECK" rev-parse HEAD)"
    echo "np: ${HENRY_NP:-64}"
  } > provenance.txt
  # uncommitted source edits at run time; the installed build may still differ
  git -C "$(git -C "$DECK" rev-parse --show-toplevel)" diff HEAD -- proteus > source.diff
  mpiexec -np "${HENRY_NP:-64}" parun ls_CCS_so.py -l3 \
    -ksp_type preonly -pc_type lu -pc_factor_mat_solver_type superlu_dist > run.out 2>&1
  echo "exit $?" >> run.out
done

# Semi-analytical steady solution (Zidane et al. 2012, WRR 48 W06533, App. A):
# Henry case a = 0.263, b = 0.1, xi = 2, Fourier truncation MA/NA/MB/NB = 7/4/4/19
# (the script's defaults), evaluated at the run's nodes.  These are the settings
# stored in ./henry_reference.npz, the reference the figures in this folder used.
# Both runs share one mesh, so one reference serves both.
cd "$DECK"
python henry_semianalytical.py --h5 "$RUN/fct/ls_CCS_so.h5" --nodes-dataset nodesSpatial_Domain0 \
  --case henry --out "$RUN/henry_reference.npz"
REF="$RUN/henry_reference.npz"

# isochlors 0.1/0.5/0.9 of each scheme against the semi-analytical solution
python plot_iso_combined.py --fct_h5 "$RUN/fct/ls_CCS_so.h5" --npz "$REF" \
  --out "$RUN/iso_combined_FCT_ref_1.png" --fct-label FCT
python plot_iso_combined.py --fct_h5 "$RUN/lo/ls_CCS_so.h5" --npz "$REF" \
  --out "$RUN/iso_combined_Stab_2_lo.png" --fct-label "Low order"
# full-field c vs semi-analytical; prints the nodal RMS and max errors
for v in fct lo; do
  python plot_henry_comparison.py --h5 "$RUN/$v/ls_CCS_so.h5" --npz "$REF" \
    --out "$RUN/henry_comparison_$v.png" 2> "$RUN/henry_error_$v.txt"
done
