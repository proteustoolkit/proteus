#!/usr/bin/env bash
# usage: run.sh [run dir]   (default: ./run)
#
# Zalesak's disk (thelper_tadr.py, problem=1): rigid rotation v = 2*pi*(0.5-y, x-0.5)
# on the unit square, so T = 1 is one revolution.  Mesh refinement 4 gives
# nnx = nny = 2**4*10+1 = 161 nodes (160 x 160 cells, 2 triangles each).
# STABILIZATION_TYPE 5 (ImplicitEV, backward Euler + Newton at cfl 0.5) by default,
# run twice:
#   <run dir>/low  FCT=False  low-order solution uLow (the Newton solution)
#   <run dir>/fct  FCT=True   Zalesak-limited high-order solution
# then plot_disc.py writes <run dir>/disc_r.png.  Serial by default (direct LU);
# ZALESAK_NP=<n> runs each variant on mpiexec -n <n> with PETSc SuperLU_dist.
# Under MPI the FCT result depends on the partition: TADR's FCTStep does not
# ghost-exchange its limiter ratios Rpos/Rneg, so alpha_ij != alpha_ji on edges
# cut by the partition and the limited flux is not conservative there.
set -u
DECK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="$(mkdir -p "${1:-$DECK/run}" && cd "${1:-$DECK/run}" && pwd)"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate tadr-check
# parun must import the installed build, not a repo checkout on PYTHONPATH
unset PYTHONPATH

# FCT=True limits the Newton solution in calculateAuxiliaryQuantitiesAfterStep.
# ZALESAK_STAB=2 runs the explicit EntropyViscosity scheme (SSP2) instead; with
# this deck's consistent mass matrix its FCT=False run is the UNLIMITED high-order
# solution, not low order (ExplicitConsistentMassMatrixForVOF skips FCTStep).
# ZALESAK_REF sets the refinement (4).
OPTS="problem=1 nd=2 T=1.0 nDTout=1 refinement=${ZALESAK_REF:-4} STABILIZATION_TYPE=${ZALESAK_STAB:-5} cfl=0.5 SSPOrder=2"
# the two variants are independent serial runs, so they run concurrently
for v in low fct; do
  (
  mkdir -p "$RUN/$v"
  cp "$DECK"/thelper_tadr{,_p,_n}.py "$RUN/$v"/
  cd "$RUN/$v"
  FCT=$([ $v = fct ] && echo True || echo False)
  {
    echo "date: $(date)"
    echo "conda env: $CONDA_DEFAULT_ENV"
    python -c "import proteus,os;print('proteus:', os.path.dirname(proteus.__file__))"
    echo "repo HEAD: $(git -C "$DECK" rev-parse HEAD)"
    echo "context: $OPTS FCT=$FCT"
  } > provenance.txt
  # uncommitted source edits at run time; the installed build may still differ
  git -C "$(git -C "$DECK" rev-parse --show-toplevel)" diff HEAD -- proteus > source.diff
  if [ "${ZALESAK_NP:-1}" -gt 1 ]; then
    # one thread per rank; SuperLU_dist through the tadr_ prefix
    cp "$DECK"/petsc.options.superlu .
    OMP_NUM_THREADS=1 mpiexec -n "$ZALESAK_NP" parun thelper_tadr_p.py thelper_tadr_n.py -l3 \
      -C "$OPTS FCT=$FCT parallel=True" -O petsc.options.superlu > run.out 2>&1
  else
    parun thelper_tadr_p.py thelper_tadr_n.py -l3 -C "$OPTS FCT=$FCT" > run.out 2>&1
  fi
  echo "exit $?" >> run.out
  ) &
done
wait

cd "$RUN"
python "$DECK/plot_disc.py" "$RUN"
