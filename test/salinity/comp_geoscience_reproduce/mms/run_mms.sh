#!/usr/bin/env bash
# usage: run_mms.sh <run dir> [VAR=value ...]
#
# One coupled-MMS run: copies the deck into <run dir>, exports the MMS_* knobs
# given as VAR=value (see mms_exact.py and domain_mms.py), and runs it with the
# tadr-check build.  MMS_NP sets the MPI rank count (1).  Runs in the
# foreground; detach long runs, which otherwise die with the session:
#   setsid nohup ./run_mms.sh <run dir> ... > /dev/null 2>&1 < /dev/null & disown
set -u
DECK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="$1"
shift
mkdir -p "$RUN"
cp "$DECK"/{mms_exact,domain_mms,flow_p,flow_n,transport_p,transport_n,mms_so}.py \
   "$DECK"/petsc.options.superlu "$RUN"/
cd "$RUN"

source ~/miniforge3/etc/profile.d/conda.sh
set +u   # conda activate scripts read unset variables
conda activate "${PROTEUS_ENV:-tadr-check}"
set -u
# parun must import the installed build, not a repo checkout on PYTHONPATH
unset PYTHONPATH
export RICHARDS_EDGE_VELOCITY=1
for kv in "$@"; do export "$kv"; done
env | grep -E '^(MMS_|RICHARDS_)' | sort > run.env
echo "repo HEAD: $(git -C "$DECK" rev-parse HEAD)" > provenance.txt
# uncommitted source edits at run time; the installed build may still differ
git -C "$(git -C "$DECK" rev-parse --show-toplevel)" diff HEAD -- proteus > source.diff

python -c "import proteus.mprans.cTADR as c; print('cTADR from', c.__file__)" > run.out
mpiexec -np "${MMS_NP:-1}" parun mms_so.py -l3 -O petsc.options.superlu >> run.out 2>&1
echo "exit $?" >> run.out
