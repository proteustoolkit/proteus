#!/bin/sh
# state_oracle_run.sh OUTDIR T0,T1,... [HOTSTARTMODE]
# The deck is given as shell assignments in DECKENV for test/step_retry_mpi.py, e.g.
#   DECKENV='DECK=damBreakLevelSet.py OPTS="he=0.1" FT=0.1 DTOUT=0.02'
#   DECKENV='DIR=ci SO="ladr_2d_p.py ladr_2d_n.py"'
# Full run; stop run at output time T0 (STOPAT); hot start at T0 on a copy of the stop run's archive. The state dumps
# (state_oracle.py; per rank, _p<rank>; RUNNER='mpiexec -n 2' runs in parallel, the diff shows rank 0) are OUTDIR/{full,hot}_<k>_p0.pkl for the output times T0, T1, ... (output times of the deck: k = 0 is
# the restore, before the first step; later ones show which differences at T0 are recomputed before they are read).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
HELPER=$HERE/step_retry_mpi.py
mkdir -p "$1"
OUT=$(cd "$1" && pwd)
TS=$2; T=${TS%%,*}; MODE=${3:-continue}
mkdir -p "$OUT/full" "$OUT/stop" "$OUT/hot"
export OMP_NUM_THREADS=1 PYTHONHASHSEED=${PYTHONHASHSEED:-1}
( cd "$OUT/full" && eval "env STATEDUMP=full STATEDUMPT=$TS $DECKENV ${RUNNER:-} python \"$HELPER\"" > log.txt 2>&1 )
( cd "$OUT/stop" && eval "env STOPAT=$T $DECKENV ${RUNNER:-} python \"$HELPER\"" > log.txt 2>&1 )
cp -r "$OUT/stop/." "$OUT/hot/"
( cd "$OUT/hot" && eval "env STATEDUMP=hot STATEDUMPT=$TS HOTSTART=$T HOTSTARTMODE=$MODE $DECKENV ${RUNNER:-} python \"$HELPER\"" > log.txt 2>&1 )
k=0
for t in $(echo "$TS" | tr , ' '); do
  echo "== t = $t (dump $k)"
  python "$HERE/state_oracle.py" "$OUT/full/full_${k}_p0.pkl" "$OUT/hot/hot_${k}_p0.pkl" "${FLOOR:-0}" | head -${HEAD:-14} || true
  k=$((k+1))
done
