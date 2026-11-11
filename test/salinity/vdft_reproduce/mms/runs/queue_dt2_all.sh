#!/usr/bin/env bash
# dt/2 runs for the Richardson time/space split of the T3 error.
# Stage 2: N<=160 (jobs_dt2.txt).  Stage 3: N=320 (jobs_dt2_n320.txt), started
# only if every stage-2 run ended with exit 0 and took twice the steps of its
# dt run (a_T3_fct<F>_N<N>).
cd "$(dirname "$0")"
step_ok() {  # <dt/2 run name>: exit 0 and nsteps = 2 x the dt run
  local base=${1%_dt2}
  tail -1 "$1/run.out" 2>/dev/null | grep -q "^exit 0" || return 1
  local a b
  a=$(grep -o "nsteps=[0-9]*" "$base/mms_so.log" | head -1 | cut -d= -f2)
  b=$(grep -o "nsteps=[0-9]*" "$1/mms_so.log" | head -1 | cut -d= -f2)
  [ -n "$a" ] && [ -n "$b" ] && [ "$b" -eq $((2 * a)) ]
}
./run_jobs.sh jobs_dt2.txt 10 > dt2.log 2>&1
for name in $(cut -d' ' -f1 jobs_dt2.txt); do
  if ! step_ok "$name"; then
    echo "NOT STARTED: $name failed (exit or step count)" > dt2_n320.log
    exit 1
  fi
done
./run_jobs.sh jobs_dt2_n320.txt 2 > dt2_n320.log 2>&1
for name in $(cut -d' ' -f1 jobs_dt2_n320.txt); do
  step_ok "$name" || echo "CHECK: $name failed (exit or step count)" >> dt2_n320.log
done
