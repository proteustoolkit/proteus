#!/usr/bin/env bash
# usage: run_jobs.sh <jobs file> <parallel jobs>; each line: <run name> VAR=value ...
# A run whose run.out already ends in "exit 0" is skipped, so the batch can be restarted.
R="$(cd "$(dirname "$0")" && pwd)"
DECK="$(dirname "$R")"
run_one() {
  name=$1; shift
  if [ -f "$R/$name/run.out" ] && tail -1 "$R/$name/run.out" | grep -q "^exit 0"; then return; fi
  "$DECK/run_mms.sh" "$R/$name" "$@" > /dev/null 2>&1
  echo "$name $(tail -1 "$R/$name/run.out")"
}
export -f run_one; export R DECK
xargs -P "$2" -L 1 bash -c 'run_one "$@"' _ < "$1"
echo ALL_DONE
