#!/usr/bin/env bash
# Cancelled vs conservative transport form with the velocity-based coupling,
# UHD level 0, Co 0.5, transport FCT off (low order), serial, 2026-11-05.
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; S="$D/../run_level0.sh"
export SIMMONS_FCT=0
TADR_VELOCITY_COUPLING=lagged TADR_CONSERVATIVE_FORM=1 "$S" "$D/conservative_lagged" > "$D/conservative_lagged.log" 2>&1 &
TADR_VELOCITY_COUPLING=lagged                          "$S" "$D/cancelled_lagged"    > "$D/cancelled_lagged.log" 2>&1 &
TADR_VELOCITY_COUPLING=eos    TADR_CONSERVATIVE_FORM=1 "$S" "$D/conservative_eos"    > "$D/conservative_eos.log" 2>&1 &
TADR_VELOCITY_COUPLING=eos                             "$S" "$D/cancelled_eos"       > "$D/cancelled_eos.log" 2>&1 &
wait
echo "all done $(date)"
