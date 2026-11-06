#!/usr/bin/env bash
# Coupling ablation, UHD level 0 (Table tab:uhd_coupling), 2026-11-05 rerun on the
# current decks: five serial runs side by side, then the table.
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; S="$D/../run_level0.sh"
SIMMONS_VELOCITY_COUPLING=eos                            "$S" "$D/velocity_co0.5"  > "$D/velocity_co0.5.log" 2>&1 &
SIMMONS_VELOCITY_COUPLING=eos SIMMONS_COURANT=0.25       "$S" "$D/velocity_co0.25" > "$D/velocity_co0.25.log" 2>&1 &
SIMMONS_VELOCITY_COUPLING=lagged                         "$S" "$D/velocity_lagged_density" > "$D/velocity_lagged_density.log" 2>&1 &
SIMMONS_VELOCITY_COUPLING=lagged SIMMONS_RICHARDS_FCT=0  "$S" "$D/velocity_lagged_density_flowfct_off" > "$D/velocity_lagged_density_flowfct_off.log" 2>&1 &
                                                      "$S" "$D/compatible"      > "$D/compatible.log" 2>&1 &
wait
echo "all done $(date)"
