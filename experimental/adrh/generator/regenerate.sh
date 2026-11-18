#!/bin/sh
# ymf spec -> generator inputs -> generated.h.  usage: sh regenerate.sh <ymf checkout>   (inside an env with sympy; PYTHONPATH gets <ymf>/src)
# The inputs are the sub-problems of the dam break (2D) and of the porous dam break (2D and 3D); the kernels listed are the ones
# the engine takes (the others, `flow`, are refused by the generator and tested as refusals).
set -e
cd "$(dirname "$0")"
YMF=${1:?path to a ymf checkout}
rm -f inputs/*.json
PYTHONPATH="$YMF/src:." python make_inputs.py "$YMF/tests/fixtures/two_phase"
python generate.py ../generated.h inputs/vof_2d.json inputs/ncls_2d.json inputs/rdls_2d.json inputs/mcorr_2d.json \
  inputs/porous_vof_2d.json inputs/porous_vof_3d.json inputs/porous_ncls_2d.json inputs/porous_ncls_3d.json \
  inputs/porous_mcorr_2d.json inputs/porous_mcorr_3d.json inputs/porous_rdls_3d.json \
  inputs/transport_2d.json inputs/transport_lagged_2d.json inputs/transport_step_2d.json \
  inputs/reacdiff_2d.json inputs/rdls_nosc_2d.json
