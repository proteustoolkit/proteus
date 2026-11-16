#!/bin/sh
# ymf plain data -> generated.h.  usage: sh regenerate.sh <ymf checkout>   (inside an env with sympy; PYTHONPATH gets <ymf>/src)
set -e
cd "$(dirname "$0")"
YMF=${1:?path to a ymf checkout}
PYTHONPATH="$YMF/src" python make_inputs.py "$YMF/tests/fixtures/two_phase/dambreak.ymf"
python generate.py ../generated.h inputs/vof_deck_2d.json inputs/vof_1d.json inputs/vof_2d.json inputs/vof_3d.json \
  inputs/ncls_1d.json inputs/ncls_2d.json inputs/ncls_3d.json
