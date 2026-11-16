#!/bin/sh
# Build the prototype module next to this script (not part of the proteus build).
# usage: sh build.sh   (inside the proteus conda environment)
set -e
cd "$(dirname "$0")"
EXT=$(python -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))")
${CXX:-c++} -O3 -std=c++17 -shared -fPIC $([ "$(uname)" = Darwin ] && echo -undefined dynamic_lookup) \
  $(python -m pybind11 --includes) module.cpp -o adrh_proto$EXT
# the generator spike's module: the same kernels with the functors generated from ymf (see generator/)
[ "$1" = gen ] && ${CXX:-c++} -O3 -std=c++17 -DADRH_GENERATED -shared -fPIC $([ "$(uname)" = Darwin ] && echo -undefined dynamic_lookup) \
  $(python -m pybind11 --includes) module.cpp -o adrh_gen$EXT
exit 0
