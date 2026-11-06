# The ADRH prototype modules and their tests. build.sh and build_kokkos.sh remain the two-line way to
# build one module; this is the same build as a CMake project, for CI and for device (CUDA/HIP) Kokkos.
#
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release     (a Python with pybind11 and numpy on PATH;
#   cmake --build build -j                              Kokkos found through CMAKE_PREFIX_PATH or $CONDA_PREFIX)
#   ctest --test-dir build --output-on-failure
#
# adrh_proto (host kernels, C++17) always builds; adrh_kokkos and bench_kokkos build when Kokkos is found
# (C++20, whichever backends that Kokkos was configured with). The modules are written next to the sources,
# where the Python scripts look for them first. A CUDA Kokkos needs its compiler: configure with
# -DCMAKE_CXX_COMPILER=<kokkos>/bin/nvcc_wrapper (ADRH_ENGINE_DESIGN.md section 19 has the recipe).
cmake_minimum_required(VERSION 3.20)
project(adrh LANGUAGES CXX)

option(ADRH_KOKKOS "build the Kokkos module and benchmark when Kokkos is found" ON)
set(ADRH_EXPECT_SPACE "" CACHE STRING "if set (Cuda, HIP, OpenMP), a test fails unless adrh_kokkos's default execution space is this one")
option(ADRH_BITWISE "no FMA contraction, so CPU and GPU kernels without libm calls give the same bits" ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python 3 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
                OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE pybind11_status)
if(NOT pybind11_status EQUAL 0)
  message(FATAL_ERROR "pybind11 is not importable from ${Python_EXECUTABLE}")
endif()
find_package(pybind11 CONFIG REQUIRED)

set(Kokkos_FOUND OFF)
if(ADRH_KOKKOS)
  if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
  endif()
  find_package(Kokkos QUIET CONFIG)
endif()

function(adrh_module target source)
  pybind11_add_module(${target} ${source})
  set_target_properties(${target} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}")
  if(ADRH_BITWISE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE -ffp-contract=off)
  endif()
endfunction()

adrh_module(adrh_proto module.cpp)
target_compile_features(adrh_proto PRIVATE cxx_std_17)

set(tests_kokkos OFF)
if(Kokkos_FOUND)
  message(STATUS "Kokkos ${Kokkos_VERSION}: building adrh_kokkos and bench_kokkos")
  adrh_module(adrh_kokkos module_kokkos.cpp)
  target_compile_features(adrh_kokkos PRIVATE cxx_std_20)
  target_link_libraries(adrh_kokkos PRIVATE Kokkos::kokkos)
  add_executable(bench_kokkos bench_kokkos.cpp)
  target_compile_features(bench_kokkos PRIVATE cxx_std_20)
  target_link_libraries(bench_kokkos PRIVATE Kokkos::kokkos)
  if(ADRH_BITWISE AND Kokkos_ENABLE_CUDA)
    target_compile_options(adrh_kokkos PRIVATE $<$<COMPILE_LANGUAGE:CXX>:--fmad=false>)
    target_compile_options(bench_kokkos PRIVATE $<$<COMPILE_LANGUAGE:CXX>:--fmad=false>)
  endif()
  set(tests_kokkos ON)
else()
  message(STATUS "Kokkos not found: building adrh_proto only")
endif()

# The tests need numpy and scipy, and proteus.Quadrature from this source tree (it needs no compiled extension).
enable_testing()
get_filename_component(PROTEUS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

function(adrh_test name module script)
  add_test(NAME ${name}
           COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/${script}" ${ARGN}
           WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
  # idle Kokkos threads would slow the rest of a mixed run (see README.md); two are enough to exercise the coloring
  set_tests_properties(${name} PROPERTIES
    ENVIRONMENT "ADRH_MODULE=${module};PYTHONPATH=${PROTEUS_ROOT};KOKKOS_NUM_THREADS=2;OMP_NUM_THREADS=2;OMP_WAIT_POLICY=passive"
    TIMEOUT 1800)
endfunction()

foreach(script stabilization_test.py upwind_test.py richards_layers.py richards_nd_test.py)
  get_filename_component(stem ${script} NAME_WE)
  adrh_test(proto_${stem} adrh_proto ${script})
  if(tests_kokkos)
    adrh_test(kokkos_${stem} adrh_kokkos ${script})
  endif()
endforeach()
# VOF against proteus's VOF kernel on the damBreak case (vof_damBreak_he0.05.npz, written by vof_capture.py); host module only so far
adrh_test(proto_vof_test adrh_proto vof_test.py)
# NCLS against proteus's NCLS kernel on the same case (ncls_damBreak_he0.05.npz, vof_capture.py model=ncls)
adrh_test(proto_ncls_test adrh_proto ncls_test.py)
if(tests_kokkos)
  # the Kokkos kernels reproduce the host kernels' bits (golden.py's inputs are fixed)
  adrh_test(kokkos_matches_host adrh_proto golden.py against adrh_kokkos)
endif()
if(tests_kokkos AND ADRH_EXPECT_SPACE)
  # a device job that quietly ran on the host would pass every other test
  add_test(NAME kokkos_default_space
           COMMAND "${Python_EXECUTABLE}" -c
           "import sys, adrh_kokkos as a; s = a.kokkos_info()['space']; print('default execution space', s); sys.exit(0 if s == '${ADRH_EXPECT_SPACE}' else 1)"
           WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
  set_tests_properties(kokkos_default_space PROPERTIES ENVIRONMENT "KOKKOS_NUM_THREADS=2;OMP_WAIT_POLICY=passive")
endif()
