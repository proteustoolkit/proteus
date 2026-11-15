"""The standalone test of proteus/UnfittedKernels.h (unfitted_kernels_test.cpp).

The kernels include only <cmath> and the test only the kernels, so this compiles them with the system C++ compiler and
runs the binary: no proteus build, no PETSc, no MPI. For each kernel the Jacobian matches a finite difference of its
residual (Galerkin and Petrov-Galerkin, 3, 4, 6 and 10 local DOFs) and the penalty block is symmetric positive
semidefinite.
"""
import os
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
INCLUDE = os.path.join(HERE, os.pardir, os.pardir, "proteus")


def _compiler():
    for name in (os.environ.get("CXX", "").split() or [None])[:1] + ["c++", "g++", "clang++"]:
        if name and shutil.which(name):
            return shutil.which(name)
    return None


@pytest.mark.skipif(_compiler() is None, reason="no C++ compiler")
def test_unfitted_kernels_standalone(tmp_path):
    exe = str(tmp_path / "unfitted_kernels_test")
    cmd = [_compiler(), "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-I" + INCLUDE,
           os.path.join(HERE, "unfitted_kernels_test.cpp"), "-o", exe]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    run = subprocess.run([exe], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stdout[-3000:]
    assert "ALL PASSED" in run.stdout
