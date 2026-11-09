"""The subdomain mesh a model holds reports the global h and hMin.

buildFromC computed them over each rank's own elements, so every epsFact * mesh.h was a different number on each
rank. For the NCLS disk with the Coupez term (sign() and the Coupez strength scale with epsFactRedistancing * mesh.h)
rank 1 had h = 0.035294 against 0.035538 serially, and 2 ranks differed from serial by 5.1e-7 at t=0.08 (3 ranks
5.1e-7). With the sizes reduced over the ranks: 4.2e-11 and 1.6e-11. The control keeps each rank's own sizes.
"""
import importlib.util
import os
import shutil
import sys

import numpy
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
# a failed launch keeps its whole output (loaded by path: --import-mode=importlib)
_spec = importlib.util.spec_from_file_location("launch_output", os.path.join(HERE, "launch_output.py"))
launch_output = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launch_output)
HELPER = os.path.join(HERE, "subdomain_mesh_size_mpi.py")


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ("mpiexec", "mpirun"):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which("mpiexec") or shutil.which("mpirun")


MPIEXEC = _mpiexec()


def _run(ranks, cwd, **overrides):
    env = dict(os.environ, OMP_NUM_THREADS="1")
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env["PETSC_OPTIONS"] = "-mat_partitioning_type current -ncls_ksp_rtol 1e-14 -ncls_ksp_atol 1e-40 -ncls_ksp_max_it 5000"
    env.update({k: str(v) for k, v in overrides.items()})
    argv = [sys.executable, HELPER] if ranks == 1 else [MPIEXEC, "-n", str(ranks), sys.executable, HELPER]
    out = launch_output.run(argv, cwd=cwd, env=env, timeout=1800)
    u = numpy.loadtxt(os.path.join(str(cwd), "u.csv"), delimiter=",", skiprows=1)
    sizes = numpy.atleast_2d(numpy.loadtxt(os.path.join(str(cwd), "h.csv"), delimiter=",", skiprows=1))
    return u, sizes


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
def test_every_rank_holds_the_global_mesh_size_and_coupez_gives_the_serial_answer(tmp_path):
    for d in ("1", "2", "3"):
        (tmp_path / d).mkdir()
    serial, serial_sizes = _run(1, tmp_path / "1")
    parallel, sizes = _run(2, tmp_path / "2")
    assert (sizes == serial_sizes).all(), (sizes, serial_sizes)
    assert numpy.abs(serial - parallel).max() < 1e-9
    control, local_sizes = _run(2, tmp_path / "3", LOCALH=1)
    assert (local_sizes != serial_sizes).any()
    assert numpy.abs(serial - control).max() > 1e-7
