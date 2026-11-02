"""TwoPhaseFlow damBreak on 2 ranks gives the serial answer once the solves are converged.

The 2-rank drift reported with proteustoolkit/proteus#1288 (1e-4 to 1e-3 relative by t=0.1, CLSVOF and RANS3PF) is the
deck's nonlinear tolerance (nl_atol_res ~1e-3): serial and parallel iterations stop at different places. Measured on
main f392a159 with the partition pinned: the largest relative difference over the models is 9.5e-4 at the deck's
tolerances, 4e-7 at atol 1e-6, 6e-10 at 1e-9 and 5e-13 at 1e-12, on 2 and 3 ranks alike. The default-tolerance run is
the control: it shows the comparison sees a difference of that size.
"""
import os
import shutil
import subprocess
import sys

import numpy
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.join(HERE, "dambreak_partition_mpi.py")


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ("mpiexec", "mpirun"):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which("mpiexec") or shutil.which("mpirun")


MPIEXEC = _mpiexec()


def _field(ranks, cwd, **overrides):
    env = dict(os.environ, OMP_NUM_THREADS="1")
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env["PETSC_OPTIONS"] = (env.get("PETSC_OPTIONS", "") + " -mat_partitioning_type current").strip()
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env.update({k: str(v) for k, v in overrides.items()})
    argv = [sys.executable, HELPER] if ranks == 1 else [MPIEXEC, "-n", str(ranks), sys.executable, HELPER]
    out = subprocess.run(argv, capture_output=True, text=True, timeout=1800, env=env, cwd=str(cwd))
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    return numpy.loadtxt(os.path.join(str(cwd), "u.csv"), delimiter=",", skiprows=1)


def _relative_difference(a, b):
    assert a.shape == b.shape and numpy.abs(a[:, :3] - b[:, :3]).max() < 1e-10
    worst = 0.0
    for key in numpy.unique(a[:, 0]):
        k = a[:, 0] == key
        worst = max(worst, numpy.abs(a[k, 3] - b[k, 3]).max() / (numpy.abs(a[k, 3]).max() + 1e-300))
    return worst


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
def test_dambreak_on_two_ranks_is_the_serial_answer_when_converged(tmp_path):
    for d in ("1", "2", "3", "4"):
        (tmp_path / d).mkdir()
    serial = _field(1, tmp_path / "1", TIGHT=1)
    assert _relative_difference(serial, _field(2, tmp_path / "2", TIGHT=1)) < 1e-10
    # control: at the deck's own tolerances the runs differ by about the tolerance
    assert _relative_difference(_field(1, tmp_path / "3"), _field(2, tmp_path / "4")) > 1e-6
