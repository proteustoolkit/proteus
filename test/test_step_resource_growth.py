"""A two-phase run on 2 ranks must not grow its MPI requests, communicators or ghosted vectors from step to step.

Main f392a159 aborted the Torino flume run (RANS2P, VOF, NCLS, RDLS, MCorr; 3.86M nodes on 1,920 Cray MPICH ranks)
after exactly 12,678 steps, twice, with MPICH's ``req != NULL`` assertion in VecGhostUpdateBegin: the request pool
was exhausted. The mprans scatterDOFArray of that commit cached a ghosted vector (a PetscSF with a persistent send and
receive per neighbour rank) per id(array) over the array itself, so every temporary it was handed stayed alive with its
requests; VOF.FCTStep hands it three per step. 76f3e125 caches one vector per array shape instead.

Measured with step_resource_growth_mpi.py's per-step record (live MPICH request handles, macOS arm64, MPICH 5.0.1),
2 ranks, damBreakLevelSet he=0.05 to t=0.3, steps 10 to 49:
  main f392a159                84 -> 318 live requests (6.0 per step per rank)
  main c09280cb (76f3e125)     32 -> 32
  1.9.x b6c60a5d               28 -> 28 (no scatterDOFArray)
The same with only f392a159's scatterDOFArray put back on c09280cb (the LEAKY=1 control below): 86 -> 320.

The record starts after WARMUP steps, by which every cache and solver object has been built, and the archive is
written every few steps, so a leak per archive write shows too. On an MPI other than MPICH, or an MPICH whose pools
cannot be read and checked, only the ghosted vectors are counted.
"""
import os
import shutil
import subprocess
import sys

import numpy
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.join(HERE, "step_resource_growth_mpi.py")
WARMUP = 5


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ("mpiexec", "mpirun"):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which("mpiexec") or shutil.which("mpirun")


MPIEXEC = _mpiexec()


def _growth(cwd, **overrides):
    """Per rank, the record of step_resource_growth_mpi.py: columns step, t, req, comm, parvec."""
    os.makedirs(str(cwd), exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1")
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env["PETSC_OPTIONS"] = (env.get("PETSC_OPTIONS", "") + " -mat_partitioning_type current").strip()
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env.update({k: str(v) for k, v in overrides.items()})
    out = subprocess.run([MPIEXEC, "-n", "2", sys.executable, HELPER], capture_output=True, text=True, timeout=1200,
                         env=env, cwd=str(cwd))
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    records = [numpy.loadtxt(os.path.join(str(cwd), "growth.%d.csv" % rank), delimiter=",", skiprows=1, ndmin=2)
               for rank in range(2)]
    for r in records:
        assert r.shape[0] >= WARMUP + 15, "too few steps (%d) to see a growth per step" % r.shape[0]
    return records


def _increase(records, column):
    """Largest increase over the ranks of a column between the end of the warm-up and the last step."""
    return max(int(r[-1, column] - r[WARMUP - 1, column]) for r in records)


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
def test_no_growth_per_step(tmp_path):
    records = _growth(tmp_path)
    steps = records[0].shape[0] - WARMUP
    assert _increase(records, 4) == 0, ("ghosted vectors grew over %d steps" % steps, [r[:, 4] for r in records])
    if records[0][0, 2] >= 0:
        assert _increase(records, 2) == 0, ("MPI requests grew over %d steps" % steps, [r[:, 2] for r in records])
    if records[0][0, 3] >= 0:
        assert _increase(records, 3) == 0, ("communicators grew over %d steps" % steps, [r[:, 3] for r in records])


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
def test_control_f392a159_scatter_grows_every_step(tmp_path):
    """The control: f392a159's scatterDOFArray keeps a ghosted vector, with its requests, per temporary."""
    records = _growth(tmp_path, LEAKY=1)
    steps = records[0].shape[0] - WARMUP
    assert _increase(records, 4) >= 3 * steps, [r[:, 4] for r in records]
    if records[0][0, 2] >= 0:
        assert _increase(records, 2) >= 6 * steps, [r[:, 2] for r in records]
