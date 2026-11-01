"""CLSVOF's start from a discontinuous initial condition on two ranks.

TwoPhaseFlow/damBreak.py starts CLSVOF from a characteristic function
(``disc_ICs``): a lumped L2 projection, then redistancing in stages that freeze
the nodes of the cells that hold the interface, and then the nodes that do not.
Two things in that spin-up depended on the partition:

* the flag that marks the interface nodes is set by the cells a rank can see, so
  a ghost node at the edge of the overlap could be flagged on its owner and not
  on its neighbour, and the freeze and its inverse differed (3.6e-3 in the level
  set after the first interval);
* the penalty ``alpha / h_min`` took the smallest element of the rank's own piece
  of the mesh rather than of the whole mesh (3e-7 more, on an unstructured mesh).

The case is copied and tightened: the stored mesh is kept, every model gets
nl_atol_res 1e-10 and a tight linear tolerance, and only the first interval
(0.001) is run, which is the spin-up and one step.
"""
import os
import shutil
import subprocess
import sys

import h5py
import numpy
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.join(HERE, "TwoPhaseFlow")

TIGHTEN = '''
[(setattr(mm.n, 'nl_atol_res', 1.0e-10), setattr(mm.n, 'l_atol_res', 1.0e-12),
  (setattr(mm.n, 'linTolFac', 1.0e-8) if hasattr(mm.n, 'linTolFac') else None))
 for mm in myTpFlowProblem.SystemPhysics.modelDict.values()]
'''


def _tool(name):
    candidate = os.path.join(os.path.dirname(sys.executable), name)
    return candidate if os.path.exists(candidate) else shutil.which(name)


PARUN = _tool("parun")
MPIEXEC = _tool("mpiexec") or _tool("mpirun")


def _copy(dest):
    os.makedirs(dest)
    for name in os.listdir(CASE):
        if name in ("damBreak.py", "__init__.py") or name.startswith("meshDambreak."):
            shutil.copy(os.path.join(CASE, name), dest)
    with open(os.path.join(dest, "damBreak.py"), "a") as f:
        f.write(TIGHTEN)


def _run(ranks, cwd):
    env = dict(os.environ, OMP_NUM_THREADS="1",
               PETSC_OPTIONS="-mat_partitioning_type current")
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    argv = [PARUN, "--TwoPhaseFlow", "damBreak.py", "-l1", "-C",
            "final_time=0.001 dt_output=0.001 he=0.1"]
    if MPIEXEC:  # even for one rank (test_archive_parallel_metadata.run_ranks says why)
        argv = [MPIEXEC, "-n", str(ranks)] + argv
    out = subprocess.run(argv, capture_output=True, text=True, timeout=900, env=env,
                         cwd=str(cwd))
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]


def _fields(cwd):
    with h5py.File(os.path.join(cwd, "damBreak.h5"), "r") as h:
        nodes = sorted(k for k in h.keys() if k.startswith("nodesSpatial_Domain"))[-1]
        x = numpy.array(h[nodes])[:, :2]
        order = numpy.lexsort((numpy.round(x[:, 1], 8), numpy.round(x[:, 0], 8)))
        return x[order], {name: numpy.array(h[name + "_t1"]).ravel()[order]
                          for name in ("phi", "vof")}


@pytest.mark.skipif(PARUN is None or MPIEXEC is None, reason="parun or mpiexec not available")
def test_clsvof_started_from_a_discontinuous_condition_matches_serial_on_two_ranks(tmp_path):
    serial, parallel = str(tmp_path / "serial"), str(tmp_path / "parallel")
    _copy(serial)
    _copy(parallel)
    _run(1, serial)
    _run(2, parallel)
    xs, a = _fields(serial)
    xp, b = _fields(parallel)
    numpy.testing.assert_allclose(xs, xp)
    assert numpy.abs(a["phi"]).max() > 1.0       # a distance function, not zeros
    for name in ("phi", "vof"):
        numpy.testing.assert_allclose(b[name], a[name], atol=1e-8, err_msg=name)
