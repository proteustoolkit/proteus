"""A DG space coupled by a numerical flux refuses a partition that leaves face neighbours of owned elements off the rank.

Such a face is a rank interface (the partitioner flags it -1) and was assembled as domain boundary, so the answer
depended on the partition: 16x16 DG Poisson through adr_advection_partition_mpi.py gives 1.04e-3 serially, 6.3e-2 and
9.4e-2 on 2 and 3 ranks with an element partition without overlap, 1.3e-1 and 2.0e-1 with a node partition and one
layer. Every rank must raise (a rank that carried on would wait in a collective). ADRProblem's own choice for DG, an
element partition with one layer, and CG on any partition run as before.
"""
import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.join(HERE, "adr_advection_partition_mpi.py")


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ("mpiexec", "mpirun"):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which("mpiexec") or shutil.which("mpirun")


MPIEXEC = _mpiexec()


def _run(ranks, cwd, **overrides):
    env = dict(os.environ, OMP_NUM_THREADS="1", PETSC_OPTIONS="-mat_partitioning_type current", BDIR="y", CELLS="16")
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env.update({k: str(v) for k, v in overrides.items()})
    return subprocess.run([MPIEXEC, "-n", str(ranks), sys.executable, HELPER], capture_output=True, text=True,
                          timeout=600, env=env, cwd=str(cwd))


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("part,layers", [("element", 0), ("node", 1)])
def test_dg_on_a_partition_without_face_neighbours_is_refused_on_every_rank(tmp_path, part, layers):
    out = _run(2, tmp_path, SPACE="DG", PART_TYPE=part, NLAYERS=layers)
    output = out.stdout + out.stderr
    assert out.returncode != 0, output
    assert output.count("faces of owned elements are rank interfaces") >= 2, output


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("space,part,layers", [("DG", "element", 1), ("CG", "element", 0), ("CG", "node", 1)])
def test_partitions_that_work_are_not_refused(tmp_path, space, part, layers):
    out = _run(2, tmp_path, SPACE=space, PART_TYPE=part, NLAYERS=layers)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    assert "RESULT" in out.stdout
