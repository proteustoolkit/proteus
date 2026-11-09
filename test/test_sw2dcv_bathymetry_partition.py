"""A bathymetry array in the input mesh's node order reaches the right nodes on every partition path.

SW2DCV and GN_SW2DCV take a bathymetry array in the order of the input mesh's nodes and, on more than one rank,
permute it to each subdomain. They read the permutation from mappings.h5, which only the ``parun -F`` partitioners
wrote: a run that partitioned the mesh in memory read whatever mappings.h5 an earlier -F run had left in its
directory (a wrong bathymetry, silently) or failed without one. Now both partition paths keep the permutation on
the mesh (nodeNumbering_global_old2new) and the models read it there. A stale mappings.h5 is planted in every run
directory, a reversed permutation, and must make no difference. obstacle_flow on its stored Triangle mesh
(149 nodes), b = 0.01 x + 0.02 y + 0.001 i with i the file node; every node of every subdomain, ghosts included.
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
HELPER = os.path.join(HERE, "sw2dcv_bathymetry_mpi.py")
NNODES = 149


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ("mpiexec", "mpirun"):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which("mpiexec") or shutil.which("mpirun")


MPIEXEC = _mpiexec()


def _bathymetry(ranks, cwd, **overrides):
    h5py = pytest.importorskip("h5py")
    with h5py.File(os.path.join(str(cwd), "mappings.h5"), "w") as f:
        f["nodeNumbering_old2new"] = numpy.arange(NNODES, dtype="i")[::-1]
    env = dict(os.environ, OMP_NUM_THREADS="1")
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env["PETSC_OPTIONS"] = (env.get("PETSC_OPTIONS", "") + " -mat_partitioning_type current").strip()
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env.update({k: str(v) for k, v in overrides.items()})
    launch_output.run([MPIEXEC, "-n", str(ranks), sys.executable, HELPER], cwd=cwd, env=env, timeout=600)
    calls = {open(os.path.join(str(cwd), "calls_rank%d.txt" % r)).read() for r in range(ranks)}
    assert calls == {"partitionNodesFromTriangleFiles" if overrides.get("FILES") else "partitionNodes"}, calls
    return [numpy.loadtxt(os.path.join(str(cwd), "b_rank%d.csv" % r), delimiter=",", skiprows=1)
            for r in range(ranks)]


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("model", ["SW2DCV", "GN_SW2DCV"])
@pytest.mark.parametrize("ranks", [2, 4])
def test_bathymetry_array_is_permuted_to_the_subdomain_nodes(tmp_path, model, ranks):
    nodes = {}
    for path, files in (("in-memory", ""), ("files", "1")):
        d = tmp_path / path
        d.mkdir()
        overrides = {"FILES": files} if files else {}
        if model == "GN_SW2DCV":
            overrides["GN"] = "1"
        seen = set()
        for a in _bathymetry(ranks, d, **overrides):
            assert numpy.array_equal(a[:, 3], a[:, 4]), (path, "a node took another node's bathymetry")
            seen.update(a[:, 2].astype(int).tolist())
            for x, y, i, b, _ in a:
                nodes.setdefault(int(i), set()).add(b)
        assert seen == set(range(NNODES)), path
    #-F and in-memory, every copy of a node: one value
    assert all(len(v) == 1 for v in nodes.values())
