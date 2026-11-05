"""Per-rank subdomain views as the same-rank restart source (C7.1 Phase D, decided by cekees 2026-11-05).

A per-rank archive (archive-modes layouts B and C) holds, in each rank's file, that rank's subdomain view:
the step-H ymf grid plus what rebuilding the solver's subdomain needs (each element's boundary numbers,
the edges, the flags, the partition's eight maps, the global counts). A hot start with the same number
of ranks builds the mesh from the views, with no global mesh and no partitioner
(MeshTools.generateMesh(restartArchive=...)); anything else -- another number of ranks, no views, a
multilevel hierarchy -- takes the old path and says why.

The drivers start their own ranks (subdomain_view_restart_mpi.py, restart_views_vortex.py), so these
tests skip under a multi-rank launch.
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from proteus import MeshTools

h5py = pytest.importorskip("h5py")
pytest.importorskip("ymf.mesh")

HERE = os.path.dirname(os.path.abspath(__file__))
SQUARE = os.path.join(HERE, "ymf_meshes", "square4x4_tagged")
CUBE = os.path.join(HERE, "ymf_meshes", "cube_3x3x3")


def _launcher_env(partitioner=True):
    env = dict((k, v) for k, v in os.environ.items() if not k.startswith(("PMI_", "PMIX_", "HYDRA_", "OMPI_COMM_WORLD")))
    env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env["OMP_NUM_THREADS"] = "1"
    # the graph partitioner is not reproducible from run to run; the restart compared against must be
    # partitioned as the run it restarts was
    env["PETSC_OPTIONS"] = "-mat_partitioning_type current"
    return env


def _mpiexec():
    for name in ("PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "PMIX_SIZE"):
        if int(os.environ.get(name, "1")) > 1:
            pytest.skip("launches its own mpiexec; run this file serially")
    beside = os.path.join(os.path.dirname(sys.executable), "mpiexec")
    found = beside if os.path.exists(beside) else shutil.which("mpiexec")
    if found is None:
        pytest.skip("no MPI launcher available")
    return found


def _run(n_ranks, script, *args, cwd):
    result = subprocess.run([_mpiexec(), "-n", str(n_ranks), sys.executable, os.path.join(HERE, script)]
                            + [str(a) for a in args], capture_output=True, text=True, env=_launcher_env(),
                            cwd=str(cwd), timeout=600)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return result.stdout


@pytest.mark.parametrize("prefix, kind, n_ranks, document", [
    (SQUARE, "scatter", 3, "global"), (SQUARE, "reversed", 2, "per-rank"),
    (CUBE, "reversed", 2, "global"), (CUBE, "scatter", 3, "per-rank")],
    ids=["tri-3-B", "tri-2-C", "tet-2-B", "tet-3-C"])
def test_a_view_rebuilds_the_partitioned_mesh_exactly(prefix, kind, n_ranks, document, tmp_path):
    """Every array of every rank's subdomain (33, element boundary and edge numbering included), the
    eight maps, the global counts and scalars, the -1 marking: as partitionMesh built them."""
    out = _run(n_ranks, "subdomain_view_restart_mpi.py", prefix, kind, document, tmp_path, cwd=tmp_path)
    lines = sorted(l for l in out.splitlines() if l.startswith("rank="))
    assert len(lines) == n_ranks, out
    for line in lines:
        fields = dict(w.split("=", 1) for w in line.split())
        assert int(fields["arrays"]) >= 30, line
        assert fields["differ"] == "-" and fields["maps"] == "-" and fields["globals"] == "-", line
        assert fields["marked"] == "True/True", line
    assert sum(int(dict(w.split("=", 1) for w in l.split())["cut"]) for l in lines) > 0


def _datasets(path):
    found = {}
    with h5py.File(path, "r") as f:
        f.visititems(lambda name, obj: found.__setitem__(name, obj[()]) if isinstance(obj, h5py.Dataset) else None)
    return found


@pytest.mark.parametrize("grid, document", [("global", "per-rank"), ("generated", "global")],
                         ids=["partitioned-C", "generated-B"])
def test_a_same_rank_restart_from_the_views_is_the_global_restart_bit_for_bit(grid, document, tmp_path):
    """vortex2D on 2 ranks, stopped at step 3 and hot started to step 7 twice from the same archive: the
    mesh from the views, and generated and partitioned. Every dataset of both ranks' files is equal."""
    stop = tmp_path / "stop"
    _run(2, "restart_views_vortex.py", stop, "stop", 3, "views", grid, document, cwd=tmp_path)
    runs = {}
    for mesh_from in ("views", "global"):
        shutil.copytree(stop, tmp_path / mesh_from)
        out = _run(2, "restart_views_vortex.py", tmp_path / mesh_from, "hot", 3, mesh_from, grid, document,
                   cwd=tmp_path)
        assert ("views=1" if mesh_from == "views" else "views=0") in out, out
        runs[mesh_from] = [_datasets(tmp_path / mesh_from / ("views_p%d.h5" % r)) for r in range(2)]
    for mine, theirs in zip(runs["views"], runs["global"]):
        assert set(mine) == set(theirs)
        assert any(name.startswith("vof_p") for name in mine)
        assert [n for n in theirs if not np.array_equal(mine[n], theirs[n])] == []


def test_another_number_of_ranks_takes_the_old_path_and_says_why(tmp_path):
    _run(2, "subdomain_view_restart_mpi.py", SQUARE, "reversed", "per-rank", tmp_path, cwd=tmp_path)
    from proteus import Comm
    Comm.init()
    with pytest.raises(MeshTools.SubdomainViewUnavailable, match="written by rank 0 of 2, read by rank 0 of 1"):
        MeshTools.subdomainViewMesh(str(tmp_path / "mesh"))
    assert MeshTools._meshFromRestartArchive(str(tmp_path / "mesh"), MeshTools.MeshOptions(nd=2)) is None
    with pytest.raises(MeshTools.SubdomainViewUnavailable, match="no per-rank file"):
        MeshTools.subdomainViewMesh(str(tmp_path / "missing"))


def test_a_multilevel_hierarchy_takes_the_old_path(tmp_path):
    options = MeshTools.MeshOptions(nd=2)
    options.nLevels = 2
    assert MeshTools._meshFromRestartArchive(str(tmp_path / "anything"), options) is None
