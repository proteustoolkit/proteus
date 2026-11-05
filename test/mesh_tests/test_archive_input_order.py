"""An archived mesh read back in the input mesh's order, whatever the rank count (C7.1 step I).

The partitioner renumbers the global mesh so each rank owns a contiguous range, and the archive writes in
that numbering. When the renumbering is not the identity, the mesh grid carries ``NodeMapInput`` and
``CellMapInput`` (each archived row's index in the input mesh), so ``readMeshFromArchive(...,
inputOrder=True)`` gives the input mesh back and the fingerprint of the mesh does not depend on how many
ranks wrote it. With contiguous owners the partitioner keeps the input order and no map is written: the
archive is what it was before.
"""
import os
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pytest

from proteus import Archiver, MeshTools

HERE = os.path.dirname(os.path.abspath(__file__))
SQUARE = os.path.join(HERE, "ymf_meshes", "square4x4_tagged")
CUBE = os.path.join(HERE, "ymf_meshes", "cube_3x3x3")


def _launcher_env():
    """The environment for a child run: this one's, without a parent mpiexec's process-manager variables
    (a child mpiexec, or a plain python child, would otherwise try to join the parent's job)."""
    env = dict((k, v) for k, v in os.environ.items() if not k.startswith(("PMI_", "PMIX_", "HYDRA_", "OMPI_COMM_WORLD")))
    env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env.pop("PETSC_OPTIONS", None)                 # owners are prescribed: no MatPartitioning runs
    return env


def _skip_under_mpiexec():
    """These tests start their own ranks; under ``mpiexec -n N`` (N > 1) every rank would."""
    for name in ("PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "PMIX_SIZE"):
        if int(os.environ.get(name, "1")) > 1:
            pytest.skip("launches its own mpiexec; run this file serially")


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ("mpiexec", "mpirun"):
        if os.path.exists(os.path.join(bindir, name)):
            return os.path.join(bindir, name)
    return shutil.which("mpiexec") or shutil.which("mpirun")


def _archive(prefix, n_ranks, owners, sync, outdir):
    """Run the driver on n_ranks; returns (fingerprint, renumbered) as rank 0 printed them."""
    os.makedirs(outdir, exist_ok=True)
    _skip_under_mpiexec()
    env = _launcher_env()
    cmd = [sys.executable, os.path.join(HERE, "archive_input_order_mpi.py"), prefix, owners, sync, str(outdir)]
    if n_ranks > 1:
        mpiexec = _mpiexec()
        if mpiexec is None:
            pytest.skip("no MPI launcher available")
        cmd = [mpiexec, "-n", str(n_ranks)] + cmd
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(outdir), timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    line = next(l for l in result.stdout.splitlines() if l.startswith("fingerprint="))
    fingerprint, renumbered = (w.split("=")[1] for w in line.split())
    return fingerprint, renumbered == "1"


def _input(prefix):
    info = MeshTools.readMeshYmf(prefix)
    return info, MeshTools.meshFingerprint(info.nodeArray, info.elementNodesArray)


@pytest.mark.parametrize("owners", ["scatter", "reversed"])
@pytest.mark.parametrize("prefix", [SQUARE, CUBE], ids=["triangles", "tetrahedra"])
def test_a_global_archive_reads_back_in_the_input_order_from_any_rank_count(prefix, owners, tmp_path):
    info, expected = _input(prefix)
    for n_ranks in (1, 2, 3):
        outdir = tmp_path / ("n%d" % n_ranks)
        fingerprint, renumbered = _archive(prefix, n_ranks, owners, "global", outdir)
        assert fingerprint == expected, n_ranks
        assert renumbered == (n_ranks > 1)
        archived = MeshTools.readMeshFromArchive(str(outdir / "mesh"))
        canonical = MeshTools.readMeshFromArchive(str(outdir / "mesh"), inputOrder=True)
        if n_ranks > 1:                            # the rows really were permuted
            assert not np.array_equal(archived.nodeArray, info.nodeArray)
            assert archived.nodeNumbering_input is not None
        np.testing.assert_array_equal(canonical.nodeArray, info.nodeArray)
        np.testing.assert_array_equal(canonical.elementNodesArray, info.elementNodesArray)
        np.testing.assert_array_equal(canonical.nodeMaterialTypes, info.nodeMaterialTypes)
        np.testing.assert_array_equal(canonical.elementMaterialTypes, info.elementMaterialTypes)
        assert MeshTools.meshFingerprint(canonical.nodeArray, canonical.elementNodesArray) == expected


def test_contiguous_owners_keep_the_input_order_and_write_no_map(tmp_path):
    info, expected = _input(SQUARE)
    fingerprint, renumbered = _archive(SQUARE, 2, "contiguous", "global", tmp_path)
    assert (fingerprint, renumbered) == (expected, False)
    archived = MeshTools.readMeshFromArchive(str(tmp_path / "mesh"))
    assert archived.nodeNumbering_input is None and archived.elementNumbering_input is None
    np.testing.assert_array_equal(archived.nodeArray, info.nodeArray)
    with h5py.File(tmp_path / "mesh.h5", "r") as f:
        assert not [name for name in f if name.startswith(("NodeMapInput", "CellMapInput"))]


def test_each_rank_grid_of_a_per_rank_archive_names_its_input_rows(tmp_path):
    """global_sync off: each rank's grid holds its subdomain, and NodeMapInput/CellMapInput say which
    input nodes and elements those are (what a per-rank reader needs)."""
    info, _ = _input(CUBE)
    _archive(CUBE, 2, "reversed", "per-rank", tmp_path)
    domain = Archiver.readArchiveDomain("mesh", str(tmp_path))
    step = Archiver.archive_collections(domain)[0]["data"][-1]
    grids = step["spatial_collection"]
    assert len(grids) == 2
    owned = []
    with h5py.File(tmp_path / "mesh.h5", "r") as f:
        def read(item):
            return f["/" + item["data"].rsplit(":/", 1)[1]][()]
        for grid in grids:
            maps = {a["name"]: read(a["data_item"]) for a in grid["attributes"]}
            # the input numbering where the partitioner renumbered, else the local-to-global map
            maps.setdefault("NodeMapInput", maps["NodeMapL2G"])
            maps.setdefault("CellMapInput", maps["CellMapL2G"])
            nodes, elements = read(grid["geometry"]["data_item"]), read(grid["topology"]["data_item"])
            np.testing.assert_array_equal(nodes, info.nodeArray[maps["NodeMapInput"]])
            # local connectivity -> input node numbers is the input connectivity of the owned elements
            np.testing.assert_array_equal(maps["NodeMapInput"][elements],
                                          info.elementNodesArray[maps["CellMapInput"]])
            owned.append(maps["CellMapInput"])
    assert sorted(np.concatenate(owned).tolist()) == list(range(len(info.elementNodesArray)))


def test_the_fingerprint_is_of_geometry_and_topology_in_order():
    info, expected = _input(SQUARE)
    moved = info.nodeArray.copy()
    moved[0, 0] += 1e-12
    assert MeshTools.meshFingerprint(moved, info.elementNodesArray) != expected
    assert MeshTools.meshFingerprint(info.nodeArray, info.elementNodesArray[::-1]) != expected
    assert MeshTools.meshFingerprint(info.nodeArray[:, :2], info.elementNodesArray) == expected
