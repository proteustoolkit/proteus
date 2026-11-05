"""The per-rank ymf view of a partitioned mesh (C7.1 step H).

``subdomainYmfGrid`` writes one rank's piece of a partitioned mesh as a ymf grid whose facets are in the
``subdomain`` view: local numbering, overlap included, maps to the input mesh and owners, and every
exterior face tagged, -1 on the faces the partition cut. ``subdomainYmfGrids`` computes every rank's
view in one process through C32's ``cpartitioning.partitionNodesLocal``; it must give, bit for bit, the
grid each rank of an MPI run gets from partitionMesh. ``subdomainMeshFromYmf`` reads a view back.
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from proteus import MeshTools

pytest.importorskip("ymf.mesh")

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


def _owners(kind, n, size):
    return {"scatter": np.arange(n) % size,
            "reversed": size - 1 - np.arange(n) * size // n}[kind].astype(np.intc)


def _run(prefix, kind, n_ranks, outdir):
    """Each rank's (grid, arrays) from partitionMesh under mpiexec."""
    mpiexec = _mpiexec()
    if mpiexec is None:
        pytest.skip("no MPI launcher available")
    _skip_under_mpiexec()
    env = _launcher_env()
    cmd = [mpiexec, "-n", str(n_ranks), sys.executable, os.path.join(HERE, "subdomain_view_mpi.py"),
           prefix, kind, str(outdir)]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(outdir), timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    ranks = []
    for r in range(n_ranks):
        with open(os.path.join(outdir, "rank%d.json" % r)) as f:
            grid = json.load(f)
        ranks.append((grid, dict(np.load(os.path.join(outdir, "rank%d.npz" % r)))))
    return ranks


def _attributes(grid):
    from ymf.archive import load_array
    return dict((a["name"], np.asarray(load_array(a["data_item"])).reshape(-1)) for a in grid["attributes"])


def _face_flags(mesh_or_arrays):
    """{sorted face nodes: flag} over the exterior faces (face numbering differs between builders)."""
    get = (lambda k: mesh_or_arrays[k]) if isinstance(mesh_or_arrays, dict) else \
        (lambda k: getattr(mesh_or_arrays, k))
    exterior = np.asarray(get("exteriorElementBoundariesArray"))
    faces = np.sort(np.asarray(get("elementBoundaryNodesArray"))[exterior], axis=1)
    flags = np.asarray(get("elementBoundaryMaterialTypes"))[exterior]
    return dict(zip(map(tuple, faces.tolist()), flags.tolist()))


#: every case cuts faces (scatter on 2 ranks gives both ranks the whole of these small meshes)
CASES = [(SQUARE, "scatter", 3), (SQUARE, "reversed", 2), (CUBE, "scatter", 3), (CUBE, "reversed", 3)]
IDS = ["tri-scatter-3", "tri-reversed-2", "tet-scatter-3", "tet-reversed-3"]


@pytest.mark.parametrize("prefix, kind, n_ranks", CASES, ids=IDS)
def test_the_views_computed_in_one_process_are_the_ranks_views(prefix, kind, n_ranks, tmp_path):
    ranks = _run(prefix, kind, n_ranks, tmp_path)
    info = MeshTools.readMeshYmf(prefix)
    local = MeshTools.subdomainYmfGrids(prefix, n_ranks, _owners(kind, info.nNodes_global, n_ranks))
    for r, ((grid, _), mine) in enumerate(zip(ranks, local)):
        assert json.loads(json.dumps(mine)) == grid, r


@pytest.mark.parametrize("prefix, kind, n_ranks", CASES, ids=IDS)
def test_a_view_reads_back_as_the_ranks_subdomain_mesh(prefix, kind, n_ranks, tmp_path):
    ranks = _run(prefix, kind, n_ranks, tmp_path)
    cut = 0
    for r, (grid, arrays) in enumerate(ranks):
        mesh = MeshTools.subdomainMeshFromYmf(grid)
        np.testing.assert_array_equal(mesh.nodeArray, arrays["nodeArray"])
        np.testing.assert_array_equal(mesh.elementNodesArray, arrays["elementNodesArray"])
        np.testing.assert_array_equal(mesh.nodeMaterialTypes, arrays["nodeMaterialTypes"])
        np.testing.assert_array_equal(mesh.elementMaterialTypes, arrays["elementMaterialTypes"])
        assert _face_flags(mesh) == _face_flags(arrays), r
        assert (mesh.nNodes_owned, mesh.nElements_owned) == (arrays["nNodes_owned"], arrays["nElements_owned"])
        assert mesh.rankInterfacesMarked and bool(arrays["marked"])
        cut += sum(1 for flag in _face_flags(mesh).values() if flag == MeshTools.RANK_INTERFACE_FLAG)
    assert cut > 0                                 # the cases do cut faces


@pytest.mark.parametrize("prefix, kind, n_ranks", CASES, ids=IDS)
def test_the_views_put_together_are_the_input_mesh(prefix, kind, n_ranks):
    info = MeshTools.readMeshYmf(prefix)
    grids = MeshTools.subdomainYmfGrids(prefix, n_ranks, _owners(kind, info.nNodes_global, n_ranks))
    owned_cells, owned_nodes = [], []
    for r, grid in enumerate(grids):
        a = _attributes(grid)
        mesh = MeshTools.subdomainMeshFromYmf(grid)
        np.testing.assert_array_equal(mesh.nodeArray, info.nodeArray[a["NodeMapInput"]])
        np.testing.assert_array_equal(a["NodeMapInput"][mesh.elementNodesArray],
                                      info.elementNodesArray[a["CellMapInput"]])
        np.testing.assert_array_equal(mesh.elementMaterialTypes, info.elementMaterialTypes[a["CellMapInput"]])
        assert np.all(a["NodeOwner"][:mesh.nNodes_owned] == r) and np.all(a["NodeOwner"][mesh.nNodes_owned:] != r)
        assert np.all(a["CellOwner"][:mesh.nElements_owned] == r)
        owned_nodes.append(a["NodeMapInput"][:mesh.nNodes_owned])
        owned_cells.append(a["CellMapInput"][:mesh.nElements_owned])
    assert sorted(np.concatenate(owned_nodes).tolist()) == list(range(info.nNodes_global))
    assert sorted(np.concatenate(owned_cells).tolist()) == list(range(info.nElements_global))


def test_one_rank_has_no_cut_faces_and_keeps_the_input_tags():
    info = MeshTools.readMeshYmf(SQUARE)
    (grid,) = MeshTools.subdomainYmfGrids(SQUARE, 1, np.zeros(info.nNodes_global, dtype=np.intc))
    mesh = MeshTools.subdomainMeshFromYmf(grid)
    whole = MeshTools.TriangularMesh()
    whole.generateFromYmfFile(SQUARE)
    assert _face_flags(mesh) == _face_flags(whole)
    assert sorted(set(_face_flags(mesh).values())) == [1, 2, 3, 4]
    np.testing.assert_array_equal(mesh.elementBoundaryUserFlags, mesh.elementBoundaryMaterialTypes)


def test_a_global_mesh_is_not_a_subdomain_view():
    from ymf import archive
    domain, _ = archive.read_ymf(SQUARE + ".ymf")
    grid = archive.inline_approximation(domain, HERE + "/ymf_meshes")["times"][0]["spaces"]["Spatial_Domain"]
    with pytest.raises(ValueError, match="not a subdomain view"):
        MeshTools.subdomainMeshFromYmf(grid)
