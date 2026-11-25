"""A static mesh is archived once, a moving mesh's nodes every frame, an adapted mesh's topology again (C1, 2026-11-24).

Until 2026-11-24 every archived frame rewrote the whole mesh (connectivity, nodes, local-to-global maps, input
numbering, material types): ``NumericalSolution._archiveSolution`` passed ``meshChanged=True``. On Mike that was
more than half of every bathyduck frame (144 ranks, 6.9 M tets) and three quarters of a 45DEG_R1 frame. Now each
mesh array is written when what it is made from changed (``Archiver.mesh_step``, ``MeshTools.Mesh.archiveEpoch``),
and the grid of every frame names the dataset of the frame it was written at, so the readers, which follow the
grid's references, need no change. These tests archive four frames of a 2D and a 3D mesh from 1 and 2 ranks with
every archive layout and check: one copy of each mesh array for a static mesh; nodes per frame and connectivity
once for a moving one; the topology written again from the frame where an adapted mesh replaced the first; every
frame's grid resolves to arrays that exist, with the declared dimensions; and the mesh read back from the last
frame is the mesh, in the input order, whatever the rank count.
"""
import os
import re
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pytest

from proteus import Archiver, Comm, MeshTools

HERE = os.path.dirname(os.path.abspath(__file__))
SQUARE = os.path.join(HERE, "ymf_meshes", "square4x4_tagged")
CUBE = os.path.join(HERE, "ymf_meshes", "cube_3x3x3")
FRAMES = 4

#: the mesh arrays the spatial mesh writes, by the dataset name without its step
MESH_FAMILY = re.compile(r"^(elements|nodes|nodeMapL2G|cellMapL2G|NodeMapInput|CellMapInput)(\d*)Spatial_Domain(\d+)$"
                         r"|^(nodeMaterialTypes|elementMaterialTypes)(_p\d+)?_t(\d+)$")
#: (sync, driver): the single-file layouts with and without global arrays, and per-rank files
LAYOUTS = [("global", "mpio"), ("per-rank", "mpio"), ("per-rank", "serial"), ("global", "serial"),
           ("per-rank", "per-rank")]


def _launcher_env():
    env = dict((k, v) for k, v in os.environ.items() if not k.startswith(("PMI_", "PMIX_", "HYDRA_", "OMPI_COMM_WORLD")))
    env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env.pop("PETSC_OPTIONS", None)                 # owners are prescribed: no MatPartitioning runs
    return env


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ("mpiexec", "mpirun"):
        if os.path.exists(os.path.join(bindir, name)):
            return os.path.join(bindir, name)
    return shutil.which("mpiexec") or shutil.which("mpirun")


def _archive(prefix, n_ranks, sync, driver, motion, outdir):
    """Run the driver; returns the global node count of each frame, as rank 0 printed it."""
    for name in ("PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "PMIX_SIZE"):
        if int(os.environ.get(name, "1")) > 1:
            pytest.skip("launches its own mpiexec; run this file serially")
    os.makedirs(outdir, exist_ok=True)
    cmd = [sys.executable, os.path.join(HERE, "archive_static_mesh_mpi.py"), prefix, sync, driver, motion,
           str(FRAMES), str(outdir)]
    mpiexec = _mpiexec()
    if mpiexec is None:
        pytest.skip("no MPI launcher available")
    cmd = [mpiexec, "-n", str(n_ranks)] + cmd
    result = subprocess.run(cmd, capture_output=True, text=True, env=_launcher_env(), cwd=str(outdir), timeout=300)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    line = next(l for l in result.stdout.splitlines() if l.startswith("nodes="))
    return [int(n) for n in line.split("=")[1].split(",")]


def _files(outdir):
    return sorted(f for f in os.listdir(outdir) if f.endswith(".h5"))


def _family(m):
    if m.group(1):
        return m.group(1) + m.group(2), int(m.group(3))
    return m.group(4) + (m.group(5) or ""), int(m.group(6))


def _mesh_datasets(outdir):
    """{family: sorted steps} of the mesh arrays written, over every .h5 of the archive (a family: the dataset name
    without its step); the soft links a step that reuses an array gets (Archiver._writeAliases) are not counted."""
    found = {}
    for name in _files(outdir):
        with h5py.File(os.path.join(outdir, name), "r") as f:
            for key in f.keys():
                m = MESH_FAMILY.match(key)
                if m is None or isinstance(f.get(key, getlink=True), h5py.SoftLink):
                    continue
                family, step = _family(m)
                found.setdefault(family, set()).add(step)
    return {k: sorted(v) for k, v in found.items()}


def _step_names_resolve(outdir):
    """Every mesh array can be opened by its per-step name at every step (a dataset, or a link to the step it was
    written at): what the readers that build ``nodesSpatial_Domain%d`` themselves rely on. Returns the links,
    {name: target}."""
    links = {}
    for name in _files(outdir):
        with h5py.File(os.path.join(outdir, name), "r") as f:
            families = {}
            for key in f.keys():
                m = MESH_FAMILY.match(key)
                if m is None:
                    continue
                link = f.get(key, getlink=True)
                if isinstance(link, h5py.SoftLink):
                    links[key] = link.path
                    assert f[key].shape == f[link.path].shape          # resolves
                families.setdefault(_family(m)[0], set()).add(_family(m)[1])
            for family, steps in families.items():
                assert steps == set(range(FRAMES)), (name, family, sorted(steps))
    return links


def _mesh_bytes(outdir):
    total = 0
    for name in _files(outdir):
        with h5py.File(os.path.join(outdir, name), "r") as f:
            for key in f.keys():
                if MESH_FAMILY.match(key) and not isinstance(f.get(key, getlink=True), h5py.SoftLink):
                    total += f[key].size * f[key].dtype.itemsize
    return total


def _frames(outdir):
    """Every frame's grids of the spatial mesh: [[grid, ...] per frame], from the archive's .h5 alone."""
    domain = Archiver.readArchiveDomain("mesh", str(outdir))
    (collection,) = [c for c in Archiver.archive_collections(domain) if "Spatial_Domain" in c["name"]]
    return [step.get("spatial_collection", [step]) for step in collection["data"]]


def _check_references(outdir):
    """check_archive's integrity check: every data item of every frame names a dataset that exists, with the
    dimensions it declares. Returns, per frame, the steps the grids' topology and geometry datasets carry."""
    from ymf.archive import load_array
    steps = []
    for grids in _frames(outdir):
        frame = set()
        for grid in grids:
            for owner in [grid["topology"], grid["geometry"]] + grid["attributes"]:
                item = owner["data_item"]
                array = load_array(item, str(outdir))
                assert array.size == np.prod(item["dimensions"]), item["data"]
            for owner in (grid["topology"], grid["geometry"]):
                frame.add(int(re.search(r"(\d+)$", owner["data_item"]["data"]).group(1)))
        steps.append(sorted(frame))
    return steps


@pytest.mark.parametrize("n_ranks", [1, 2])
@pytest.mark.parametrize("sync,driver", LAYOUTS)
@pytest.mark.parametrize("prefix", [SQUARE, CUBE], ids=["triangles", "tetrahedra"])
def test_a_static_mesh_is_archived_once(prefix, sync, driver, n_ranks, tmp_path):
    _archive(prefix, n_ranks, sync, driver, "static", tmp_path)
    datasets = _mesh_datasets(tmp_path)
    assert datasets and all(steps == [0] for steps in datasets.values()), datasets
    # every frame names the arrays of frame 0, and they are there
    assert _check_references(tmp_path) == [[0]] * FRAMES
    links = _step_names_resolve(tmp_path)
    assert links and all(re.search(r"\D0$", target) for target in links.values()), links
    # the last frame's mesh is the mesh, in the input order whatever the rank count
    info = MeshTools.readMeshYmf(prefix)
    canonical = MeshTools.readMeshFromArchive(str(tmp_path / "mesh"), inputOrder=True) if driver != "per-rank" else None
    if canonical is not None and (sync == "global" or n_ranks == 1):
        np.testing.assert_array_equal(canonical.nodeArray, info.nodeArray)
        np.testing.assert_array_equal(canonical.elementNodesArray, info.elementNodesArray)
        np.testing.assert_array_equal(canonical.elementMaterialTypes, info.elementMaterialTypes)


@pytest.mark.parametrize("n_ranks", [1, 2])
@pytest.mark.parametrize("prefix", [SQUARE, CUBE], ids=["triangles", "tetrahedra"])
def test_the_static_mesh_costs_one_frame_of_mesh_not_one_per_frame(prefix, n_ranks, tmp_path):
    """The before (every frame forced, as _archiveSolution did) and after, on the same layout."""
    _archive(prefix, n_ranks, "global", "mpio", "forced", tmp_path / "before")
    _archive(prefix, n_ranks, "global", "mpio", "static", tmp_path / "after")
    assert all(steps == list(range(FRAMES)) for steps in _mesh_datasets(tmp_path / "before").values())
    assert _mesh_bytes(tmp_path / "before") == FRAMES * _mesh_bytes(tmp_path / "after")
    assert _check_references(tmp_path / "before") == [[k] for k in range(FRAMES)]


@pytest.mark.parametrize("sync,driver", [("global", "mpio"), ("per-rank", "mpio")])
@pytest.mark.parametrize("n_ranks", [1, 2])
@pytest.mark.parametrize("prefix", [SQUARE, CUBE], ids=["triangles", "tetrahedra"])
def test_a_moving_mesh_writes_its_nodes_every_frame_and_its_topology_once(prefix, n_ranks, sync, driver, tmp_path):
    _archive(prefix, n_ranks, sync, driver, "moving", tmp_path)
    datasets = _mesh_datasets(tmp_path)
    for family, steps in datasets.items():
        assert steps == (list(range(FRAMES)) if family.startswith("nodes") and "Map" not in family else [0]), \
            (family, steps)
    _check_references(tmp_path)
    _step_names_resolve(tmp_path)
    # frame k's geometry is the mesh moved k times: y += 0.01 x per frame
    frames = _frames(tmp_path)
    from ymf.archive import load_array
    first = np.concatenate([load_array(g["geometry"]["data_item"], str(tmp_path)) for g in frames[0]])
    for k, grids in enumerate(frames):
        nodes = np.concatenate([load_array(g["geometry"]["data_item"], str(tmp_path)) for g in grids])
        np.testing.assert_allclose(nodes[:, 1], first[:, 1] + 0.01 * k * first[:, 0], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(nodes[:, 0], first[:, 0])


@pytest.mark.parametrize("n_ranks", [1, 2])
@pytest.mark.parametrize("prefix", [SQUARE, CUBE], ids=["triangles", "tetrahedra"])
def test_an_adapted_mesh_writes_its_topology_again(prefix, n_ranks, tmp_path):
    """A new mesh object with another connectivity, as PUMI adaptivity builds, from frame FRAMES // 2 on."""
    nodes = _archive(prefix, n_ranks, "global", "mpio", "adapted", tmp_path)
    change = FRAMES // 2
    assert nodes[0] != nodes[change]
    for family, steps in _mesh_datasets(tmp_path).items():
        assert steps == [0, change], (family, steps)
    assert _check_references(tmp_path) == [[0]] * change + [[change]] * (FRAMES - change)
    _step_names_resolve(tmp_path)
    for k, grids in enumerate(_frames(tmp_path)):
        (grid,) = grids
        assert grid["geometry"]["data_item"]["dimensions"][0] == nodes[k]


def test_the_mesh_epoch_changes_when_its_arrays_do():
    """MeshTools.Mesh.archiveEpoch: read from the arrays, so any producer of a change is seen."""
    Comm.init()
    comm = Comm.get()
    mesh = MeshTools.TetrahedralMesh()
    mesh.generateFromYmfFile(CUBE)
    mesh.partitionMesh(nLayersOfOverlap=1, nodeOwners=np.zeros(mesh.nNodes_global, dtype=np.intc))
    sub = mesh.subdomainMesh
    kinds = ("topology", "geometry", "materials")
    first = sub.archiveEpoch(comm, kinds, 0)
    assert sub.archiveEpoch(comm, kinds, 1) == first                     # nothing changed
    sub.nodeArray[0, 0] += 1.0e-3                                        # moved in place
    moved = sub.archiveEpoch(comm, kinds, 2)
    assert moved[0] == first[0] and moved[2] == first[2] and moved[1] != first[1]
    assert sub.archiveEpoch(comm, kinds, 2) == moved                     # cached for the step
    sub.elementMaterialTypes[0] += 1
    retyped = sub.archiveEpoch(comm, kinds, 3)
    assert retyped[:2] == moved[:2] and retyped[2] != moved[2]
    other = MeshTools.TetrahedralMesh()                                  # a new object: new numbers
    other.generateFromYmfFile(CUBE)
    other.partitionMesh(nLayersOfOverlap=1, nodeOwners=np.zeros(other.nNodes_global, dtype=np.intc))
    renewed = other.subdomainMesh.archiveEpoch(comm, kinds, 3)
    assert all(a != b for a, b in zip(renewed, retyped))


def test_a_hot_start_after_a_topology_change_is_refused(tmp_path):
    """Archiver.frame_mesh_mismatch: the steps before the adapted mesh are this mesh's, the steps after are not
    (NumericalSolution._checkHotStartMesh raises on them instead of reading fields of another mesh)."""
    _archive(CUBE, 1, "global", "mpio", "adapted", tmp_path)
    Comm.init()
    mesh = MeshTools.TetrahedralMesh()
    mesh.generateFromYmfFile(CUBE)
    mesh.partitionMesh(nLayersOfOverlap=1, nodeOwners=np.zeros(mesh.nNodes_global, dtype=np.intc))
    ar = Archiver.XdmfArchive(str(tmp_path), "mesh", readOnly=True, global_sync=True)
    try:
        for k in range(FRAMES // 2):
            assert ar.frame_mesh_mismatch(k, mesh.subdomainMesh) is None, k
        for k in range(FRAMES // 2, FRAMES):
            why = ar.frame_mesh_mismatch(k, mesh.subdomainMesh)
            assert why is not None and "has 162 elements and 64 nodes" in why and "this run's has 48 and 27" in why, why
    finally:
        ar.close()


def test_a_solver_state_of_another_connectivity_is_refused():
    """NS_base._stateFingerprintMatches: the connectivity is the last entry of the fingerprint; a state written
    before it was added is compared on the partition only, as it was."""
    from proteus.NumericalSolution import NS_base
    current = np.array([2, 48, 27, 11, 12, 13], dtype="int64")
    assert NS_base._stateFingerprintMatches(current.copy(), current) == (True, None)
    assert NS_base._stateFingerprintMatches(current[:5], current) == (True, None)          # an older archive
    ok, why = NS_base._stateFingerprintMatches(np.array([2, 48, 27, 11, 12, 99]), current)
    assert not ok and "another connectivity" in why
    ok, why = NS_base._stateFingerprintMatches(np.array([2, 48, 27, 11, 77, 13]), current)
    assert not ok and "different partition" in why
