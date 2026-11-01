"""The archive's in-HDF5 grid metadata, including under real MPI.

Phase 2 of the YMF I/O campaign replaced the XML fragments that
``allGatherIncremental`` used to stash in the HDF5 file with YAML
describing ymf grid dicts, and made the final document be assembled from
those rather than from an accumulating ElementTree.

The parallel behaviour is the part that most needs a test and had none:
the dataset holding per-rank metadata is created collectively, so every
rank must agree on its width, and the width depends on data only some
ranks have. These tests run the real thing under ``mpiexec`` at several
rank counts, with deliberately different grid sizes per rank so that an
agreement bug shows up as a truncated payload rather than passing by
luck.

Run directly (``pytest``) they cover the serial path; the MPI cases
re-invoke themselves through ``mpiexec`` as subprocesses.
"""

from proteus.Archiver import archive_collections
import os
import shutil
import subprocess
import sys
import textwrap

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("proteus.Archiver")

def _find_mpiexec():
    """The launcher that matches this interpreter's mpi4py, not just any.

    A mismatched launcher does not fail -- it starts N independent
    single-rank jobs, so a test asking for 3 ranks quietly exercises 1 and
    passes for the wrong reason. On this machine PATH resolves mpiexec to
    Homebrew's OpenMPI while mpi4py is built against conda's MPICH, which
    is exactly that situation. Prefer the launcher installed beside
    sys.executable.
    """
    beside = os.path.join(os.path.dirname(sys.executable), "mpiexec")
    if os.path.exists(beside):
        return beside
    return shutil.which("mpiexec")


MPIEXEC = _find_mpiexec()

# The body run inside each MPI rank. Kept as source text rather than a
# module so the rank count and mode can be varied without a fixture file
# per combination, and so a failure inside a rank surfaces as that rank's
# traceback on stderr.
RANK_SCRIPT = textwrap.dedent(
    '''
    import os, shutil, sys
    import numpy as np

    from proteus import Comm
    Comm.init()
    comm = Comm.get()
    rank, size = comm.rank(), comm.size()

    global_sync = os.environ["GLOBAL_SYNC"] == "1"
    datadir = os.environ["DATADIR"]
    if rank == 0 and os.environ.get("HOT_START") not in ("1", "back"):
        shutil.rmtree(datadir, ignore_errors=True)
        os.makedirs(datadir, exist_ok=True)
    comm.barrier()

    from proteus import Archiver

    hot = os.environ.get("HOT_START") in ("1", "back")
    if os.environ.get("WITH_LOG") == "1":
        from proteus import Profiling
        Profiling.procID = rank
        Profiling.openLog("mpitest.log", 3, logLocation=datadir)
        Profiling.logEvent("a line of the run's log")
    ar = Archiver.XdmfArchive(datadir, "mpitest", global_sync=global_sync,
                              hotStart=hot,
                              mode=os.environ.get("MODE") or None,
                              driver=os.environ.get("DRIVER") or None)
    if os.environ.get("NO_INCREMENTAL") == "1":
        ar.writeIncrementalYmf = False
    if hot:
        # what NumericalSolution does: continue after the archive's last step
        # (or, for HOT_START=back, from the first step of two: the second
        # is written again, as when the requested time is not the last)
        last = 0 if os.environ.get("HOT_START") == "back" else len(ar.archived_times) - 1
        ar.continue_from(last)

    # Sizes differ per rank on purpose: the metadata dataset is fixed-width
    # and created collectively, so if the width agreement is wrong the
    # longest rank's YAML is silently truncated.
    n_elements, n_nodes = 4 + rank, 6 + rank
    if hot:
        # found by name, as XdmfWriter.setGridCollectionAndGridElements does;
        # the archive seeded it from the .h5 when it was opened
        collection = ar.domain.collection("Mesh Spatial_Domain", create=False)
        assert collection is not None, ar.domain.names()
    else:
        collection = ar.domain.collection("Mesh Spatial_Domain")

    # global_sync describes one assembled global array, so the metadata
    # carries global counts and each rank contributes only the slice it owns.
    comm_world = comm.comm.tompi4py()
    node_offsets = np.concatenate(
        ([0], np.cumsum(comm_world.allgather(n_nodes)))).astype("i")
    n_nodes_global = int(node_offsets[-1])
    element_offsets = np.concatenate(
        ([0], np.cumsum(comm_world.allgather(n_elements)))).astype("i")
    n_elements_global = int(element_offsets[-1])
    # this rank's mesh arrays: a triangle per element over the first nodes,
    # and nodes on a line, so every rank's piece is distinguishable
    elements = np.array([[i % n_nodes, (i + 1) % n_nodes, (i + 2) % n_nodes]
                         for i in range(n_elements)], dtype="i")
    nodes = np.zeros((n_nodes, 3), "d")
    nodes[:, 0] = node_offsets[rank] + np.arange(n_nodes)

    times = [1.0] if hot else [0.0, 0.5]
    for tCount, t in enumerate(times, start=ar.n_datasets):
        grid = ar.write_grid(collection, "Grid_p%d" % rank, t, tCount)
        if global_sync:
            ar.write_topology(grid, "Triangle", n_elements_global,
                              [n_elements_global, 3],
                              "elements_t%d" % tCount, "elements%d" % tCount,
                              dtype="i")
            ar.write_geometry(grid, [n_nodes_global, 3],
                              "nodes_t%d" % tCount, "nodes%d" % tCount,
                              dtype="d")
            # the mesh arrays, as a mesh writer writes them on init
            ar.create_dataset_sync("elements_t%d" % tCount, offsets=element_offsets,
                                   data=elements + node_offsets[rank])
            ar.create_dataset_sync("nodes_t%d" % tCount, offsets=node_offsets,
                                   data=nodes)
            ar.write_field(grid, "u", np.arange(n_nodes, dtype="d"), tCount,
                           dimensions=[n_nodes_global],
                           sync_offsets=node_offsets,
                           sync_data=np.arange(n_nodes, dtype="d"))
        else:
            ar.write_topology(grid, "Triangle", n_elements, [n_elements, 3],
                              "elements_p%d_t%d" % (rank, tCount),
                              "elements%d" % tCount, dtype="i")
            ar.write_geometry(grid, [n_nodes, 3],
                              "nodes_p%d_t%d" % (rank, tCount),
                              "nodes%d" % tCount, dtype="d")
            ar.create_dataset_async("elements_p%d_t%d" % (rank, tCount), data=elements)
            ar.create_dataset_async("nodes_p%d_t%d" % (rank, tCount), data=nodes)
            ar.write_field(grid, "u", np.arange(n_nodes, dtype="d"), tCount,
                           dimensions=[n_nodes])
        ar.sync()

    if os.environ.get("SKIP_CLOSE") == "1":
        # what a killed run leaves behind: the steps are synced, close()
        # never happens. Flush so the HDF5 is as it would be, then stop.
        if ar.hdfFile is not None:
            ar.hdfFile.flush()
    else:
        ar.close()
    comm.barrier()
    if rank == 0:
        print("OBSERVED_SIZE=%d" % size)
    '''
)


def run_ranks(tmp_path, n_ranks, global_sync, hot_start=False,
              skip_close=False, incremental=True, mode=None, driver=None,
              with_log=False, name="archive"):
    """Run the rank script under mpiexec (directly only if there is none).

    ``hot_start`` continues the archive an earlier call wrote, for one
    more step, the way a hot-started run does. ``mode`` and ``driver`` are
    the archive's; ``with_log`` opens a proteus log first, as a run does.
    """
    datadir = str(tmp_path / name)
    env = dict(os.environ, GLOBAL_SYNC="1" if global_sync else "0",
               DATADIR=datadir, HOT_START=("back" if hot_start == "back" else "1") if hot_start else "0",
               SKIP_CLOSE="1" if skip_close else "0",
               NO_INCREMENTAL="0" if incremental else "1",
               MODE=mode or "", DRIVER=driver or "",
               WITH_LOG="1" if with_log else "0")
    # More ranks than cores (3 on a 2-core CI runner): Open MPI refuses
    # unless told it may oversubscribe -- OMPI_MCA_* for Open MPI 4,
    # PRTE_MCA_* for Open MPI 5's PRRTE launcher. MPICH oversubscribes by
    # default and ignores both.
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    # Under mpiexec even for one rank. Run directly, the script is an MPICH
    # singleton, and MPICH 5 names a singleton's shared memory after its pid
    # (/mpich_shm_<hex pid>_0), attaching to the name if it already exists. A
    # killed MPI process never unlinks it, so on a long-running machine stale
    # names pile up, and once a pid comes round again MPI_Init finds one --
    # another user's gives EACCES and the run aborts in "MPIDI_POSIX_comm_bootstrap:
    # Out of memory" (three CI runs in a row on a shared Mac). Under mpiexec the
    # name is a hash of the job's name, which hydra makes random per launch.
    cmd = [sys.executable, "-c", RANK_SCRIPT]
    if MPIEXEC:
        cmd = [MPIEXEC, "-n", str(n_ranks)] + cmd
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                            env=env)
    assert result.returncode == 0, (
        "rank script failed with %d ranks:\nSTDOUT:\n%s\nSTDERR:\n%s"
        % (n_ranks, result.stdout[-2000:], result.stderr[-4000:]))
    # Refuse to pass on a degraded run: a mismatched launcher yields N
    # independent single-rank jobs rather than an error.
    observed = [line for line in result.stdout.splitlines()
                if line.startswith("OBSERVED_SIZE=")]
    assert observed, (
        "rank 0 never reported its communicator size; stdout:\n%s"
        % result.stdout[-2000:])
    got = int(observed[0].split("=")[1])
    assert got == n_ranks, (
        "asked for %d ranks but the job ran with a communicator of %d -- the "
        "launcher %s does not match this interpreter's mpi4py, so this test "
        "would otherwise pass while exercising a single rank"
        % (n_ranks, got, MPIEXEC))
    return datadir


def read_domain(datadir):
    from ymf.archive import read_ymf

    return read_ymf(os.path.join(datadir, "mpitest.ymf"))[0]


# --------------------------------------------------------------------------
# serial
# --------------------------------------------------------------------------


def test_serial_per_rank_archive_is_written_as_ymf(tmp_path):
    domain = read_domain(run_ranks(tmp_path, 1, global_sync=False))
    (collection,) = archive_collections(domain)
    assert collection["name"] == "Mesh Spatial_Domain"
    assert len(collection["data"]) == 2
    assert [s["time"] for s in collection["data"]] == [0.0, 0.5]


def test_serial_global_sync_archive_has_uniform_steps(tmp_path):
    domain = read_domain(run_ranks(tmp_path, 1, global_sync=True))
    (collection,) = archive_collections(domain)
    for step in collection["data"]:
        assert "spatial_collection" not in step
        assert step["topology"]["type"] == "Triangle"


def test_metadata_is_yaml_not_xml(tmp_path):
    datadir = run_ranks(tmp_path, 1, global_sync=False)
    with h5py.File(os.path.join(datadir, "mpitest.h5"), "r") as f:
        (name,) = [k for k in f if k.startswith("Mesh_Spatial_Domain_0")]
        payload = f[name][0].decode("utf-8")
    assert not payload.lstrip().startswith("<"), "metadata is still XML"
    from ymf.archive import load_grid

    grid = load_grid(payload)
    assert grid["topology"]["type"] == "Triangle"


def test_the_metadata_format_version_is_recorded(tmp_path):
    from proteus.Archiver import AR_base

    datadir = run_ranks(tmp_path, 1, global_sync=False)
    with h5py.File(os.path.join(datadir, "mpitest.h5"), "r") as f:
        assert int(f.attrs[AR_base.METADATA_VERSION_ATTR]) == \
            AR_base.METADATA_FORMAT_VERSION


def test_an_archive_without_the_version_marker_is_read_as_version_one(tmp_path):
    """A missing marker means proteus <= 1.9.x wrote it, and it is readable.

    This started out as a refusal. The campaign gained an objective after
    the fact -- preserve the XDMF consumer pathways, since 1.9.x is being
    released and its archives will be around -- so the absence of the
    attribute is now a version rather than an error. See
    test_archive_v1_compat.py for reading both v1 layouts.
    """
    from proteus.Archiver import AR_base

    datadir = run_ranks(tmp_path, 1, global_sync=False)
    path = os.path.join(datadir, "mpitest.h5")
    with h5py.File(path, "a") as f:
        del f.attrs[AR_base.METADATA_VERSION_ATTR]

    ar = AR_base.__new__(AR_base)
    ar.hdfFilename = "mpitest.h5"
    with h5py.File(path, "r") as f:
        ar.hdfFile = f
        assert ar._metadata_version() == 1


def test_a_future_version_marker_is_refused(tmp_path):
    """Older layouts are read; newer ones cannot be guessed at."""
    from proteus.Archiver import AR_base

    datadir = run_ranks(tmp_path, 1, global_sync=False)
    path = os.path.join(datadir, "mpitest.h5")
    with h5py.File(path, "a") as f:
        f.attrs[AR_base.METADATA_VERSION_ATTR] = 99

    ar = AR_base.__new__(AR_base)
    ar.hdfFilename = "mpitest.h5"
    with h5py.File(path, "r") as f:
        ar.hdfFile = f
        with pytest.raises(ValueError, match="version 99"):
            ar._metadata_version()


# --------------------------------------------------------------------------
# parallel
# --------------------------------------------------------------------------


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("n_ranks", [2, 3])
def test_every_rank_contributes_a_grid_to_each_step(tmp_path, n_ranks):
    domain = read_domain(run_ranks(tmp_path, n_ranks, global_sync=False))
    (collection,) = archive_collections(domain)
    assert len(collection["data"]) == 2
    for step in collection["data"]:
        subs = step["spatial_collection"]
        assert len(subs) == n_ranks
        assert [g["name"] for g in subs] == [
            "Grid_p%d" % r for r in range(n_ranks)
        ]


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("n_ranks", [2, 3])
def test_differently_sized_ranks_are_not_truncated(tmp_path, n_ranks):
    """The collective width agreement must fit the *longest* rank.

    Each rank writes n_elements = 4 + rank, so the YAML payloads differ in
    length. A width agreed from only one rank's view would truncate the
    others and their topology counts would come back wrong or unparseable.
    """
    domain = read_domain(run_ranks(tmp_path, n_ranks, global_sync=False))
    (collection,) = archive_collections(domain)
    for step in collection["data"]:
        counts = [g["topology"]["number_of_elements"]
                  for g in step["spatial_collection"]]
        assert counts == [4 + r for r in range(n_ranks)]


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
def test_parallel_archive_validates_as_a_ymf_document(tmp_path):
    from ymf.archive import validate_domain

    validate_domain(read_domain(run_ranks(tmp_path, 2, global_sync=False)))


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
def test_ymf2xmf_holds_every_ranks_grid(tmp_path):
    """The .xmf for viewers is derived from the .ymf, and holds every rank.

    The archive itself writes no .xmf (Phase 4). Before that, the legacy
    code wrote ``self.treeGlobal`` at close; once the grids stopped
    accumulating in that tree, that write truncated the file back to an
    empty temporal collection -- the .ymf was complete and the .xmf a stub.
    """
    from xml.etree.ElementTree import parse
    from ymf.cli import ymf2xmf

    datadir = run_ranks(tmp_path, 2, global_sync=False)
    xmf = ymf2xmf(os.path.join(datadir, "mpitest.ymf"),
                  str(tmp_path / "derived.xmf"))
    root = parse(xmf).getroot()
    temporal = root.find("Domain").find("Grid")
    assert temporal.attrib["CollectionType"] == "Temporal"
    steps = temporal.findall("Grid")
    assert len(steps) == 2, "expected one spatial collection per timestep"
    for step in steps:
        assert step.attrib["CollectionType"] == "Spatial"
        assert step.find("Time") is not None
        assert len(step.findall("Grid")) == 2


# --------------------------------------------------------------------------
# mode (a): global arrays, with the decomposition invisible to a consumer
# --------------------------------------------------------------------------


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("n_ranks", [2, 3])
def test_global_sync_hides_the_decomposition(tmp_path, n_ranks):
    """The point of global_sync: a consumer sees one undivided mesh.

    Each rank writes only the nodes it owns into one collectively-sized
    global array, and the metadata describes that global array. So the
    archive must hold uniform steps -- no spatial collection, no per-rank
    grids -- carrying the global counts.
    """
    domain = read_domain(run_ranks(tmp_path, n_ranks, global_sync=True))
    (collection,) = archive_collections(domain)
    expected_nodes = sum(6 + r for r in range(n_ranks))
    expected_elements = sum(4 + r for r in range(n_ranks))
    for step in collection["data"]:
        assert "spatial_collection" not in step, \
            "the decomposition leaked into a global archive"
        assert step["topology"]["number_of_elements"] == expected_elements
        assert step["geometry"]["data_item"]["dimensions"] == [expected_nodes, 3]
        (attr,) = step["attributes"]
        assert attr["data_item"]["dimensions"] == [expected_nodes]


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("n_ranks", [2, 3])
def test_global_sync_assembles_every_ranks_contribution(tmp_path, n_ranks):
    """The global array must actually contain every rank's values.

    A collective write with wrong offsets produces an array of the right
    shape with holes in it, which no amount of metadata checking catches.
    """
    datadir = run_ranks(tmp_path, n_ranks, global_sync=True)
    with h5py.File(os.path.join(datadir, "mpitest.h5"), "r") as f:
        u = f["u_t0"][:]
    expected = np.concatenate([np.arange(6 + r, dtype="d")
                               for r in range(n_ranks)])
    assert u.shape == expected.shape
    np.testing.assert_array_equal(u, expected)


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
def test_global_sync_datasets_carry_no_rank_in_their_names(tmp_path):
    # A consumer of a global archive should not need to know how many ranks
    # produced it.
    datadir = run_ranks(tmp_path, 2, global_sync=True)
    with h5py.File(os.path.join(datadir, "mpitest.h5"), "r") as f:
        names = [k for k in f if not k.startswith("Mesh_")]
    assert names, "no data datasets were written"
    assert not [n for n in names if "_p" in n], names


# --------------------------------------------------------------------------
# reading back: hot start supports exactly the two modes the archive writes
# --------------------------------------------------------------------------


def _archive_with_fields(tmp_path):
    """An archive holding one per-subdomain field and one global field."""
    path = str(tmp_path / "a.h5")
    with h5py.File(path, "w") as f:
        for r in range(4):
            f["u_p%d_t0" % r] = np.arange(3, dtype="d")
        f["v_t0"] = np.arange(12, dtype="d")
    return path


class _StubArchive:
    """Just what field_dataset touches."""

    def __init__(self, hdf, global_sync, rank, size):
        from proteus.Archiver import AR_base

        self.field_dataset = AR_base.field_dataset.__get__(self)
        self.hdfFile = hdf
        self.hdfFilename = "a.h5"
        self.global_sync = global_sync
        self.size = size

        class _Comm:
            def __init__(s, r):
                s._r = r

            def rank(s):
                return s._r

        self.comm = _Comm(rank)


def test_a_global_field_reads_back_at_any_task_count(tmp_path):
    path = _archive_with_fields(tmp_path)
    with h5py.File(path, "r") as f:
        for size in (1, 4, 9):
            ds = _StubArchive(f, True, 0, size).field_dataset("v", 0)
            assert ds.shape == (12,), "global mode must not depend on task count"


def test_a_per_subdomain_field_reads_back_at_the_matching_task_count(tmp_path):
    path = _archive_with_fields(tmp_path)
    with h5py.File(path, "r") as f:
        for rank in range(4):
            ds = _StubArchive(f, False, rank, 4).field_dataset("u", 0)
            assert ds.shape == (3,)


def test_a_per_subdomain_field_at_the_wrong_task_count_says_so(tmp_path):
    """The failure mode a user will actually hit, with a usable message.

    Rank i reads subdomain i, so a per-subdomain archive is only readable
    at the task count that wrote it. Without this the symptom is a bare
    KeyError on a mangled dataset name.
    """
    path = _archive_with_fields(tmp_path)
    with h5py.File(path, "r") as f:
        with pytest.raises(KeyError) as exc:
            _StubArchive(f, False, 5, 6).field_dataset("u", 0)
    message = str(exc.value)
    assert "same number of MPI tasks" in message
    assert "holds 4 subdomain" in message      # what the archive has
    assert "6 task" in message                 # what this run has


def test_reading_a_per_subdomain_archive_in_global_mode_says_so(tmp_path):
    path = _archive_with_fields(tmp_path)
    with h5py.File(path, "r") as f:
        with pytest.raises(KeyError, match="written per-subdomain"):
            _StubArchive(f, True, 0, 1).field_dataset("u", 0)


def test_the_pytables_hot_start_path_is_gone():
    """hdfFileGlb read fields with get_node, a PyTables method h5py lacks.

    It had been dead since the PyTables migration and silently so, because
    the file open was wrapped in a bare `except: pass`. Removed rather than
    repaired -- either supported write mode can be hot started from.
    """
    import inspect

    from proteus import Archiver, FemTools

    assert not hasattr(Archiver.AR_base, "hdfFileGlb")
    for module in (Archiver, FemTools):
        source = inspect.getsource(module)
        # the explanatory docstring may name it; live code must not use it
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("*"):
                continue
            assert "get_node(" not in stripped, line
            assert "ar.hdfFileGlb" not in stripped, line


# --------------------------------------------------------------------------
# rebuilding the document from the .h5 alone
# --------------------------------------------------------------------------


def load_gather_times():
    """scripts/gatherTimes has no .py suffix, so import it by path."""
    import importlib.machinery
    import importlib.util

    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts", "gatherTimes")
    loader = importlib.machinery.SourceFileLoader("gatherTimes", path)
    module = importlib.util.module_from_spec(
        importlib.util.spec_from_loader("gatherTimes", loader))
    loader.exec_module(module)
    return module


def test_the_archive_records_which_layout_it_used(tmp_path):
    """A reader cannot infer the layout, so the writer records it.

    At one rank a global write and a per-subdomain write both leave
    metadata of shape (1,), but the first means a uniform step and the
    second a spatial collection holding one grid.
    """
    from proteus.Archiver import AR_base

    for global_sync in (True, False):
        datadir = run_ranks(tmp_path / str(global_sync), 1,
                            global_sync=global_sync)
        with h5py.File(os.path.join(datadir, "mpitest.h5"), "r") as f:
            recorded = int(f.attrs[AR_base.GLOBAL_SYNC_ATTR])
        assert recorded == (1 if global_sync else 0)


@pytest.mark.parametrize("global_sync", [True, False])
def test_the_domain_can_be_read_from_the_h5_alone(tmp_path, global_sync):
    """No .xmf or .ymf sidecar, and no flags supplied by the caller."""
    from proteus.Archiver import readArchiveDomain

    datadir = run_ranks(tmp_path, 1, global_sync=global_sync)
    os.remove(os.path.join(datadir, "mpitest.ymf"))
    domain = readArchiveDomain("mpitest", dataDir=datadir)
    (collection,) = archive_collections(domain)
    assert len(collection["data"]) == 2
    # the recorded layout decides the shape, not a caller's guess
    has_spatial = "spatial_collection" in collection["data"][0]
    assert has_spatial == (not global_sync)


def test_gather_times_rebuilds_both_sidecars_from_the_h5(tmp_path):
    """The recovery path: a run whose .ymf/.xmf was lost or never written.

    Also a regression test. Phase 2 changed the in-HDF5 metadata from XML
    fragments to YAML, and gatherTimes still called fromstring() on it --
    ParseError on every invocation. Nothing caught it because the scripts
    have no tests.
    """
    datadir = run_ranks(tmp_path, 1, global_sync=True)
    for suffix in (".ymf", ".xmf"):
        path = os.path.join(datadir, "mpitest" + suffix)
        if os.path.exists(path):
            os.remove(path)
    assert sorted(os.listdir(datadir)) == ["mpitest.h5"]

    load_gather_times().gatherTimes("mpitest", dataDir=datadir, tCount=-1)

    from ymf.archive import read_ymf

    domain, _ = read_ymf(os.path.join(datadir, "mpitest_complete.ymf"))
    assert len(archive_collections(domain)[0]["data"]) == 2
    from xml.etree.ElementTree import parse

    root = parse(os.path.join(datadir, "mpitest_complete.xmf")).getroot()
    assert root.tag == "Xdmf"
    assert len(root.find("Domain").find("Grid").findall("Grid")) == 2


def test_gather_times_can_truncate_while_rebuilding(tmp_path):
    datadir = run_ranks(tmp_path, 1, global_sync=True)
    load_gather_times().gatherTimes("mpitest", dataDir=datadir, tCount=1)

    from ymf.archive import read_ymf

    domain, _ = read_ymf(os.path.join(datadir, "mpitest_complete.ymf"))
    assert len(archive_collections(domain)[0]["data"]) == 1


def test_gather_times_needs_no_xmf_to_seed_from(tmp_path):
    """The previous version opened <name>.xmf for its Domain skeleton.

    So it could not recover an archive whose .xmf was the missing part --
    which is the case it exists for.
    """
    import ast
    import inspect

    # It does *write* an .xmf, so a text search for ".xmf" proves nothing.
    # What matters is that it never parses one: no ElementTree, no open().
    tree = ast.parse(inspect.getsource(load_gather_times().gatherTimes))
    called = {node.func.id for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "open" not in called, "gatherTimes still opens a file itself"
    assert "ElementTree" not in called, "gatherTimes still parses XML"


def test_h5toxmf_script_is_gone():
    """Deleted: uninstalled, unreferenced, PyTables-based, and duplicating
    a function extractSolution.py already defines internally. gatherTimes
    is the supported way to regenerate a document from an .h5."""
    scripts = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
    assert not os.path.exists(os.path.join(scripts, "H5toXMF.py"))
    assert os.path.exists(os.path.join(scripts, "gatherTimes"))


# --------------------------------------------------------------------------
# no .xmf on the write path (Phase 4), and hot start without one
# --------------------------------------------------------------------------


@pytest.mark.parametrize("global_sync", [True, False])
def test_an_archive_writes_no_xmf(tmp_path, global_sync):
    datadir = run_ranks(tmp_path, 1, global_sync=global_sync)
    assert sorted(os.listdir(datadir)) == ["mpitest.h5", "mpitest.ymf"]


@pytest.mark.parametrize("global_sync", [True, False])
def test_a_hot_start_continues_the_archive_from_the_h5_alone(tmp_path,
                                                             global_sync):
    """The Domain the writers append to comes from the .h5's own record.

    It used to come from parsing the .xmf, so an archive without one --
    every archive, now -- could not be hot started.
    """
    datadir = run_ranks(tmp_path, 1, global_sync=global_sync)
    run_ranks(tmp_path, 1, global_sync=global_sync, hot_start=True)
    (collection,) = archive_collections(read_domain(datadir))
    assert [step["time"] for step in collection["data"]] == [0.0, 0.5, 1.0]
    assert sorted(os.listdir(datadir)) == ["mpitest.h5", "mpitest.ymf"]


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
def test_a_parallel_hot_start_continues_the_archive(tmp_path):
    datadir = run_ranks(tmp_path, 2, global_sync=False)
    run_ranks(tmp_path, 2, global_sync=False, hot_start=True)
    (collection,) = archive_collections(read_domain(datadir))
    assert [step["time"] for step in collection["data"]] == [0.0, 0.5, 1.0]


@pytest.mark.parametrize("n_ranks", [1, pytest.param(
    2, marks=pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available"))])
@pytest.mark.parametrize("global_sync", [True, False])
def test_a_hot_start_from_an_earlier_step_replaces_the_later_ones(
        tmp_path, n_ranks, global_sync):
    """The requested time need not be the last one archived."""
    datadir = run_ranks(tmp_path, n_ranks, global_sync=global_sync)
    run_ranks(tmp_path, n_ranks, global_sync=global_sync, hot_start="back")
    (collection,) = archive_collections(read_domain(datadir))
    assert [step["time"] for step in collection["data"]] == [0.0, 1.0]


# --------------------------------------------------------------------------
# the document is current during the run, not only after a clean close
# --------------------------------------------------------------------------


def test_an_interrupted_run_still_leaves_a_readable_document(tmp_path):
    """The point of the incremental write.

    A run that is killed, times out on a scheduler or dies on one rank never
    reaches close(), and used to leave an archive with no .ymf at all: the
    metadata was in the HDF5 file but the document that describes it was
    assembled only at close. Recovering it meant a separate pass. Now the
    document is already there and already correct.
    """
    datadir = run_ranks(tmp_path, 1, global_sync=False, skip_close=True)
    (collection,) = archive_collections(read_domain(datadir))
    assert [s["time"] for s in collection["data"]] == [0.0, 0.5]


def test_what_it_looked_like_before(tmp_path):
    """The same interrupted run with the incremental write switched off.

    Kept as the control: without it there is no document, which is what
    makes the test above worth having rather than a tautology.
    """
    datadir = run_ranks(tmp_path, 1, global_sync=False, skip_close=True,
                        incremental=False)
    assert not os.path.exists(os.path.join(datadir, "mpitest.ymf"))
    assert os.path.exists(os.path.join(datadir, "mpitest.h5"))


def test_the_running_document_agrees_with_the_one_written_at_close(tmp_path):
    """Two independent paths to the same document, so they can be compared.

    The running document is appended to in memory from the grids master
    already holds; the one at close is re-assembled from the YAML in the
    HDF5 file. They are built by different code from different sources, and
    they have to agree -- if they ever diverge, the archive of record
    depends on whether the run finished, which is exactly the property this
    campaign exists to remove.
    """
    running = read_domain(run_ranks(tmp_path / "a", 1, global_sync=False,
                                    skip_close=True))
    closed = read_domain(run_ranks(tmp_path / "b", 1, global_sync=False))
    assert running == closed


@pytest.mark.parametrize("global_sync", [False, True])
def test_the_running_document_agrees_for_both_layouts(tmp_path, global_sync):
    running = read_domain(run_ranks(tmp_path / "a", 1, global_sync=global_sync,
                                    skip_close=True))
    closed = read_domain(run_ranks(tmp_path / "b", 1, global_sync=global_sync))
    assert running == closed


@pytest.mark.skipif(MPIEXEC is None, reason="no MPI launcher available")
@pytest.mark.parametrize("n_ranks", [2, 3])
def test_the_running_document_holds_every_ranks_grid(tmp_path, n_ranks):
    """Only master writes it, but it must describe the whole decomposition."""
    datadir = run_ranks(tmp_path, n_ranks, global_sync=False, skip_close=True)
    (collection,) = archive_collections(read_domain(datadir))
    assert len(collection["data"]) == 2
    for step in collection["data"]:
        subs = step["spatial_collection"]
        assert len(subs) == n_ranks
        assert [g["name"] for g in subs] == [
            "Grid_p%d" % r for r in range(n_ranks)]


def test_no_leftover_temporary_file(tmp_path):
    """The write is rename-based, so nothing may be left beside the target."""
    datadir = run_ranks(tmp_path, 1, global_sync=False, skip_close=True)
    leftovers = [f for f in os.listdir(datadir) if f.endswith(".writing")]
    assert leftovers == []


def test_the_running_document_is_appended_not_rewritten(tmp_path):
    """The file is a stream mid-run, which is what makes the write O(1).

    Rewriting the whole document to record a step is O(steps) per step and
    so quadratic over a run -- 35 ms a step at 500 steps, minutes over a
    long one. Appending writes one record whatever the archive holds. The
    observable difference is the shape of the file: a stream of documents
    while the run is going, collapsed to a single document at close.
    """
    datadir = run_ranks(tmp_path, 1, global_sync=False, skip_close=True)
    text = open(os.path.join(datadir, "mpitest.ymf"), encoding="utf-8").read()
    assert text.count("\n---\n") == 2        # one record per step
    assert text.count("\n...\n") == 2        # each one completed


def test_close_leaves_a_single_document(tmp_path):
    """A finished run is in the canonical shape, so nothing downstream has
    to care that the live form exists."""
    datadir = run_ranks(tmp_path, 1, global_sync=False)
    text = open(os.path.join(datadir, "mpitest.ymf"), encoding="utf-8").read()
    assert "\n---\n" not in text


def test_an_ymf_without_append_step_costs_the_document_not_the_run(tmp_path):
    """proteus and ymf move independently, so the skew has to be survivable.

    ``append_step`` arrived in ymf after this writer was written, and the
    import of it used to sit outside the guard -- which turned "your live
    document is stale" into an exception out of the middle of a timestep.
    CI caught it honestly: three legs of one run resolved ``ymf@main`` five
    minutes before the merge that added the function. Any user whose ymf
    predates it would have hit the same thing.
    """
    import types
    import proteus.Archiver as AR
    import ymf.archive

    live = types.SimpleNamespace(dataDir=str(tmp_path), filename="skew",
                                 global_sync=True, _liveYmfStarted=False,
                                 _liveYmfWarned=False)
    step = [(0, "Grid", 0.0, [{"topology": {}, "geometry": {}}])]

    saved = ymf.archive.append_step
    del ymf.archive.append_step
    try:
        AR.AR_base._appendLiveStep(live, step)   # must not raise
    finally:
        ymf.archive.append_step = saved

    assert live._liveYmfWarned, "the failure has to be reported, not hidden"


# --------------------------------------------------------------------------
# the modes: one writer, switches, four names
# --------------------------------------------------------------------------


def test_the_modes_are_combinations_of_the_switches():
    from proteus.Archiver import AR_base

    assert AR_base.resolve_mode() == AR_base.resolve_mode("split")
    assert AR_base.resolve_mode("without-a-net")["incremental"] is False
    assert AR_base.resolve_mode("kitchen-sink")["includeLog"] is True
    assert AR_base.resolve_mode("ymf-only")["inline"] is True
    # an explicit switch wins over the mode
    assert AR_base.resolve_mode("without-a-net", incremental=True)["incremental"] is True
    assert AR_base.resolve_mode("split", driver="serial")["driver"] == "serial"
    with pytest.raises(ValueError, match="archive mode 'belt-and-braces'"):
        AR_base.resolve_mode("belt-and-braces")
    with pytest.raises(ValueError, match="archive driver 'core'"):
        AR_base.resolve_mode(driver="core")


def datasets_of(datadir):
    """Every array in the archive's .h5, by name, for comparing layouts."""
    with h5py.File(os.path.join(datadir, "mpitest.h5"), "r") as f:
        return {k: f[k][()] for k in f if f[k].dtype.kind != "S"}


@pytest.mark.parametrize("n_ranks", [1, 2, 3])
@pytest.mark.parametrize("global_sync", [True, False])
def test_the_serial_driver_writes_the_file_the_mpio_driver_writes(tmp_path, n_ranks, global_sync):
    """Master alone writes, the others send; the archive is the same."""
    mpio = run_ranks(tmp_path, n_ranks, global_sync, driver="mpio", name="mpio")
    serial = run_ranks(tmp_path, n_ranks, global_sync, driver="serial", name="serial")
    a, b = datasets_of(mpio), datasets_of(serial)
    assert sorted(a) == sorted(b)
    for name in a:
        np.testing.assert_array_equal(a[name], b[name], err_msg=name)
    from ymf.archive import read_ymf
    assert read_ymf(os.path.join(mpio, "mpitest.ymf"))[0] == \
        read_ymf(os.path.join(serial, "mpitest.ymf"))[0]


def test_a_serial_driver_archive_cannot_be_hot_started_in_parallel(tmp_path):
    run_ranks(tmp_path, 2, global_sync=True, driver="serial")
    with pytest.raises(AssertionError, match="written by master alone"):
        run_ranks(tmp_path, 2, global_sync=True, driver="serial", hot_start=True)


def test_without_a_net_is_the_h5_alone_until_close(tmp_path):
    """Mode 1: nothing but the .h5 during the run; the .ymf derived at close.

    And derivable again: the document close writes is, byte for byte,
    the one readArchiveDomain rebuilds from the .h5, which is the net.
    """
    from ymf.archive import write_ymf
    from proteus.Archiver import readArchiveDomain

    datadir = run_ranks(tmp_path, 2, global_sync=False, mode="without-a-net",
                        skip_close=True, name="killed")
    assert not os.path.exists(os.path.join(datadir, "mpitest.ymf"))
    datadir = run_ranks(tmp_path, 2, global_sync=False, mode="without-a-net")
    written = open(os.path.join(datadir, "mpitest.ymf")).read()
    rebuilt_path = tmp_path / "rebuilt.ymf"
    write_ymf(readArchiveDomain("mpitest", datadir), rebuilt_path,
              extra=_extra_of(written))
    assert rebuilt_path.read_text() == written


def _extra_of(text):
    import yaml
    return yaml.safe_load(text).get("extra")


def test_the_kitchen_sink_carries_the_log_and_the_provenance(tmp_path):
    from ymf.archive import read_ymf

    datadir = run_ranks(tmp_path, 2, global_sync=True, mode="kitchen-sink", with_log=True)
    _, extra = read_ymf(os.path.join(datadir, "mpitest.ymf"))
    provenance = extra["proteus"]
    assert provenance["archive_mode"] == "kitchen-sink"
    assert provenance["ranks"] == 2
    assert "command" in provenance and "version" in provenance
    assert provenance["log_file"] == "mpitest.log"
    assert "a line of the run's log" in provenance["log"]


def test_split_carries_the_provenance_but_not_the_log(tmp_path):
    from ymf.archive import read_ymf

    datadir = run_ranks(tmp_path, 1, global_sync=True, with_log=True)
    _, extra = read_ymf(os.path.join(datadir, "mpitest.ymf"))
    assert extra["proteus"]["archive_mode"] == "split"
    # neither the log nor the command line: two runs of the same
    # computation (pro run and parun, say) leave identical documents
    assert "log" not in extra["proteus"] and "command" not in extra["proteus"]


def test_the_running_document_already_names_the_run(tmp_path):
    """The header written at the first step carries the provenance, so a
    killed run's document still says what wrote it."""
    from ymf.archive import read_ymf

    datadir = run_ranks(tmp_path, 1, global_sync=True, skip_close=True)
    _, extra = read_ymf(os.path.join(datadir, "mpitest.ymf"))
    assert extra["proteus"]["ranks"] == 1


@pytest.mark.parametrize("n_ranks", [1, 2])
@pytest.mark.parametrize("global_sync", [True, False])
def test_ymf_only_is_one_self_contained_file(tmp_path, n_ranks, global_sync):
    """Mode 4: no .h5; every array is in the document, and equals what the
    mpio archive wrote to its .h5."""
    from ymf.archive import load_array, read_ymf

    reference = datasets_of(run_ranks(tmp_path, n_ranks, global_sync, name="h5"))
    datadir = run_ranks(tmp_path, n_ranks, global_sync, mode="ymf-only", name="inline")
    assert sorted(os.listdir(datadir)) == ["mpitest.ymf"]
    domain, extra = read_ymf(os.path.join(datadir, "mpitest.ymf"))
    (collection,) = archive_collections(domain)
    assert len(collection["data"]) == 2
    seen = 0
    for step in collection["data"]:
        for grid in step.get("spatial_collection", [step]):
            for owner in [grid["topology"], grid["geometry"]] + grid["attributes"]:
                item = owner["data_item"]
                assert item["format"] == "XML" and "values" in item and "data" not in item
            # the field of this grid is sized by its geometry
            n_nodes = grid["geometry"]["data_item"]["dimensions"][0]
            u = load_array(grid["attributes"][0]["data_item"])
            assert u.shape == (n_nodes,)
            seen += 1
    assert seen == 2 * (1 if global_sync else n_ranks)
    # the same values as the .h5 archive: compare the field at the last step
    last = collection["data"][-1]
    grids = last.get("spatial_collection", [last])
    for r, grid in enumerate(grids):
        key = "u_t1" if global_sync else "u_p%d_t1" % r
        np.testing.assert_array_equal(load_array(grid["attributes"][0]["data_item"]),
                                      reference[key])


def test_ymf_only_streams_its_steps_too(tmp_path):
    """The inline document is appended a step at a time like any other."""
    from ymf.archive import read_ymf

    datadir = run_ranks(tmp_path, 1, global_sync=True, mode="ymf-only", skip_close=True)
    domain, _ = read_ymf(os.path.join(datadir, "mpitest.ymf"))
    assert len(archive_collections(domain)[0]["data"]) == 2
    assert "values" in archive_collections(domain)[0]["data"][1]["geometry"]["data_item"]
