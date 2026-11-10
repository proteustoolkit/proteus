"""Per-rank records in the archive: each rank's own named arrays in a fixed number of datasets.

The solver state a hot start restores (NumericalSolution.archiveSolverState) is rank-local: arrays at the
quadrature points of each rank's elements. It used to be written one dataset per rank and item
(``<item>_p<rank>``, through ``create_dataset_async``), and under the mpio driver every rank creates every
rank's datasets: ranks x items collective creates per model and archived step. On proteus main 266eb974 the
Torino flume runs died at the first archive write, inside H5Dcreate2, at 1,536 ranks (Carpenter, HDF5 1.14.6:
an assertion in the collective metadata cache, ``H5C_construct_candidate_list__min_clean``) and at 4,096 ranks
(LONI QB: "could not determine which branch to follow out of this node"); 1.9.x, which has no solver state
archive, ran the same cases. On a machine where it does not crash the cost is quadratic in the rank count: the
old layout's creates (60 items, 3 models, 2 steps) took 5.1 s at 96 ranks and 19.4 s at 192 (ceds-model,
HDF5 2.2.0).

``Archiver.create_rank_records`` writes the ranks' arrays as three datasets whatever the number of ranks and
items. These tests run it under ``mpiexec`` at 1 to 3 ranks with every driver, arrays whose shape, number of
dimensions and type differ from rank to rank (and an empty rank), and check that every rank reads its own
arrays back, and that the number of datasets does not depend on the number of ranks (the old layout's grew
with it: the check that fails before the change).
"""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("proteus.Archiver")

from test_archive_parallel_metadata import MPIEXEC  # noqa: E402  (the launcher beside this interpreter)

RANK_SCRIPT = textwrap.dedent(
    '''
    import os, shutil
    import numpy as np
    from proteus import Comm
    Comm.init()
    comm = Comm.get()
    rank, size = comm.rank(), comm.size()
    datadir = os.environ["DATADIR"]
    driver = os.environ["DRIVER"]
    if rank == 0:
        shutil.rmtree(datadir, ignore_errors=True)
        os.makedirs(datadir)
    comm.barrier()
    from proteus import Archiver

    def records(step):
        """What each rank writes: shapes, dimensions and types that differ from rank to rank."""
        n = 3 * rank                         # rank 0 holds no elements
        return [("fingerprint", np.array([size, rank, 7 * rank + step], dtype="int64")),
                ("shockCapturing/a.numDiff_last", np.arange(n * 4, dtype="d").reshape(n, 4) + step + 0.5),
                ("shockCapturing/s.lag", np.atleast_1d(np.asarray(rank % 2 == 0))),
                # an int on rank 0, a float elsewhere: promoted to one type
                ("timeIntegration/s.dt", np.atleast_1d(np.asarray(1 if rank == 0 else 0.25 * rank))),
                # one dimension on rank 0, three elsewhere
                ("coefficients/a.q_p", np.zeros(0) if rank == 0 else np.full((rank, 2, 3), float(rank + step))),
                ("policy", np.frombuffer(b'{"DT": 0.01}', dtype="uint8"))]

    ar = Archiver.XdmfArchive(datadir, "records", driver=driver)
    for step in (0, 1):   # the second write replaces the first in place
        assert ar.create_rank_records("solver_state/m/records", records(step))
    # the ranks disagree on the keys: nothing is written, on every rank
    disagree = [("a", np.zeros(1))] if rank == 0 else [("b", np.zeros(1))]
    assert ar.create_rank_records("solver_state/m/disagree", disagree) is (size == 1)
    ar.close()
    comm.barrier()

    # read back, with the driver a hot start reads it with (a serial file is read through mpio)
    reader = Archiver.XdmfArchive(datadir, "records", readOnly=True,
                                  driver="per-rank" if driver == "per-rank" else "mpio")
    got = reader.read_rank_records("solver_state/m/records")
    want = records(1)
    assert [k for k, _ in got] == [k for k, _ in want], got
    for (key, a), (_, b) in zip(got, want):
        assert a.shape == b.shape, (key, a.shape, b.shape)
        np.testing.assert_array_equal(a, b)
    assert got[3][1].dtype == np.float64 if size > 1 else got[3][1].dtype.kind == "i"
    assert reader.read_rank_records("solver_state/m/absent") is None
    reader.close()
    comm.barrier()
    if rank == 0:
        print("OBSERVED_SIZE=%d" % size)
    '''
)


def run_ranks(tmp_path, n_ranks, driver):
    datadir = str(tmp_path / ("%s_%d" % (driver, n_ranks)))
    env = dict(os.environ, DATADIR=datadir, DRIVER=driver)
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    cmd = [MPIEXEC, "-n", str(n_ranks), sys.executable, "-c", RANK_SCRIPT]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    assert result.returncode == 0, "%d ranks, %s:\n%s\n%s" % (n_ranks, driver, result.stdout[-2000:],
                                                             result.stderr[-4000:])
    assert "OBSERVED_SIZE=%d" % n_ranks in result.stdout, result.stdout[-2000:]
    return datadir


def datasets(path):
    """The datasets in an HDF5 file, by name."""
    found = []
    with h5py.File(path, "r") as f:
        f.visititems(lambda name, obj: found.append(name) if isinstance(obj, h5py.Dataset) else None)
    return sorted(found)


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("driver", ["mpio", "serial", "per-rank"])
def test_every_rank_reads_its_own_records_back(tmp_path, driver):
    for n_ranks in (1, 3):
        run_ranks(tmp_path, n_ranks, driver)


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("driver", ["mpio", "serial"])
def test_the_number_of_datasets_does_not_grow_with_the_ranks(tmp_path, driver):
    """Three datasets for the records, on 1 rank as on 3 (one dataset per rank and item was 6 x ranks)."""
    names = {}
    for n_ranks in (1, 3):
        datadir = run_ranks(tmp_path, n_ranks, driver)
        names[n_ranks] = [n for n in datasets(os.path.join(datadir, "records.h5")) if n.startswith("solver_state/m/records")]
    assert names[1] == names[3] == ["solver_state/m/records", "solver_state/m/records.index",
                                    "solver_state/m/records.keys"], names
    with h5py.File(os.path.join(tmp_path, "%s_3" % driver, "records.h5"), "r") as f:
        index = f["solver_state/m/records.index"][...]
        assert index.shape[0] == 3 and list(index[:, 0]) == [0, 1, 2]
        # one rank after the other, rank 0 first
        assert list(index[1:, 1]) == list(np.cumsum(index[:-1, 2]))
