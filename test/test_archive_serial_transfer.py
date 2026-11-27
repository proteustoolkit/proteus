"""The serial driver brings the ranks' arrays to master one rank and one piece at a time.

With the serial driver only master holds the file, and the archive used to bring every rank's array to it with
one pickled ``gather``. For the solver state (``create_rank_records``) of bathyduck at 144 ranks that is a
message of more than 2 GiB, which Intel MPI 2021's 32-bit counts refuse ("Invalid argument", Mike 954316 and
QB 1202709), and every rank's array on master at once plus the copy it assembles (318 of 335 GB on a
Carpenter node). ``AR_base._rows_to_master`` sends each rank's array in pieces of at most
``TRANSFER_CHUNK_BYTES`` into one reused buffer on master, which writes them where they belong.

These tests run the three writers that took the gather (``create_dataset_async``, ``create_dataset_sync``,
``create_rank_records``) under ``mpiexec`` at 1 and 3 ranks with a 40-byte piece (rows larger than a piece, an
empty rank, a 0-d array, 1-D to 3-D arrays, a type that differs from rank to rank) and check that the serial
file holds the same bytes as the mpio file and as the serial file written with the default piece, and that an
inline document (master keeps the arrays) gets the same arrays.
"""
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
    """The launcher beside this interpreter (a mismatched one runs N single-rank jobs), else the one on PATH."""
    beside = os.path.join(os.path.dirname(sys.executable), "mpiexec")
    return beside if os.path.exists(beside) else shutil.which("mpiexec")


MPIEXEC = _find_mpiexec()

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
    inline = os.environ.get("INLINE") == "1"
    if rank == 0:
        shutil.rmtree(datadir, ignore_errors=True)
        os.makedirs(datadir)
    comm.barrier()
    from proteus import Archiver
    if os.environ.get("CHUNK"):
        Archiver.AR_base.TRANSFER_CHUNK_BYTES = int(os.environ["CHUNK"])

    def owned(r):
        """Rank r's slice of the global array: rank 1 owns nothing; rows of 3 x 2 doubles (48 bytes, more than a piece)."""
        n = 0 if r == 1 else 2 * r + 3
        return n, np.arange(n * 6, dtype="d").reshape(n, 3, 2) + 100.0 * r

    def mine_async():
        n = 4 * rank + 1
        return [("nodes_p%d" % rank, np.linspace(0.0, 1.0, 3 * n).reshape(n, 3) + rank),
                ("elements_p%d" % rank, (np.arange(5 * rank, dtype="int32") * 7)),   # empty on rank 0
                # 0-d (the mpio path's ``dataset[:] = data`` cannot write a 0-d array: (1,) there)
                ("scalar_p%d" % rank, np.array(2.5 * rank + 1) if driver != "mpio" else np.array([2.5 * rank + 1])),
                ("flags_p%d" % rank, np.arange(n, dtype="int8"))]

    ar = (Archiver.XdmfArchive(datadir, "transfer", inline=True) if inline else
          Archiver.XdmfArchive(datadir, "transfer", driver=driver))
    for name, a in mine_async():
        ar.create_dataset_async(name, data=a)
    counts = [owned(r)[0] for r in range(size)]
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype("int64")
    ar.create_dataset_sync("global_q", offsets, owned(rank)[1])
    n = 3 * rank
    records = [("fingerprint", np.array([size, rank], dtype="int64")),
               ("a.q", np.arange(n * 4, dtype="d").reshape(n, 4) + 0.5),
               ("s.dt", np.atleast_1d(np.asarray(1 if rank == 0 else 0.25 * rank))),
               ("policy", np.frombuffer(b'{"DT": 0.01}', dtype="uint8"))]
    assert ar.create_rank_records("solver_state/m/records", records)
    if inline and rank == 0:
        got = ar._inlineArrays
        want = np.concatenate([owned(r)[1] for r in range(size)])
        np.testing.assert_array_equal(got["global_q"], want)
        assert got["scalar_p0"].shape == () and float(got["scalar_p0"]) == 1.0
        for r in range(size):
            n_r = 4 * r + 1
            np.testing.assert_array_equal(got["nodes_p%d" % r], np.linspace(0.0, 1.0, 3 * n_r).reshape(n_r, 3) + r)
            np.testing.assert_array_equal(got["elements_p%d" % r], np.arange(5 * r, dtype="int32") * 7)
            assert float(got["scalar_p%d" % r]) == 2.5 * r + 1
        index = got["solver_state/m/records.index"]
        assert got["solver_state/m/records"].size == index[:, 2].sum()
        print("INLINE_CHECKED")
    ar.close()
    comm.barrier()
    if rank == 0:
        print("OBSERVED_SIZE=%d" % size)
    '''
)


def run_ranks(tmp_path, n_ranks, driver, chunk=None, inline=False):
    datadir = str(tmp_path / ("%s_%d_%s_%d" % (driver, n_ranks, chunk, inline)))
    env = dict(os.environ, DATADIR=datadir, DRIVER=driver, INLINE="1" if inline else "0")
    if chunk:
        env["CHUNK"] = str(chunk)
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    cmd = [MPIEXEC, "-n", str(n_ranks), sys.executable, "-c", RANK_SCRIPT]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    assert result.returncode == 0, "%d ranks, %s:\n%s\n%s" % (n_ranks, driver, result.stdout[-2000:],
                                                             result.stderr[-4000:])
    assert "OBSERVED_SIZE=%d" % n_ranks in result.stdout, result.stdout[-2000:]
    if inline:
        assert "INLINE_CHECKED" in result.stdout
    return datadir


def contents(datadir):
    """Every dataset of the archive's .h5: name -> (dtype, shape, bytes)."""
    found = {}
    with h5py.File(os.path.join(datadir, "transfer.h5"), "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                a = obj[()]
                found[name] = (obj.dtype.str, obj.shape, np.asarray(a).tobytes())
        f.visititems(visit)
    return found


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("n_ranks", [1, 3])
def test_the_serial_file_holds_what_the_mpio_file_holds_whatever_the_piece(tmp_path, n_ranks):
    mpio = contents(run_ranks(tmp_path, n_ranks, "mpio"))
    serial = contents(run_ranks(tmp_path, n_ranks, "serial"))
    pieces = contents(run_ranks(tmp_path, n_ranks, "serial", chunk=40))
    assert sorted(mpio) == sorted(serial) == sorted(pieces)
    assert "global_q" in mpio and "solver_state/m/records" in mpio and "scalar_p%d" % (n_ranks - 1) in mpio
    for name in mpio:
        if name.startswith("scalar_p"):
            assert serial[name][1] == pieces[name][1] == () and mpio[name][1] == (1,)
            assert serial[name][2] == pieces[name][2] == mpio[name][2], name
            continue
        assert serial[name] == mpio[name], name
        assert pieces[name] == mpio[name], name


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
def test_an_inline_document_gets_every_rank_s_arrays_in_pieces(tmp_path):
    run_ranks(tmp_path, 3, "serial", chunk=40, inline=True)
