"""The partitioner marks the faces it cut with the reserved flag -1.

A rank's exterior faces are the real boundary of the domain plus the faces cut
by the partition. The domain's flags are copied to the subdomain, and a face
interior to the global mesh carries 0 there, so a cut face used to look exactly
like a real boundary face the domain left untagged. Now the cut faces carry -1
(reserved for the partitioner), 0 means "no tag", and positive flags are the
domain's conditions.
"""
import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.join(HERE, "rank_interface_flags_mpi.py")


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ("mpiexec", "mpirun"):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which("mpiexec") or shutil.which("mpirun")


MPIEXEC = _mpiexec()


def _run(ranks, cwd, pinned=True, **env_overrides):
    env = dict(os.environ, OMP_NUM_THREADS="1")
    env.pop("PETSC_OPTIONS", None)
    if pinned:
        # the contiguous initial partition: reproducible, and node numbers that
        # stay what they were. Unpinned, the graph partitioner renumbers.
        env["PETSC_OPTIONS"] = "-mat_partitioning_type current"
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env.update({k: str(v) for k, v in env_overrides.items()})
    # even for one rank (test_archive_parallel_metadata.run_ranks says why)
    argv = [sys.executable, HELPER]
    if MPIEXEC:
        argv = [MPIEXEC, "-n", str(ranks)] + argv
    out = subprocess.run(argv, capture_output=True, text=True, timeout=600, env=env,
                         cwd=str(cwd))
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    rows = []
    for line in out.stdout.splitlines():
        if line.startswith("RANK"):
            row = dict(item.split("=") for item in line.split()[1:])
            row["tags"] = {int(k): int(v) for k, v in
                           (kv.split(":") for kv in row["tags"].split(","))}
            rows.append(row)
    return rows


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("part,layers", [("element", 0), ("element", 1), ("node", 1)])
@pytest.mark.parametrize("shape", ["tri", "tet"])
@pytest.mark.parametrize("tags", ["real", "zero"])
@pytest.mark.parametrize("pinned", [True, False], ids=["pinned", "graph"])
def test_cut_faces_are_minus_one_and_real_boundary_keeps_its_flag(part, layers, shape, tags,
                                                                  pinned, tmp_path):
    rows = _run(2, tmp_path, pinned=pinned, PART_TYPE=part, NLAYERS=layers, SHAPE=shape,
                TAGS=tags)
    assert len(rows) == 2
    for row in rows:
        assert int(row["wrong"]) == 0, row
        assert row["tags"].get(-1, 0) > 0, row            # there is a cut
        if tags == "zero":
            assert set(row["tags"]) <= {-1, 0}, row        # the untagged boundary is still 0
        else:
            assert 0 not in row["tags"], row               # every real face keeps 1..4


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
def test_one_rank_has_no_cut_faces(tmp_path):
    (row,) = _run(1, tmp_path, PART_TYPE="element", NLAYERS=0, SHAPE="tri", TAGS="zero")
    assert row["tags"] == {0: int(row["exterior"])}
