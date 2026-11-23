"""``parun -F`` on a stored tetgen mesh tests the files, not ``triangleOptions`` (C7.1).

``_generateMesh`` (the PiecewiseLinearComplexDomain branch) asked ``meshOptions.triangleOptions``
whether the mesh had been made with the ``f`` and ``ee`` flags. A deck that reads the mesh it finds
(genMesh False) sets that string only where it generates, so the check raised ``TypeError`` (unset) or
exited with "Remake the mesh with the f flag" (a string without ``f``) on a mesh whose files were fine,
and passed a string that said ``feen`` over a mesh whose ``.edge`` file was gone (then a cython error).
Now the files decide: a missing ``.edge`` (the one file ``_generateMesh`` does not regenerate a mesh for;
a missing ``.node``, ``.ele`` or ``.face`` makes rank 0 run tetgen with the deck's options, as without -F)
exits on every rank and names the file; ``triangleOptions`` do not matter.

minus_f_stored_mesh_mpi.py meshes the stored mesh of test/ci/meshPartition (131 nodes, 441 tetrahedra,
made by ``tetgen -Vfeen``) on every rank, with -F and in memory. The driver starts its own ranks, so
these tests skip under a multi-rank launch.
"""
import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
STORED = os.path.join(HERE, "..", "ci", "meshPartition")
EXTENSIONS = ("poly", "node", "ele", "face", "edge")
NODES, ELEMENTS = 131, 441


def _launcher_env():
    env = dict((k, v) for k, v in os.environ.items() if not k.startswith(("PMI_", "PMIX_", "HYDRA_", "OMPI_COMM_WORLD")))
    env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env.setdefault("FI_PROVIDER", "tcp")
    env["OMP_NUM_THREADS"] = "1"
    # the contiguous default partition, which makes -F and the in-memory partition comparable
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


def _stage(tmp_path, name, drop=()):
    """The stored mesh as ``<name>.*`` in its own directory, without the extensions in ``drop``."""
    cwd = tmp_path / name
    cwd.mkdir()
    for ext in EXTENSIONS:
        if ext not in drop:
            shutil.copy(os.path.join(STORED, "plc." + ext), str(cwd / ("plc." + ext)))
    return cwd


def _run(cwd, ranks, options, mode):
    result = subprocess.run([_mpiexec(), "-n", str(ranks), sys.executable,
                             os.path.join(HERE, "minus_f_stored_mesh_mpi.py"), "plc", options, mode],
                            capture_output=True, text=True, env=_launcher_env(), cwd=str(cwd), timeout=600)
    lines = [l for l in result.stdout.splitlines() if l.startswith("RESULT")]
    return result, lines


def _fields(line):
    return dict(kv.split("=") for kv in line.split()[1:])


def _sum(lines, key):
    return sum(int(_fields(l)[key]) for l in lines)


@pytest.mark.parametrize("ranks", [1, 2])
@pytest.mark.parametrize("options", ["none", "pq1.2", "feen"])
def test_f_partitions_the_stored_files_whatever_triangle_options_say(tmp_path, ranks, options):
    """Unset, a string without ``f`` and ``ee``, and one with them: the same partition as the in-memory one."""
    cwd = _stage(tmp_path, "mesh")
    result, lines = _run(cwd, ranks, options, "F")
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert len(lines) == ranks
    # the branch under test ran: the file check, then the partitioner that reads the files
    assert all(_fields(l)["req"] == "1" and _fields(l)["files"] == "1" for l in lines), lines
    assert _sum(lines, "ownedNodes") == NODES and _sum(lines, "ownedElements") == ELEMENTS
    assert all(int(_fields(l)["globalNodes"]) == NODES and int(_fields(l)["globalElements"]) == ELEMENTS for l in lines)
    # against the mesh read whole and partitioned in memory, rank by rank: the same nodes per rank (the
    # two paths own elements by different rules, so only their total is compared, above)
    reference, ref_lines = _run(_stage(tmp_path, "reference"), ranks, options, "memory")
    assert reference.returncode == 0, reference.stdout[-3000:] + reference.stderr[-3000:]
    assert all(_fields(l)["req"] == "0" and _fields(l)["files"] == "0" for l in ref_lines), ref_lines
    key = lambda l: int(_fields(l)["rank"])
    for line, ref in zip(sorted(lines, key=key), sorted(ref_lines, key=key)):
        for field in ("ownedNodes", "globalNodes", "globalElements"):
            assert _fields(line)[field] == _fields(ref)[field], (line, ref)


@pytest.mark.parametrize("ranks", [1, 2])
@pytest.mark.parametrize("options", ["feen", "none"])
def test_a_missing_edge_file_exits_on_every_rank_and_is_named(tmp_path, ranks, options):
    """Even when triangleOptions claim ``feen``: the files are what is tested (it was a cython error)."""
    cwd = _stage(tmp_path, "mesh", drop=("edge",))
    result, lines = _run(cwd, ranks, options, "F")
    assert result.returncode != 0
    assert len(lines) == ranks and all("exit=" in l for l in lines), lines
    assert all("missing plc.edge" in l for l in lines), lines
    assert "tetgen -Vfeen plc.poly" in lines[0]


def test_an_edge_file_without_markers_is_read_with_flag_zero(tmp_path):
    """The reader takes the ``.edge`` markers when its header announces them and flags 0 otherwise, so the
    content past the header does not matter to it: only the file's presence is required."""
    cwd = _stage(tmp_path, "mesh")
    edge = cwd / "plc.edge"
    rows = edge.read_text().splitlines()
    edge.write_text("\n".join(["%s 0" % rows[0].split()[0]] + [" ".join(r.split()[:3]) for r in rows[1:]]) + "\n")
    result, lines = _run(cwd, 2, "none", "F")
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert _sum(lines, "ownedNodes") == NODES and _sum(lines, "ownedElements") == ELEMENTS


def test_the_file_problems_are_listed_in_one_message(tmp_path):
    from proteus import MeshTools
    cwd = _stage(tmp_path, "mesh", drop=("face", "edge"))
    prefix = str(cwd / "plc")
    assert MeshTools._tetgenFileProblems(prefix) == ["missing %s.face" % prefix, "missing %s.edge" % prefix]
    assert MeshTools._tetgenFileProblems(str(tmp_path / "mesh" / "other")) == [
        "missing %s/mesh/other.%s" % (tmp_path, ext) for ext in ("node", "ele", "face", "edge")]
