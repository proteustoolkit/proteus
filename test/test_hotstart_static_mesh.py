"""A hot start from a frame that names an earlier frame's mesh reproduces the run (C1 static mesh, 2026-11-24).

A static mesh is archived once (frame 0) and every later frame's grid names those arrays. These runs archive the 2D
and 3D dam breaks of test/TwoPhaseFlow (five models: RANS2P, VOF, NCLS, RDLS, MCorr, as bathyduck and 45DEG_R1) on 1
and 2 ranks: once uninterrupted, and once stopped at frame 2 (whose fields name frame 0's mesh; the solver state is
written there, at the final step) and hot started from it to the end. The last frame equals the uninterrupted run's,
every frame of both archives resolves to arrays that exist, the uninterrupted archive holds one copy of the mesh and
the hot-started one two (the first frame after a hot start writes it again: the new process has no record of what
the file holds). A hot start from frame 2 of the uninterrupted archive would restart the history instead: the state
is one section, rewritten in place, and holds the last write only.
"""
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

HERE = os.path.dirname(os.path.abspath(__file__))
DECKS = os.path.join(HERE, "TwoPhaseFlow")
CASES = {
    "2d": ("damBreakLevelSet.py", "he=0.1 genMesh=True"),
    "3d": ("damBreakLevelSet3D.py", "he=0.4"),
}
FINAL, STEP = 0.04, 0.01


def _mpiexec():
    beside = os.path.join(os.path.dirname(sys.executable), "mpiexec")
    return beside if os.path.exists(beside) else shutil.which("mpiexec")


def _parun(cwd, ranks, deck, options, *extra, final=FINAL):
    parun = os.path.join(HERE, "..", "scripts", "parun")
    cmd = [sys.executable, parun, "--TwoPhaseFlow", deck, "-l1"] + list(extra) + [
        "-C", "final_time=%r dt_output=%g %s" % (final, STEP, options)]
    mpiexec = _mpiexec()
    if mpiexec is None:
        pytest.skip("needs mpiexec")
    cmd = [mpiexec, "-n", str(ranks)] + cmd
    # the partition pinned (the stop, hot and uninterrupted runs must partition alike for the state to be read)
    env = dict(os.environ, OMP_NUM_THREADS="1", PETSC_OPTIONS="-mat_partitioning_type current")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    ran = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True, timeout=900)
    assert ran.returncode == 0, ran.stdout[-3000:] + ran.stderr[-3000:]


def _times(h5):
    with h5py.File(h5, "r") as f:
        return [float(f[k].attrs["Time"]) for k in sorted(
            (k for k in f if re.match(r"^Mesh_Spatial_Domain_\d+$", k)), key=lambda k: int(k.rsplit("_", 1)[1]))]


def _spatial_mesh_steps(h5):
    """{family: steps} of the spatial mesh's connectivity and node datasets."""
    found = {}
    with h5py.File(h5, "r") as f:
        for k in f:
            m = re.match(r"^(elements|nodes)(\d*)Spatial_Domain(\d+)$", k)
            if m and isinstance(f.get(k, getlink=True), h5py.SoftLink):
                # a step that reuses an earlier step's array has a link by its own name, to that array
                assert f[k].shape == f[f.get(k, getlink=True).path].shape, k
            elif m:
                found.setdefault(m.group(1) + m.group(2), []).append(int(m.group(3)))
    return {k: sorted(v) for k, v in found.items()}


def _check_references(datadir, name):
    from ymf.archive import load_array
    from proteus.Archiver import archive_collections, readArchiveDomain
    domain = readArchiveDomain(name, str(datadir))
    for collection in archive_collections(domain):
        for step in collection["data"]:
            for grid in step.get("spatial_collection", [step]):
                for owner in [grid["topology"], grid["geometry"]] + grid["attributes"]:
                    item = owner["data_item"]
                    assert load_array(item, str(datadir)).size == np.prod(item["dimensions"]), item["data"]


@pytest.mark.parametrize("ranks", [1, 2])
@pytest.mark.parametrize("case", sorted(CASES))
def test_a_run_stopped_at_frame_2_of_a_static_mesh_and_hot_started_reproduces_the_run(case, ranks, tmp_path):
    deck, options = CASES[case]
    full, hot = tmp_path / "full", tmp_path / "hot"
    full.mkdir()
    shutil.copy(os.path.join(DECKS, deck), full)
    _parun(full, ranks, deck, options)
    name = deck[:-3]
    h5 = "%s.h5" % name
    times = _times(full / h5)
    last = len(times) - 1
    assert last >= 4, times
    assert all(steps == [0] for steps in _spatial_mesh_steps(full / h5).values()), _spatial_mesh_steps(full / h5)
    _check_references(full, name)
    hot.mkdir()
    shutil.copy(os.path.join(DECKS, deck), hot)
    _parun(hot, ranks, deck, options, final=times[2])
    assert _times(hot / h5) == times[:3]
    _parun(hot, ranks, deck, options, "-H")
    logs = "".join(open(os.path.join(hot, f)).read() for f in os.listdir(hot) if f.endswith(".log"))
    assert "no solver state to restore" not in logs
    assert _times(hot / h5) == times
    # the first frame after the hot start (3) writes the mesh again; frames 0-2 still name frame 0's
    assert all(steps == [0, 3] for steps in _spatial_mesh_steps(hot / h5).values()), _spatial_mesh_steps(hot / h5)
    _check_references(hot, name)
    with h5py.File(hot / h5, "r") as f:                  # the per-step names readers build themselves resolve
        for k in range(len(times)):
            assert "nodesSpatial_Domain%d" % k in f and "elementsSpatial_Domain%d" % k in f, k
    with h5py.File(full / h5, "r") as a, h5py.File(hot / h5, "r") as b:
        fields = [k for k in a if k.endswith("_t%d" % last) and k in b and a[k].dtype.kind == "f"
                  and not k.startswith(("nodeMaterial", "elementMaterial"))]
        assert len(fields) >= 5, sorted(a)
        for k in fields:
            x, y = np.asarray(a[k]).ravel(), np.asarray(b[k]).ravel()
            assert np.linalg.norm(x - y) <= 1e-10 * max(np.linalg.norm(x), 1e-300), k
