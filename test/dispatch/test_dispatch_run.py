"""C5 step 2: a split discretization runs from its dispatch through ymf_run, and the run is the deck's, bitwise.

The same case two ways, each in a directory of its own:

* the deck, ``parun --TwoPhaseFlow damBreakLevelSet.py -C "final_time=0.03 dt_output=0.01 he=0.08 genMesh=True"``;
* the spec, ``pro run dambreak_short.ymf``: proteus.Dispatch maps level_set_five onto RANS2P, VOF, NCLS, RDLS and
  MCorr, and proteus.DispatchRun builds the TwoPhaseFlowProblem from that and from ymf's split plain data.

Every numeric array the two archives hold (the five models' unknowns at t = 0, 0.001, 0.01, 0.02, 0.03, the mesh,
the material types) must be bitwise equal, on one rank and on two (the partition pinned by
``-mat_partitioning_type current``). The mesh records may differ only in the name of their HDF5 file.
"""
import os
import shutil
import subprocess
import sys

import pytest

pytest.importorskip("ymf.symbolic.split")
if "hypot" not in __import__("ymf.names", fromlist=["FUNCTION_NAMES"]).FUNCTION_NAMES:
    # dambreak_short.ymf writes the deck's phi_init with hypot (cekees/ymf notation-hypot)
    pytest.skip("this ymf has no hypot in its notation (cekees/ymf branch notation-hypot)", allow_module_level=True)
h5py = pytest.importorskip("h5py")
numpy = pytest.importorskip("numpy")

HERE = os.path.dirname(os.path.abspath(__file__))
DECKS = os.path.join(os.path.dirname(HERE), "TwoPhaseFlow")


def _mpiexec():
    beside = os.path.join(os.path.dirname(sys.executable), "mpiexec")
    return beside if os.path.exists(beside) else shutil.which("mpiexec")


MPIEXEC = _mpiexec()


def _run(command, cwd, ranks):
    env = dict(os.environ, OMP_NUM_THREADS="1", PETSC_OPTIONS="-mat_partitioning_type current")
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    if ranks > 1:
        command = [MPIEXEC, "-n", str(ranks)] + command
    ran = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True, timeout=600, env=env)
    return ran


def _arrays(path, stem):
    """{dataset: array} of an archive; a mesh record (text) with its file's name replaced by a placeholder."""
    out = {}
    with h5py.File(str(path), "r") as f:
        def visit(key, item):
            if not isinstance(item, h5py.Dataset):
                return
            value = item[()]
            if isinstance(value, numpy.ndarray) and value.dtype.kind in "SOU":
                value = numpy.array([v.replace(stem.encode() + b".h5", b"<archive>.h5") for v in value])
            elif isinstance(value, bytes):
                value = value.replace(stem.encode() + b".h5", b"<archive>.h5")
            out[key] = value
        f.visititems(visit)
    return out


def _parun():
    found = shutil.which("parun", path=os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", ""))
    return [found] if found else [sys.executable, os.path.join(os.path.dirname(os.path.dirname(HERE)), "scripts",
                                                               "parun")]


@pytest.mark.parametrize("ranks", [1, 2])
def test_level_set_five_from_the_spec_is_damBreakLevelSet_bitwise(tmp_path, ranks):
    if ranks > 1 and MPIEXEC is None:
        pytest.skip("mpiexec not available")
    deck, spec = tmp_path / "deck", tmp_path / "spec"
    deck.mkdir()
    spec.mkdir()
    # the deck meshes beside itself (domain.polyfile is its directory): a copy, so the stored mesh stays as it is
    shutil.copy(os.path.join(DECKS, "damBreakLevelSet.py"), str(deck))
    ran = _run(_parun() + ["--TwoPhaseFlow", "damBreakLevelSet.py", "-l1",
                           "-C", "final_time=0.03 dt_output=0.01 he=0.08 genMesh=True"], deck, ranks)
    assert ran.returncode == 0, ran.stdout[-3000:] + ran.stderr[-3000:]
    ran = _run([sys.executable, "-m", "proteus.cli", "run", os.path.join(HERE, "dambreak_short.ymf"),
                "--outdir", str(spec)], spec, ranks)
    assert ran.returncode == 0, ran.stdout[-3000:] + ran.stderr[-3000:]
    assert "level_set_five -- split, dispatched to flow RANS2P, vof VOF, ncls NCLS, rdls RDLS, mcorr MCorr" \
        in ran.stdout
    (archive,) = [f for f in os.listdir(str(spec)) if f.endswith(".h5")]
    expected = _arrays(deck / "damBreakLevelSet.h5", "damBreakLevelSet")
    found = _arrays(spec / archive, archive[:-3])
    assert sorted(found) == sorted(expected)
    assert "u_t4" in found and float(numpy.abs(found["velocity_t4"]).max()) > 0.1     # the water moved
    different = [key for key in expected if not numpy.array_equal(found[key], expected[key])]
    assert not different, "%d of %d datasets differ: %s" % (len(different), len(expected), different[:10])
