"""C5 step 2: a split discretization runs from its dispatch through ymf_run, and the run is the deck's, bitwise.

Each case two ways, each in a directory of its own:

* the deck through parun: ``parun --TwoPhaseFlow damBreakLevelSet.py -C "final_time=0.03 dt_output=0.01 he=0.08
  genMesh=True"``, ``parun --SWEs dam3Bumps.py -C "final_time=0.5 dt_output=0.1 structured=False he=2.0"``;
* the spec through ``pro run``: proteus.Dispatch maps level_set_five onto RANS2P, VOF, NCLS, RDLS and MCorr, and
  convex_limited onto SW2DCV, and proteus.DispatchRun builds the TwoPhaseFlowProblem or SWFlowProblem from that and
  from ymf's split plain data.

Every numeric array the two archives hold (every model's unknowns at every output time, the mesh, the material
types) must be bitwise equal, on one rank and on two (the partition pinned by ``-mat_partitioning_type current``).
The mesh records may differ only in the name of their HDF5 file.

dambreak_short.ymf writes the deck's phi_init with hypot, which ymf's notation has since cekees/ymf#12; with an
older ymf the dam-break case fails to parse, and should: it cannot match the deck without it.
"""
import os
import shutil
import subprocess
import sys

import pytest

pytest.importorskip("ymf.symbolic.split")
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


#: (deck directory, deck, parun's flag, its options, spec, what ymf_run says it dispatched to, a field that moves)
CASES = {
    "damBreakLevelSet": ("TwoPhaseFlow", "damBreakLevelSet.py", "--TwoPhaseFlow",
                         "final_time=0.03 dt_output=0.01 he=0.08 genMesh=True", "dambreak_short.ymf",
                         "level_set_five -- split, dispatched to flow RANS2P, vof VOF, ncls NCLS, rdls RDLS, "
                         "mcorr MCorr", "velocity_t4"),
    # the same deck with interface=1: RANS2P and CLSVOF. The spec's atmosphere (p = 0, no flux, φ signed + where flow
    # enters) is setAtmosphere, which sets no CLSVOF data; the deck has the same boundary
    "damBreakLevelSet_clsvof": ("TwoPhaseFlow", "damBreakLevelSet.py", "--TwoPhaseFlow",
                                "final_time=0.03 dt_output=0.01 he=0.08 genMesh=True interface=1",
                                "dambreak_short_clsvof.ymf",
                                "level_set_clsvof -- split, dispatched to flow RANS2P, clsvof CLSVOF", "velocity_t4"),
    "dam3Bumps": ("SWFlow", "dam3Bumps.py", "--SWEs", "final_time=0.5 dt_output=0.1 structured=False he=2.0",
                  "dam3bumps.ymf", "convex_limited -- split, dispatched to swe SW2DCV", "h_u_t6"),
}


#: damBreakLevelSet.py interface=1 (RANS2P + CLSVOF) on two ranks is not run-to-run reproducible under load, deck
#: alone: 2 of 20 runs of this test, and 1 of 16 decks run at once with nothing of the spec's, differ from the rest in
#: 64-66 of ~215 datasets (macOS arm64, 2026-11-11). The spec's path is the deck's; the cause is in the model
#: (C0's territory: HANDOFF_2026-11-11_c5-step3.md), so this one case is not asserted bitwise on two ranks.
NONREPRODUCIBLE = {("damBreakLevelSet_clsvof", 2)}


def _params():
    return [pytest.param(case, ranks, marks=[pytest.mark.xfail(
        strict=False, reason="the deck itself differs from run to run on two ranks with CLSVOF (see NONREPRODUCIBLE)")]
        if (case, ranks) in NONREPRODUCIBLE else [], id="%s-%d" % (case, ranks))
        for case in sorted(CASES) for ranks in (1, 2)]


@pytest.mark.parametrize("case, ranks", _params())
def test_the_spec_runs_the_deck_bitwise(tmp_path, case, ranks):
    if ranks > 1 and MPIEXEC is None:
        pytest.skip("mpiexec not available")
    folder, script, flag, options, spec_file, banner, moving = CASES[case]
    deck, spec = tmp_path / "deck", tmp_path / "spec"
    deck.mkdir()
    spec.mkdir()
    # the deck meshes in (or beside) its own directory: a copy, so the stored mesh files stay as they are
    shutil.copy(os.path.join(os.path.dirname(HERE), folder, script), str(deck))
    ran = _run(_parun() + [flag, script, "-l1", "-C", options], deck, ranks)
    assert ran.returncode == 0, ran.stdout[-3000:] + ran.stderr[-3000:]
    ran = _run([sys.executable, "-m", "proteus.cli", "run", os.path.join(HERE, spec_file), "--outdir", str(spec)],
               spec, ranks)
    assert ran.returncode == 0, ran.stdout[-3000:] + ran.stderr[-3000:]
    assert banner in ran.stdout
    (archive,) = [f for f in os.listdir(str(spec)) if f.endswith(".h5")]
    expected = _arrays(deck / (script[:-3] + ".h5"), script[:-3])
    found = _arrays(spec / archive, archive[:-3])
    assert sorted(found) == sorted(expected)
    assert float(numpy.abs(found[moving]).max()) > 0.1          # the water moved
    different = [key for key in expected if not numpy.array_equal(found[key], expected[key])]
    assert not different, "%d of %d datasets differ: %s" % (len(different), len(expected), different[:10])
