"""TwoPhaseFlow decks on 2 ranks give the serial answer, every model and every owned dof.

Measured on main f392a159 (macOS arm64, partition pinned with -mat_partitioning_type current), largest relative
difference over a model's dofs, serial against 2 ranks:

  damBreak (CLSVOF + RANS3PF projection, he=0.1, t=0.1)   atol 1e-12: 5.4e-13 (3 ranks 3.4e-13); deck tolerances
                                                           (~1e-3): 9.5e-4; 1e-6: 4.1e-7; 1e-9: 6.3e-10
  TwoDimBucklingFlow (refinement=2, t=0.05)               atol 1e-12: 2.5e-11
  fillingTank (he=0.1, t=0.05)                            atol 1e-9: 1.9e-6. A tighter atol is out of reach for the
                                                           parallel CLSVOF KSP (1000 iterations at 7e-10), the steps
                                                           fail and are retried; see the note on retries below.
  risingBubble (refinement=1, t=0.05)                     atol 1e-12: CLSVOF and velocity 2.6e-13. The pressure
                                                           increment, pressure and initial pressure differ by a
                                                           constant (8.1, -28 and -63 against 1142 Pa): with the
                                                           constant removed 2.3e-12, 3.9e-14, 2.4e-15.

The deck tolerances are the control for the comparison: the difference follows the tolerance, so a run that agrees at
a tight one is a partition test. The pressure constant is a real serial/parallel difference, not round-off: in a
closed tank the pressure is fixed only up to a constant, and serial and parallel fix it differently (PresInc pins
dof 0 in serial and uses the constant null space in parallel; PressureInitial takes whatever its solver returns), so
pressure gauges depend on the rank count. That test is marked xfail until a convention is chosen.

Retries: forcing one CLSVOF failure at the same call in serial and on 2 ranks gives the same answer on both (2.4e-7),
so recovery is partition-correct; but the retried serial run differs from the unforced one by 16% in velocity at
t=0.05, which is a time-stepping question, not a partition one.
"""
import os
import shutil
import subprocess
import sys

import numpy
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.join(HERE, "twophaseflow_partition_mpi.py")


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ("mpiexec", "mpirun"):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which("mpiexec") or shutil.which("mpirun")


MPIEXEC = _mpiexec()
PRESSURES = ("pressureIncrement0", "pressure0", "pressureInitial0")


def _field(ranks, cwd, **overrides):
    os.makedirs(str(cwd), exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1")
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env["PETSC_OPTIONS"] = (env.get("PETSC_OPTIONS", "") + " -mat_partitioning_type current").strip()
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env.update({k: str(v) for k, v in overrides.items()})
    argv = [sys.executable, HELPER] if ranks == 1 else [MPIEXEC, "-n", str(ranks), sys.executable, HELPER]
    out = subprocess.run(argv, capture_output=True, text=True, timeout=1800, env=env, cwd=str(cwd))
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    names = dict(line.strip().split(",") for line in open(os.path.join(str(cwd), "models.csv")))
    u = numpy.loadtxt(os.path.join(str(cwd), "u.csv"), delimiter=",", skiprows=1)
    return u, {int(k): v for k, v in names.items()}


def _differences(a, b, names, remove_constant=()):
    """The largest relative difference of each model component, after removing the mean difference of the named ones."""
    assert a.shape == b.shape and numpy.abs(a[:, :3] - b[:, :3]).max() < 1e-10
    out = {}
    for key in numpy.unique(a[:, 0]):
        k = a[:, 0] == key
        d = a[k, 3] - b[k, 3]
        if names[int(key)] in remove_constant:
            d = d - d.mean()
        out[names[int(key)]] = numpy.abs(d).max() / (numpy.abs(a[k, 3]).max() + 1e-300)
    return out


CASES = {
    "damBreak": (dict(DECK="damBreak.py", OPTS="he=0.1", FT=0.1, ATOL=1e-12), 1e-10),
    "TwoDimBucklingFlow": (dict(DECK="TwoDimBucklingFlow.py", OPTS="refinement=2", FT=0.05, ATOL=1e-12), 1e-9),
    "fillingTank": (dict(DECK="fillingTank.py", OPTS="he=0.1", FT=0.05, ATOL=1e-9), 1e-5),
}


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("case", sorted(CASES))
def test_twophaseflow_on_two_ranks_is_the_serial_answer(tmp_path, case):
    options, tolerance = CASES[case]
    serial, names = _field(1, tmp_path / "1", **options)
    parallel, _ = _field(2, tmp_path / "2", **options)
    worst = _differences(serial, parallel, names)
    assert max(worst.values()) < tolerance, worst


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
def test_dambreak_differs_at_the_decks_own_tolerances():
    """The control: at nl_atol_res ~1e-3 serial and parallel iterations stop at different places."""
    options = dict(DECK="damBreak.py", OPTS="he=0.1", FT=0.1)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        serial, names = _field(1, os.path.join(d, "1"), **options)
        parallel, _ = _field(2, os.path.join(d, "2"), **options)
    assert max(_differences(serial, parallel, names).values()) > 1e-6


RISING_BUBBLE = dict(DECK="risingBubble.py", OPTS="refinement=1", FT=0.05, ATOL=1e-12)


@pytest.fixture(scope="module")
def rising_bubble(tmp_path_factory):
    if MPIEXEC is None:
        pytest.skip("mpiexec not available")
    d = tmp_path_factory.mktemp("risingBubble")
    serial, names = _field(1, d / "1", **RISING_BUBBLE)
    parallel, _ = _field(2, d / "2", **RISING_BUBBLE)
    return serial, parallel, names


def test_rising_bubble_on_two_ranks_is_the_serial_answer_up_to_the_pressure_constant(rising_bubble):
    serial, parallel, names = rising_bubble
    worst = _differences(serial, parallel, names, remove_constant=PRESSURES)
    assert max(worst.values()) < 1e-10, worst


@pytest.mark.xfail(strict=True, reason="serial and parallel fix the pressure constant differently (C0 round 3 finding)")
def test_rising_bubble_pressure_constant_does_not_depend_on_the_rank_count(rising_bubble):
    serial, parallel, names = rising_bubble
    worst = _differences(serial, parallel, names)
    assert max(worst[n] for n in PRESSURES if n in worst) < 1e-10, worst
