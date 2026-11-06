"""TwoPhaseFlow and so-file decks on 2 ranks give the serial answer, every model component and every owned dof.

Measured on main f392a159 (macOS arm64, partition pinned with -mat_partitioning_type current), largest relative
difference over a model's dofs, serial against 2 ranks:

  damBreak (CLSVOF + RANS3PF projection, he=0.1, t=0.1)   atol 1e-12: 5.4e-13 (3 ranks 3.4e-13); deck tolerances
                                                           (~1e-3): 9.5e-4; 1e-6: 4.1e-7; 1e-9: 6.3e-10
  TwoDimBucklingFlow (refinement=2, t=0.05)               atol 1e-12: 2.5e-11
  fillingTank (he=0.1, t=0.05)                            atol 1e-9: 1.9e-6. A tighter atol is out of reach for the
                                                           parallel CLSVOF KSP (1000 iterations at 7e-10), the steps
                                                           fail and are retried; see the note on retries below.
  risingBubble (refinement=1, t=0.05)                     atol 1e-12: CLSVOF and velocity 2.6e-13. The pressure
                                                           increment, pressure and initial pressure differed by a
                                                           constant (8.1, -28 and -63 against 1142 Pa); with zero-mean
                                                           normalization (round 4) 4.0e-12, 4.8e-14, 3.0e-15.

Added in C0 round 4 (same measure; every component of a model now, and dofs matched by x, y and z: before, only the
last component of RANS2P/RANS3PF was compared and 3D dofs sharing x, y were matched in arbitrary order):

  damBreakLevelSet (RANS2P + VOF + NCLS + RDLS + MCorr, t=0.5)  atol 1e-10: 9.6e-15 (3 ranks 6.0e-14); deck tolerances
                                                           3.1e-14. RDLS needs MAXITS (50 Newton iterations stall
                                                           at ~1e-7 for atol 1e-10).
  damBreakLevelSet interface=1 (RANS2P + CLSVOF, t=0.5)    atol 1e-10: 2.4e-13
  marin (3D, CLSVOF + RANS3PF projection, he=0.5, t=0.1)   atol 1e-10: 2.4e-10
  moses (3D, surface tension, he=0.5, t=0.1)               atol 1e-10: 7.6e-12
  surface_tension/rising_bubble_rans3p (2D, RANS3PF + VOF3P + NCLS3P + RDLS elliptic redistancing + MCorr3P,
    its own so file, LU replaced by a PETSc direct solve)  atol 1e-10: 5.5e-14. Before: the 2-rank run failed (the
                                                           deck pinned PresInc dof 0 with the top open, see its p
                                                           file); with that fixed, 2.0 in MCorr3P (1.9e-2 velocity);
                                                           with the NCLS interface locator and the RDLS normal
                                                           reconstruction scattered, 1.1e-2; with the entropy
                                                           viscosity normalized over the domain, 5.5e-14.

The deck tolerances are the control for the comparison: the difference follows the tolerance, so a run that agrees at
a tight one is a partition test. The pressure constant was a real serial/parallel difference, not round-off: in a
closed tank the pressure is fixed only up to a constant, and serial and parallel fixed it differently (PresInc pinned
dof 0 in serial and used the constant null space in parallel; PressureInitial took whatever its solver returned), so
pressure gauges depended on the rank count. Since round 4 PresInc, Pres and PresInit are shifted to zero integral mean
when no Dirichlet condition fixes them (proteus.mprans.PressureNormalization), and the pressures agree as they are.

Retries: forcing one CLSVOF failure at the same call in serial and on 2 ranks gives the same answer on both (2.4e-7),
so recovery is partition-correct; but the retried serial run differs from the unforced one by 16% in velocity at
t=0.05, which is a time-stepping question, not a partition one.
"""
import importlib.util
import os
import shutil
import sys

import numpy
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
# a failed launch keeps its whole output (loaded by path: --import-mode=importlib)
_spec = importlib.util.spec_from_file_location("launch_output", os.path.join(HERE, "launch_output.py"))
launch_output = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launch_output)
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
    out = launch_output.run(argv, cwd=cwd, env=env, timeout=1800)
    names = dict(line.strip().split(",") for line in open(os.path.join(str(cwd), "models.csv")))
    u = numpy.loadtxt(os.path.join(str(cwd), "u.csv"), delimiter=",", skiprows=1)
    return u, {int(k): v for k, v in names.items()}


def _differences(a, b, names, remove_constant=()):
    """The largest relative difference of each model component (the model's name for component 0, name[i] for
    component i), after removing the mean difference of the named ones."""
    assert a.shape == b.shape and numpy.abs(a[:, :4] - b[:, :4]).max() < 1e-10
    out = {}
    for key in numpy.unique(a[:, 0]):
        k = a[:, 0] == key
        d = a[k, 4] - b[k, 4]
        name = names[int(key)] if int(key) % 100 == 0 else "%s[%d]" % (names[int(key)], int(key) % 100)
        if name in remove_constant:
            d = d - d.mean()
        out[name] = numpy.abs(d).max() / (numpy.abs(a[k, 4]).max() + 1e-300)
    return out


CASES = {
    "damBreak": (dict(DECK="damBreak.py", OPTS="he=0.1", FT=0.1, ATOL=1e-12), 1e-10),
    "TwoDimBucklingFlow": (dict(DECK="TwoDimBucklingFlow.py", OPTS="refinement=2", FT=0.05, ATOL=1e-12), 1e-9),
    "fillingTank": (dict(DECK="fillingTank.py", OPTS="he=0.1", FT=0.05, ATOL=1e-9), 1e-5),
    "damBreakLevelSet": (dict(DECK="damBreakLevelSet.py", OPTS="he=0.1", FT=0.5, ATOL=1e-10, MAXITS=1000), 1e-10),
    "damBreakLevelSet-clsvof": (dict(DECK="damBreakLevelSet.py", OPTS="he=0.1 interface=1", FT=0.5, ATOL=1e-10,
                                     MAXITS=1000), 1e-10),
    # k-epsilon (Kappa, Dissipation): 3.4e-9 / 6.8e-9 relative, 3e-13 absolute on values ~1e-4 (solver tolerance)
    "damBreakLevelSet-turbulence": (dict(DECK="damBreakLevelSet.py", OPTS="he=0.1 turbulence=1", FT=0.1, ATOL=1e-10,
                                         MAXITS=1000), 1e-7),
    "marin": (dict(DECK="marin.py", OPTS="he=0.5", FT=0.1, ATOL=1e-10, MAXITS=200), 1e-8),
    "moses": (dict(DECK="moses.py", OPTS="he=0.5", FT=0.1, ATOL=1e-10, MAXITS=200), 1e-9),
    "rising_bubble_rans3p": (dict(DIR="surface_tension/rising_bubble_rans3p", SO="risingBubble_so.py",
                                  OPTS="nd=2 USE_SUPG_NS=1 ARTIFICIAL_VISCOSITY_NS=1", DIRECT=1, ATOL=1e-10,
                                  MAXITS=200), 1e-10),
    # fixNullSpace with the top open: the Dirichlet condition fixes the increment, so neither the serial pin of dof 0
    # nor the parallel constant null space may apply (main: 2-rank run fails)
    "rising_bubble_rans3p-fixNullSpace": (dict(DIR="surface_tension/rising_bubble_rans3p", SO="risingBubble_so.py",
                                               OPTS="nd=2 USE_SUPG_NS=1 ARTIFICIAL_VISCOSITY_NS=1", DIRECT=1,
                                               ATOL=1e-10, MAXITS=200, FIXNULLSPACE=1), 1e-10),
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


def test_rising_bubble_pressure_constant_does_not_depend_on_the_rank_count(rising_bubble):
    serial, parallel, names = rising_bubble
    worst = _differences(serial, parallel, names)
    assert max(worst[n] for n in PRESSURES if n in worst) < 1e-10, worst


def _boundary_integrals(path):
    """{(tag, quantity, component): [(time, value)]} from a BoundaryIntegralGauges file."""
    lines = open(path).read().splitlines()
    assert lines[0] == "time,tag,quantity,component,units,value"
    series = {}
    for line in lines[1:]:
        t, tag, quantity, component, units, value = [x.strip() for x in line.split(",")]
        series.setdefault((tag, quantity, component), []).append((float(t), float(value)))
    return {k: numpy.array(v) for k, v in series.items()}


BOUNDARY_INTEGRAL_CASES = {
    # measured: damBreak (RANS3PF) 1.2e-11, damBreakLevelSet (RANS2P) 1.8e-11, the largest on values near 1e-18
    "damBreak-rans3pf": dict(DECK="damBreak.py", OPTS="he=0.1", FT=0.1, ATOL=1e-12, BIGAUGES="1,2,3,4"),
    "damBreakLevelSet-rans2p": dict(DECK="damBreakLevelSet.py", OPTS="he=0.1", FT=0.5, ATOL=1e-10, MAXITS=1000,
                                    BIGAUGES="1,2,3,4"),
}


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("case", sorted(BOUNDARY_INTEGRAL_CASES))
def test_boundary_integral_gauges_on_two_ranks_are_the_serial_ones(tmp_path, case):
    """Every tag, quantity (wetted area, pressure, viscous and total force, moment about a declared point) and
    component of a BoundaryIntegralGauges file, serial against 2 ranks."""
    options = BOUNDARY_INTEGRAL_CASES[case]
    _field(1, tmp_path / "1", **options)
    _field(2, tmp_path / "2", **options)
    serial = _boundary_integrals(str(tmp_path / "1" / "boundary_integrals.csv"))
    parallel = _boundary_integrals(str(tmp_path / "2" / "boundary_integrals.csv"))
    assert sorted(serial) == sorted(parallel)
    quantities = set(k[1] for k in serial)
    assert quantities == {"wetted_area", "force_pressure", "force_viscous", "force", "moment"}
    for key, s in serial.items():
        p = parallel[key]
        assert s.shape == p.shape and numpy.allclose(s[:, 0], p[:, 0], rtol=1e-12, atol=0), key
        assert numpy.abs(s[:, 1] - p[:, 1]).max() <= 1e-9 * max(numpy.abs(s[:, 1]).max(), 1.0), key

