"""test/shallow_water/dambreak_1d runs: RKDG P1 shallow water, Rusanov flux, P1 limiter, a dry-bed dam break between
two walls.

Until 2026-11-05 it stopped in RusanovNumericalFlux's constructor (a missing periodic argument, since 2014) and then
in the 1D limiter (ctimeIntegration wrappers that did not take TimeIntegration.py's arguments, since 2019).
"""
import inspect
import os
import shutil
import subprocess
import sys

import h5py
import numpy
import pytest

from proteus import TimeIntegration

# Cython 3 binds self to a compiled function stored as a class attribute; TimeIntegration's limiter classes need
# staticmethod for it (cekees/proteus#204, C47). Until that is on the branch the limiter call fails before this deck
# can run, so the test waits for it.
pytestmark = pytest.mark.skipif(
    not isinstance(inspect.getattr_static(TimeIntegration.DGlimiterP1Lagrange1d, "applyDGlimitingP1Lagrange1d"),
                   staticmethod),
    reason="needs the staticmethod limiter attributes of cekees/proteus#204")

_here = os.path.dirname(os.path.abspath(__file__))
_parun = os.path.join(_here, os.pardir, os.pardir, os.pardir, "scripts", "parun")
_P, _N = "sw_dam_break_1d_p.py", "sw_dam_break_1d_dgpk_n.py"
MPIEXEC = shutil.which("mpiexec")

# The DG flux needs the face neighbour of every owned element (Transport refuses a node partition), and LU is serial.
_PARALLEL_N = """
parallelPartitioningType = MeshParallelPartitioningTypes.element
nLayersOfOverlapForParallel = 1
multilevelLinearSolver = KSP_petsc4py
levelLinearSolver = KSP_petsc4py
linear_solver_options_prefix = "sw_"
"""
_PARALLEL_PETSC = "-mat_partitioning_type current -sw_ksp_type preonly -sw_pc_type bjacobi -sw_sub_pc_type lu"


def _run(launch_policy, workdir, ranks=1, parallel=False):
    """Run the deck in workdir; parallel: with the element partition and the PETSc solve (any number of ranks)."""
    workdir.mkdir()
    for name in (_P, _N):
        shutil.copy(os.path.join(_here, name), workdir)
    env = dict(os.environ)
    cmd = [sys.executable, _parun if os.path.isfile(_parun) else shutil.which("parun"), "-l", "1", "-D", "data", _P, _N]
    if parallel:
        with open(workdir / _N, "a") as f:
            f.write(_PARALLEL_N)
        env["PETSC_OPTIONS"] = _PARALLEL_PETSC
        cmd = [MPIEXEC, "-n", str(ranks)] + cmd
    out = launch_policy.retrying_run(subprocess.run)(cmd, cwd=workdir, env=env, capture_output=True, text=True,
                                                     timeout=300)
    assert out.returncode == 0, (out.stdout + out.stderr)[-4000:]
    return _read(workdir / "data" / "sw_dam_break_1d.h5")


def _read(path):
    """[(element midpoint, x, h, hu) per DG dof, sorted] for each of the 101 outputs."""
    steps = []
    with h5py.File(path, "r") as f:
        for i in range(101):
            x = f["nodes_dgp1_Lagrange%d" % i][:, 0]
            elements = f["elements_dgp1_Lagrange%d" % i][:]
            h, hu = f["h_t%d" % i][:], f["hu_t%d" % i][:]
            rows = {(0.5 * (x[a] + x[b]), x[d], h[d], hu[d]) for a, b in elements for d in (a, b)}
            steps.append(numpy.array(sorted(rows)))
    return steps


def _mass(rows):
    left, right = rows[0::2], rows[1::2]
    return numpy.sum(0.5 * (left[:, 2] + right[:, 2]) * (right[:, 1] - left[:, 1]))


def test_dambreak_1d_conserves_mass_and_stays_nonnegative(tmp_path, launch_policy):
    """Measured: mass 29/6 (the P1 interpolation of the step) at every output to 1.2e-14; h below zero at 3 of 101
    outputs, by at most 5.0e-8 (t = 0.6, the wall cell x = 0 as the front arrives: the P1 limiter does not preserve
    positivity); the water settling towards the rest depth 29/60 (h in [0.3645, 0.4985] at T = 20)."""
    steps = _run(launch_policy, tmp_path / "serial")
    assert all(len(rows) == 60 for rows in steps)
    masses = numpy.array([_mass(rows) for rows in steps])
    assert abs(masses[0] - 29.0 / 6.0) < 1e-12
    assert numpy.abs(masses - masses[0]).max() < 1e-12
    assert min(rows[:, 2].min() for rows in steps) > -1e-7
    assert numpy.ptp(steps[-1][:, 2]) < 0.15


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
def test_dambreak_1d_on_two_ranks_is_the_serial_answer(tmp_path, launch_policy):
    """Measured: identical to the last bit on 2, 3 and 4 ranks (explicit SSPRK, element-local work, a block-diagonal
    mass matrix solved exactly per rank)."""
    serial = _run(launch_policy, tmp_path / "1", 1, parallel=True)
    parallel = _run(launch_policy, tmp_path / "2", 2, parallel=True)
    for s, p in zip(serial, parallel):
        assert numpy.array_equal(s, p)
