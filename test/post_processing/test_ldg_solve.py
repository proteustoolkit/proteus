"""The LDG solve (DG with Diffusion_LDG) on the VPP Poisson problem: a = diag(x + 1, y + 2), u = x^2 + y^2.

Until 2026-11-05 it was wrong and not reproducible (L2 velocity error 377 through point-eval, 1.7e2 to 3.6e2 through
point-eval-gwvd, a different number on every run): Diffusion_LDG passed its exterior flux Jacobian kernel (rowptr,
colind, isDiffusiveFluxBoundary) for (isDiffusiveFluxBoundary, rowptr, colind), so the kernel read the face flags past
the end of the 3-entry rowptr and dropped a V.n on every face whose normal is not along x. The velocity here is
point-eval's (-a grad u of the DG solution at the quadrature points). The face flux reads the neighbour's velocity,
which is solved from the neighbour's own faces, so a rank needs two layers of overlap; Transport refuses one.
"""
import os
import sys

import numpy
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_velocity_postprocessing import MPIEXEC, _run, _summary


def test_ldg_converges_at_first_order_and_is_exact_for_p2(tmp_path):
    """L2 velocity error 0.254, 0.124, 0.0615 on 7, 13, 25 nodes per side (rates 1.03, 1.02); P2 3e-13 to 1.5e-12."""
    errors = []
    for nn in (7, 13, 25):
        _run(1, tmp_path / str(nn), "point-eval", 1, space="ldg", nn=nn)
        errors.append(_summary(tmp_path / str(nn))["l2_velocity_error"])
    rates = numpy.log2(numpy.array(errors[:-1]) / numpy.array(errors[1:]))
    assert errors[1] < 0.13 and numpy.abs(rates - 1.0).max() < 0.1, (errors, rates)
    _run(1, tmp_path / "p2", "point-eval", 2, space="ldg")
    assert _summary(tmp_path / "p2")["l2_velocity_error"] < 1e-10


def test_ldg_is_reproducible(tmp_path):
    """Identical runs gave different answers (uninitialised memory read as face flags)."""
    first = _run(1, tmp_path / "a", "point-eval", 1, space="ldg")
    second = _run(1, tmp_path / "b", "point-eval", 1, space="ldg")
    assert numpy.array_equal(first, second)


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("ranks", [2, 4])
def test_ldg_on_ranks_with_two_layers_is_the_serial_answer(tmp_path, ranks):
    """Measured: the L2 velocity error equal to serial within 1e-15 on 2 and 4 ranks (0.124374745417200); with one
    layer, through point-eval-gwvd, 0.113510 and 0.113549 against 0.113510 serial."""
    serial = _run(1, tmp_path / "1", "point-eval", 1, space="ldg")
    parallel = _run(ranks, tmp_path / "n", "point-eval", 1, overlap=2, space="ldg")
    assert abs(_summary(tmp_path / "1")["l2_velocity_error"] - _summary(tmp_path / "n")["l2_velocity_error"]) < 1e-12
    s, p = (rows[numpy.lexsort((rows[:, 1], rows[:, 0]))] for rows in (serial, parallel))
    assert numpy.abs(s[:, :2] - p[:, :2]).max() < 1e-12
    assert numpy.abs(s[:, -12:] - p[:, -12:]).max() < 1e-10 * numpy.abs(s[:, -12:]).max()


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
def test_ldg_on_one_layer_is_refused_on_every_rank(tmp_path):
    out = _run(2, tmp_path, "point-eval", 1, overlap=1, space="ldg", check=False)
    output = out.stdout + out.stderr
    assert out.returncode != 0
    assert output.count("owned elements or their face neighbours (LDG)") >= 2, output[-3000:]
