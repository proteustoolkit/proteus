"""proteus.Gauges.BoundaryIntegralGauges: what it reports and what it refuses (no simulation; the partition, retry
and hot-start behaviour is tested in test_twophaseflow_partition_independence.py, test_step_retry.py and
test_hotstart_history_files.py)."""
import numpy as np
import pytest

from proteus.Gauges import BoundaryIntegralGauges


class _Coefficients(object):
    """Per-flag arrays as RANS2P/RANS3PF leave them: flag 1 has a wetted area, forces and a moment about (1, 0, 0)."""
    def __init__(self):
        self.wettedAreas = np.array([0.0, 2.0, 0.0])
        self.netForces_p = np.array([[0.0] * 3, [3.0, 4.0, 0.0], [0.0] * 3])
        self.netForces_v = np.array([[0.0] * 3, [1.0, -1.0, 0.0], [0.0] * 3])
        self.netMoments = np.array([[0.0] * 3, [0.0, 0.0, 5.0], [0.0] * 3])
        self.barycenters = np.array([[0.0] * 3, [1.0, 0.0, 0.0], [0.0] * 3])


def _gauge(**kwargs):
    g = BoundaryIntegralGauges(tags={"wall": 1}, **kwargs)
    g.coefficients, g.nd = _Coefficients(), 2
    return g


def test_values_in_2d():
    out = dict((q, (c, list(v))) for q, c, v in _gauge().values("wall", 1))
    assert out["wetted_area"] == (("",), [2.0])
    assert out["force_pressure"] == (("x", "y"), [3.0, 4.0])
    assert out["force_viscous"] == (("x", "y"), [1.0, -1.0])
    assert out["force"] == (("x", "y"), [4.0, 3.0])
    assert out["moment"] == (("z",), [5.0])


def test_the_moment_about_a_declared_point_is_shifted_by_the_force():
    # about (0, 0): M = M_b + (b - p) x F = 5 + (1, 0) x (4, 3) = 5 + 3
    out = dict((q, list(v)) for q, c, v in _gauge(moment_points={"wall": (0.0, 0.0, 0.0)}).values("wall", 1))
    assert out["moment"] == [8.0]


@pytest.mark.parametrize("kwargs, message", [
    (dict(tags=[0]), "not a boundary condition flag"),
    (dict(tags=[1], quantities=("drag",)), "unknown boundary integral quantity"),
    (dict(tags=[1], moment_points={"2": (0, 0, 0)}), "not gauged"),
])
def test_bad_definitions_are_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        BoundaryIntegralGauges(**kwargs)
