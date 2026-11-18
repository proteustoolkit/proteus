"""Newton ends a solve whose iterate or residual is not finite, or has grown by more than 1e30, as a failed
solve (the step controller then retries with a smaller dt). A diverging Richards implicit-FCT first step went
to |u| = 1e189 over 13 retries, the linear solve of that state overflowed and the next Jacobian was all NaN.
A solve that stays in range is untouched: the guard only reads u and r (VOF STAB=1 serial and on 2 ranks,
1146 guard calls, none fired, bit-identical solution; measured in the commit message)."""
import numpy as np
import pytest

from proteus import NonlinearSolvers


class _Solver(NonlinearSolvers.NonlinearSolver):
    """the guard's inputs without a model: the residual norm and the norm of the first residual"""

    def __init__(self, norm_r0=1.0):
        self.norm_r0 = norm_r0
        self.its = 3

    def norm(self, r):
        return float(np.linalg.norm(r))


@pytest.mark.parametrize("u, r, norm_r0, diverged", [
    (np.array([1.0, -45.0]), np.array([1.0e3, -2.0]), 1.0, False),                 # an ordinary iterate
    (np.array([1.0, -45.0]), np.array([1.0e20, 0.0]), 1.0, False),                 # a large residual is not a divergence
    (np.array([1.0e29, 0.0]), np.array([1.0, 0.0]), 1.0, False),                   # below 1e30 times max(1, |u0|)
    (np.array([1.0, np.nan]), np.array([1.0, 0.0]), 1.0, True),                    # non-finite iterate
    (np.array([1.0, np.inf]), np.array([1.0, 0.0]), 1.0, True),
    (np.array([1.0, 2.0]), np.array([np.nan, 0.0]), 1.0, True),                    # non-finite residual
    (np.array([1.0, 2.0]), np.array([np.inf, 0.0]), 1.0, True),
    (np.array([5.0e31, 0.0]), np.array([1.0, 0.0]), 1.0, True),                    # |u| grew by 1e30 over |u0| = 45
    (np.array([1.0, 2.0]), np.array([1.0e31, 0.0]), 1.0, True),                    # residual grew by 1e30 over ||r0||
    (np.array([1.0, 2.0]), np.array([1.0e5, 0.0]), 1.0e-20, False),                # a tiny ||r0|| does not make 1e5 a divergence
])
def test_iterate_diverged(u, r, norm_r0, diverged):
    assert _Solver(norm_r0).iterateDiverged(u, r, 45.0) is diverged


def test_iterate_diverged_measures_growth_against_the_start_of_the_solve():
    solver = _Solver(1.0e10)
    u, r = np.array([1.0, 2.0]), np.array([1.0e39, 0.0])
    assert not solver.iterateDiverged(u, r, 1.0)    # 1e39 < 1e30 * 1e10
    assert solver.iterateDiverged(u, np.array([1.0e41, 0.0]), 1.0)
