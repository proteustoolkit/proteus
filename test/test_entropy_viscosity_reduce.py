"""EntropyViscosity.reduce_numDiff: the domain-wide normalization of the explicit entropy viscosity.

On one rank the kernels' own q_numDiff_u stands; on more, reduce_numDiff recomputes it with the norm_factor of the
whole domain. A comm whose size() is 2 over MPI.COMM_SELF exercises that path in one process: the reductions are
the identity, so the expected values are the kernels' formula on the given statistics.
"""
import numpy as np
from mpi4py import MPI

from proteus.mprans import EntropyViscosity


class _TwoRanks(object):
    """Comm.get() as reduce_numDiff uses it: size() and comm.tompi4py()."""

    class _Petsc(object):
        def tompi4py(self):
            return MPI.COMM_SELF

    comm = _Petsc()

    def size(self):
        return 2


def _reduce(stats, element_viscosity, nQuad=3):
    q = np.full((len(element_viscosity), nQuad), -1.0)
    EntropyViscosity.reduce_numDiff(_TwoRanks(), np.array(stats, 'd'), np.array(element_viscosity, 'd'), q)
    return q


def test_the_entropy_viscosity_is_normalized_by_the_domain_wide_factor():
    # E in [0, 4] with mean 1: norm_factor = max(|4 - 1|, |1 - 0|) = 3
    q = _reduce([2.0, 2.0, 4.0, 0.0], [[1.0, 0.6], [0.1, 0.6]])
    np.testing.assert_allclose(q, [[0.2] * 3, [0.1] * 3], rtol=1e-15)


def test_a_uniform_entropy_keeps_the_linear_viscosity():
    """norm_factor is 0: 0/0 and x/0 give the linear viscosity, as the kernels' fmin does, never NaN."""
    q = _reduce([3.0, 1.5, 2.0, 2.0], [[0.5, 0.0], [0.7, 0.3]])
    assert np.isfinite(q).all()
    np.testing.assert_array_equal(q, [[0.5] * 3, [0.7] * 3])
