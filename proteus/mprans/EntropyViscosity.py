"""The domain-wide normalization of the explicit (Taylor-Galerkin) entropy viscosity, over the ranks.

The kernels with an entropy-viscosity shock capturing (NCLS3P, VOF3P, VOF, TADR) set, per element,
``numDiff = min(cMax h |v|_max, cE h^2 |R_entropy|_max / norm_factor)`` with
``norm_factor = max(|E_max - E_mean|, |E_mean - E_min|)`` over the domain. A kernel sees only its subdomain, so it
accumulates the entropy statistics over its owned elements and hands them out (``entropy_stats``: sum of E dV, sum of
dV, max E, min E) with each element's two viscosities (``element_viscosity``: linear, then entropy times
norm_factor). Its own ``q_numDiff_u`` uses the subdomain's factor; in parallel the caller replaces it here.
"""
import numpy as np


def arrays(nElements_global):
    """The kernel's output arrays: (entropy_stats, element_viscosity)."""
    return np.zeros(4, 'd'), np.zeros((nElements_global, 2), 'd')


def reduce_numDiff(comm, entropy_stats, element_viscosity, q_numDiff_u):
    """Recompute q_numDiff_u (elements x quadrature points) with the norm_factor of the whole domain."""
    if comm.size() == 1:
        return
    from mpi4py import MPI
    mpi = comm.comm.tompi4py()
    total = mpi.allreduce(np.array(entropy_stats[:2]), op=MPI.SUM)
    maxEntropy = mpi.allreduce(entropy_stats[2], op=MPI.MAX)
    minEntropy = mpi.allreduce(entropy_stats[3], op=MPI.MIN)
    meanEntropy = total[0] / total[1]
    norm_factor = max(abs(maxEntropy - meanEntropy), abs(meanEntropy - minEntropy))
    numDiff = np.minimum(element_viscosity[:, 0], element_viscosity[:, 1] / norm_factor)
    q_numDiff_u[:] = numDiff[:, None]
