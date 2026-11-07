#!/usr/bin/env python
"""Helper run under mpiexec by test_meshPartitionCharacterisation.py; not collected.

Checks the two exchanges the parun -F partitioners use to number the nodes of
their second element read (cpartitioning.fileNodeNumbers) under a random
permutation, which the stored-mesh cases cannot reach: with
-mat_partitioning_type current the new numbering is the file numbering.

Usage: meshPartition_exchange_mpi.py N SEED. Prints ``RESULT ranks=<n> owned=<total> asked=<total>``.
"""
import sys

import numpy as np
from proteus import Comm
comm = Comm.init()
from mpi4py import MPI
from proteus import cpartitioning

mpi = comm.comm.tompi4py()
rank, size = mpi.rank, mpi.size
n, seed = int(sys.argv[1]), int(sys.argv[2])
rng = np.random.default_rng(seed)
old2new = rng.permutation(n)
off_old = np.cumsum([0] + [n // size + (n % size > r) for r in range(size)])
#uneven new ranges, one possibly empty, as a graph partitioner may give
cuts = np.sort(rng.integers(0, n + 1, size - 1))
off_new = np.r_[0, cuts, n]
chunk = old2new[off_old[rank]:off_old[rank + 1]]
wanted = np.unique(np.random.default_rng(seed + 1 + rank).integers(0, n, n // 3))
owned, answered = cpartitioning.fileNodeNumbers(mpi, off_old.tolist(), off_new.tolist(), chunk.tolist(),
                                                wanted.tolist())
mine = np.flatnonzero((old2new >= off_new[rank]) & (old2new < off_new[rank + 1]))
assert owned == {int(o): int(old2new[o]) for o in mine}, (rank, owned)
assert answered == {int(w): int(old2new[w]) for w in wanted}, (rank, answered)
totals = mpi.allreduce(np.array([len(owned), len(answered)]), op=MPI.SUM)
assert totals[0] == n
if rank == 0:
    print('RESULT ranks=%d owned=%d asked=%d' % (size, totals[0], totals[1]))

#the directory exchange of the element/face/edge numberings
#(cpartitioning.subdomainNumbers): random owners, so the owned file numbers
#are scattered and each rank's new range is uneven (possibly empty); each
#subdomain is its owned entities plus a random sample of the others
owner = rng.integers(0, size, n)
counts = np.bincount(owner, minlength=size)
expected = np.empty(n, dtype=int)
for r in range(size):
    expected[np.flatnonzero(owner == r)] = counts[:r].sum() + np.arange(counts[r])
mineOwned = np.flatnonzero(owner == rank)
subdomain = np.union1d(mineOwned, wanted)
got = cpartitioning.subdomainNumbers(mpi, n, int(counts[:rank].sum()), mineOwned.tolist(), subdomain.tolist())
assert got == {int(s): int(expected[s]) for s in subdomain}, (rank, got)
dirTotal = mpi.allreduce(len(got) - len(mineOwned), op=MPI.SUM)
if rank == 0:
    print('RESULT directory=%d' % dirTotal)
