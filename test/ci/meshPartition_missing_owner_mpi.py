#!/usr/bin/env python
"""Helper run under mpiexec by test_meshPartitionCharacterisation.py; not collected.

Asks the parun -F partitioners' directory lookups (askOwningRanks) for a number
no rank can answer, which a consistent set of mesh files never does, and which
used to come back as -1 (a missing owner) or as an out-of-bounds read (a number
outside the ranks' ranges) for the caller to use as a new number. Each case must
end the run on every rank with a message naming the number and the asking rank.

Usage: meshPartition_missing_owner_mpi.py CASE N, CASE one of
  directory  an entity in the last rank's subdomain that no rank owns (subdomainNumbers)
  range      an entity number nGlobal in the last rank's subdomain (subdomainNumbers)
  node       a file node number N wanted by the last rank (fileNodeNumbers)
Prints ``RESULT returned=<what came back>`` if the lookup returns, which is the failure.
"""
import sys

import numpy as np
from proteus import Comm
comm = Comm.init()
from proteus import cpartitioning

mpi = comm.comm.tompi4py()
rank, size = mpi.rank, mpi.size
case, n = sys.argv[1], int(sys.argv[2])
asker = size - 1
owner = np.arange(n) % size
missing = 0                                     #owned by rank 0, until it is dropped
if case in ('directory', 'range'):
    mine = np.flatnonzero(owner == rank)
    begin = int(np.count_nonzero(owner < rank))
    if case == 'directory':
        mine = mine[mine != missing]
    subdomain = list(mine)
    if case == 'directory':
        if rank == asker:
            subdomain = sorted(set(subdomain) | {missing})
    elif rank == asker:
        subdomain = subdomain + [n]
    got = cpartitioning.subdomainNumbers(mpi, n, begin, mine.tolist(), subdomain)
    wrong = {k: v for k, v in got.items() if k in (missing, n)} if rank == asker else {}
else:
    old2new = np.arange(n)
    off = np.cumsum([0] + [n // size + (n % size > r) for r in range(size)])
    owned, answered = cpartitioning.fileNodeNumbers(mpi, off.tolist(), off.tolist(),
                                                    old2new[off[rank]:off[rank + 1]].tolist(),
                                                    [n] if rank == asker else [])
    wrong = answered
wrong = mpi.gather(wrong, root=0)
if rank == 0:
    print('RESULT returned=%s' % repr([w for w in wrong if w]).replace(' ', ''))
