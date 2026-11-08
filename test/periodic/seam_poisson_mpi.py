#!/usr/bin/env python
"""Solve seam_poisson_p/_n once (any number of ranks) and print the L2 error and the Newton iteration count.

Run by test_periodic_seam.py with SEAM_* in the environment (seam_poisson_p.py lists them). The PETSc options of a
parallel run come with the ``seam_`` prefix in PETSC_OPTIONS.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from proteus import Comm
comm = Comm.init()
from proteus import NonlinearSolvers, NumericalSolution, default_s, default_so, defaults, runtime

p = defaults.load_physics("seam_poisson_p", HERE)
n = defaults.load_numerics("seam_poisson_n", HERE)
its = []
_solve = NonlinearSolvers.Newton.solve


def solve(self, *a, **k):
    r = _solve(self, *a, **k)
    its.append(self.its)
    return r


NonlinearSolvers.Newton.solve = solve
so = default_so
so.name = p.name
so.pnList = [(p, n)]
so.sList = [default_s]
so.tnList = [0.0, 1.0]
so.archiveFlag = None
ns = NumericalSolution.NS_base(so, [p], [n], so.sList, runtime.RunOptions())
ns.calculateSolution("seam")
lm = ns.modelList[0].levelModelList[-1]
nE = getattr(lm.mesh, 'nElements_owned', lm.mesh.nElements_global)
x = lm.q['x'][:nE]
err2 = float(np.sum((lm.q[('u', 0)][:nE] - p.exact(np.moveaxis(x, -1, 0)))**2*lm.q['dV'][:nE]))
if comm.size() > 1:
    from mpi4py import MPI
    err2 = comm.comm.tompi4py().allreduce(err2, op=MPI.SUM)
    its = comm.comm.tompi4py().allreduce(max(its), op=MPI.MAX)
else:
    its = max(its)
if comm.rank() == 0:
    print("RESULT ranks=%d L2=%.17e newton=%d" % (comm.size(), np.sqrt(err2), its), flush=True)
