#!/usr/bin/env python
"""TwoPhaseFlow damBreak (CLSVOF + RANS3PF projection scheme, stored mesh, he=0.1) to t=0.1, for the partition test.

Run under mpiexec by test_dambreak_partition_independence.py in an empty directory; leaves u.csv there: one row per
owned dof of every model component, sorted by (100 * model + component, x, y): key, x, y, value. TIGHT=1 converges
the nonlinear and linear solves (atol 1e-12 / 1e-14); without it the deck's own tolerances (nl_atol_res ~1e-3) apply.
"""
import glob, os, runpy, shutil, sys
import numpy as np

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TwoPhaseFlow")
for f in [os.path.join(SRC, "damBreak.py")] + glob.glob(os.path.join(SRC, "meshDambreak*")):
    shutil.copy(f, ".")

from proteus import Comm
from proteus import NumericalSolution as NS

_init = NS.NS_base.__init__
def init(self, so, pList, nList, *args, **kwargs):
    if os.environ.get("TIGHT"):
        for n in nList:
            n.nl_atol_res, n.l_atol_res, n.l_rtol_res, n.linTolFac, n.tolFac = 1.0e-12, 1.0e-14, 0.0, 0.0, 0.0
    return _init(self, so, pList, nList, *args, **kwargs)
NS.NS_base.__init__ = init

_calc = NS.NS_base.calculateSolution
def calc(self, *a, **k):
    r = _calc(self, *a, **k)
    comm = Comm.get().comm.tompi4py()
    rows = []
    for mi, mod in enumerate(self.modelList):
        lm = mod.levelModelList[-1]
        for ci in range(lm.nc):
            dm = lm.u[ci].femSpace.dofMap
            offsets = dm.dof_offsets_subdomain_owned
            nOwned = int(offsets[comm.rank + 1] - offsets[comm.rank]) if comm.size > 1 else lm.u[ci].dof.shape[0]
            x = getattr(dm, "lagrangeNodesArray", lm.mesh.nodeArray)[:nOwned, :2]
            rows.append(np.hstack([np.full((nOwned, 1), 100 * mi + ci), x, lm.u[ci].dof[:nOwned, None]]))
    gathered = comm.gather(np.vstack(rows), root=0)
    if comm.rank == 0:
        a = np.vstack(gathered)
        np.savetxt("u.csv", a[np.lexsort((a[:, 2], a[:, 1], a[:, 0]))], delimiter=",", header="key,x,y,u")
    return r
NS.NS_base.calculateSolution = calc

sys.argv = ['parun', '--TwoPhaseFlow', 'damBreak.py', '-l1', '-C', 'final_time=0.1 dt_output=0.1 he=0.1']
runpy.run_path(shutil.which('parun'), run_name='__main__')
