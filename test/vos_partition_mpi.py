#!/usr/bin/env python
"""VOS3P edge-based FCT (STABILIZATION_TYPE=4), 1D advection of test/VOS to t=0.4, for the partition-independence test.

Run under mpiexec by test_vos_partition_independence.py in an empty directory; leaves u.csv there (owned nodes,
sorted by x, y). NOFIX=1 turns off the owner-to-ghost scatter of uLow, as a control.
"""
import os, shutil, sys, runpy
import numpy as np
from proteus import Comm

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "VOS")
for n in ("vos.py", "vos_p.py", "vos_n.py"):
    s = open(os.path.join(SRC, n)).read()
    s = s.replace("parallel = False", "parallel = True")
    s = s.replace("ct.T=1.0\n    ct.nDTout=20", "ct.T=0.4\n    ct.nDTout=8")
    open(n, "w").write(s)

if os.environ.get("NOFIX"):
    from proteus.mprans import VOS3P
    VOS3P.LevelModel.scatterDOFArray = lambda self, a: None

from proteus import NumericalSolution as NS
_calc = NS.NS_base.calculateSolution
def calc(self, *a, **k):
    r = _calc(self, *a, **k)
    m = self.modelList[0].levelModelList[-1]
    nOwned = m.mesh.nNodes_owned
    comm = Comm.get().comm.tompi4py()
    gathered = comm.gather(np.column_stack([m.mesh.nodeArray[:nOwned, :2], m.u[0].dof[:nOwned]]), root=0)
    if comm.rank == 0:
        a = np.vstack(gathered)
        np.savetxt("u.csv", a[np.lexsort((a[:, 1], a[:, 0]))], delimiter=",", header="x,y,u")
    return r
NS.NS_base.calculateSolution = calc

sys.argv = ['parun', 'vos_p.py', 'vos_n.py', '-l1', '-C', 'problem=0 STABILIZATION_TYPE=4 refinement=3']
runpy.run_path(shutil.which('parun'), run_name='__main__')
