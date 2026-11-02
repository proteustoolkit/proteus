#!/usr/bin/env python
"""A test/TwoPhaseFlow deck (DECK, with OPTS for its context options) to t=FT, for the partition tests.

Run under mpiexec by test_twophaseflow_partition_independence.py in an empty directory; copies the deck and the stored
meshes next to it and leaves u.csv there: one row per owned dof of every model component, sorted by
(100 * model + component, x, y): key, x, y, value, and models.csv: key, model name. ATOL=a sets every model's
nonlinear atol to a and its linear atol to a/100 (relative tolerances off); without it the deck's own tolerances (nl_atol_res ~1e-3) apply.
"""
import glob, os, runpy, shutil, sys
import numpy as np

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TwoPhaseFlow")
DECK = os.environ.get("DECK", "damBreak.py")
for f in [os.path.join(SRC, DECK)] + glob.glob(os.path.join(SRC, "mesh*")):
    shutil.copy(f, ".")

from proteus import Comm
from proteus import NumericalSolution as NS

_init = NS.NS_base.__init__
def init(self, so, pList, nList, *args, **kwargs):
    if os.environ.get("ATOL"):
        atol = float(os.environ["ATOL"])
        for n in nList:
            n.nl_atol_res, n.l_atol_res, n.l_rtol_res, n.linTolFac, n.tolFac = atol, 0.01 * atol, 0.0, 0.0, 0.0
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
        with open("models.csv", "w") as f:
            for mi, mod in enumerate(self.modelList):
                lm = mod.levelModelList[-1]
                for ci in range(lm.nc):
                    f.write("%d,%s\n" % (100 * mi + ci, lm.name))
    return r
NS.NS_base.calculateSolution = calc

FT = os.environ.get("FT", "0.1")
sys.argv = ['parun', '--TwoPhaseFlow', DECK, '-l1', '-C',
            'final_time=%s dt_output=%s %s' % (FT, FT, os.environ.get("OPTS", "he=0.1"))]
runpy.run_path(shutil.which('parun'), run_name='__main__')
