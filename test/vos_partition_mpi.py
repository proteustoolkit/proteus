#!/usr/bin/env python
"""VOS3P edge-based FCT (STABILIZATION_TYPE=4), advection of the test/VOS pulse on the unit square (unstructured,
he=1/40, to t=0.4), for the partition-independence test.

Run under mpiexec by test_vos_partition_independence.py in an empty directory; leaves u.csv there (owned nodes,
sorted by x, y). The partition is pinned to a slanted cut x + 0.37 y >= 0.40 through the pulse. Controls:
NOFIX=1 turns off the owner-to-ghost scatter of uLow. GHOSTCFL=1 puts garbage in the edge based cfl of the ghost
dofs; OLDMAX=1 makes the time step read it again (a control).
"""
import os, shutil, sys, runpy
import numpy as np
from proteus import Comm, MeshTools as MT
import rank_staging  # every rank stages the same files: atomically

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "VOS")
for name in ("vos.py", "vos_p.py", "vos_n.py"):
    text = open(os.path.join(SRC, name)).read()
    text = text.replace("parallel = False", "parallel = True")
    text = text.replace("ct.T=1.0\n    ct.nDTout=20", "ct.T=0.4\n    ct.nDTout=8")
    text = text.replace("unstructured=False", "unstructured=True").replace("nny=int((nnx-1)/10+1)", "nny=nnx")
    text = text.replace("0.1 if ct.problem==0 else 1.0", "1.0")
    rank_staging.write(name, text)

_partition = MT.Mesh.partitionMesh
def partitionMesh(self, nLayersOfOverlap=1, parallelPartitioningType=MT.MeshParallelPartitioningTypes.node,
                  nodeOwners=None, elementOwners=None, nodeWeights=None):
    if (nodeOwners is None and Comm.get().size() > 1
            and parallelPartitioningType == MT.MeshParallelPartitioningTypes.node):
        nodeOwners = (self.nodeArray[:, 0] + 0.37 * self.nodeArray[:, 1] >= 0.40).astype(np.intc)
    return _partition(self, nLayersOfOverlap=nLayersOfOverlap, parallelPartitioningType=parallelPartitioningType,
                      nodeOwners=nodeOwners, elementOwners=elementOwners, nodeWeights=nodeWeights)
MT.Mesh.partitionMesh = partitionMesh

from proteus.mprans import VOS3P
if os.environ.get("NOFIX"):
    VOS3P.LevelModel.scatterDOFArray = lambda self, a: None
if os.environ.get("GHOSTCFL"):
    _getResidual = VOS3P.LevelModel.getResidual
    def getResidual(self, u, r):
        out = _getResidual(self, u, r)
        self.edge_based_cfl[self.mesh.nNodes_owned:] = 300.0
        return out
    VOS3P.LevelModel.getResidual = getResidual
if os.environ.get("OLDMAX"):
    VOS3P.RKEV.ownedMaxCFL = lambda self: self.cfl.max()

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

sys.argv = ['parun', 'vos_p.py', 'vos_n.py', '-l1', '-C', 'problem=0 STABILIZATION_TYPE=4 refinement=2']
runpy.run_path(shutil.which('parun'), run_name='__main__')
