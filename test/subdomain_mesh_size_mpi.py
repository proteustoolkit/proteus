#!/usr/bin/env python
"""NCLS disk rotation of test/LS_with_edgeBased_EV with the Coupez term (smoothness indicator, no redistancing loop),
to t=TFINAL (0.08), for test_subdomain_mesh_size.py.

The Coupez term and NCLS's sign() are scaled by epsFactRedistancing * mesh.h, so they see the subdomain mesh's h.
Run under mpiexec in an empty directory; leaves u.csv (owned nodes, sorted by x, y) and h.csv (one row per rank:
h, hMin of the mesh the model holds). The partition is pinned to slanted bands of x + 0.37 y with equal node counts. LOCALH=1 keeps each
rank's own h and hMin (the old behaviour), as a control.
"""
import os, shutil, sys, runpy
import numpy as np
from proteus import Comm, MeshTools as MT
import rank_staging  # every rank stages the same files: atomically

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LS_with_edgeBased_EV", "NCLS")
sys.path.insert(0, os.getcwd())
for name in ("ncls.py", "ncls_p.py", "ncls_n.py"):
    text = open(os.path.join(SRC, name)).read().replace("from .ncls", "from ncls")
    if name == "ncls.py":
        text = text.replace("parallel = False", "parallel = True")
    if name == "ncls_n.py":
        text += "\nnl_atol_res = 1.0e-13; l_atol_res = 1.0e-15; linTolFac = 0.0; tolFac = 0.0\n"
    rank_staging.write(name, text)

_partition = MT.Mesh.partitionMesh
def partitionMesh(self, nLayersOfOverlap=1, parallelPartitioningType=MT.MeshParallelPartitioningTypes.node,
                  nodeOwners=None, elementOwners=None, nodeWeights=None):
    if (nodeOwners is None and Comm.get().size() > 1
            and parallelPartitioningType == MT.MeshParallelPartitioningTypes.node):
        # slanted bands with equal node counts, so every rank owns some
        s = self.nodeArray[:, 0] + 0.37 * self.nodeArray[:, 1]
        cuts = np.quantile(s, np.arange(1, Comm.get().size()) / Comm.get().size())
        nodeOwners = np.searchsorted(cuts, s, side='right').astype(np.intc)
    return _partition(self, nLayersOfOverlap=nLayersOfOverlap, parallelPartitioningType=parallelPartitioningType,
                      nodeOwners=nodeOwners, elementOwners=elementOwners, nodeWeights=nodeWeights)
MT.Mesh.partitionMesh = partitionMesh
if os.environ.get("LOCALH"):
    MT.Mesh._reduceSubdomainMeshSize = lambda self, comm: None

from proteus import NumericalSolution as NS
_calculateSolution = NS.NS_base.calculateSolution
def calculateSolution(self, *args, **kwargs):
    result = _calculateSolution(self, *args, **kwargs)
    model = self.modelList[0].levelModelList[-1]
    nOwned = model.mesh.nNodes_owned
    comm = Comm.get().comm.tompi4py()
    gathered = comm.gather(np.column_stack([model.mesh.nodeArray[:nOwned, :2], model.u[0].dof[:nOwned]]), root=0)
    sizes = comm.gather([model.mesh.h, model.mesh.hMin], root=0)
    diam = comm.gather((comm.rank, nOwned, int(model.mesh.nElements_global), float(model.mesh.elementDiametersArray.max()),
                        float(model.mesh.elementDiametersArray.min()), model.mesh.h, model.mesh.hMin), root=0)
    if comm.rank == 0:
        # (rank, owned nodes, elements held, largest and smallest element held, mesh.h, mesh.hMin): printed by the test
        # if a control does not differ
        open("diag.txt", "w").write(repr(diam))
        a = np.vstack(gathered)
        np.savetxt("u.csv", a[np.lexsort((a[:, 1], a[:, 0]))], delimiter=",", header="x,y,u")
        np.savetxt("h.csv", np.array(sizes), delimiter=",", header="h,hMin")
    return result
NS.NS_base.calculateSolution = calculateSolution

sys.argv = ['parun', 'ncls_p.py', 'ncls_n.py', '-l1', '-C',
            'problem=1 STABILIZATION_TYPE=2 DO_REDISTANCING=False COUPEZ=True unstructured=True refinement=2 T='
            + os.environ.get('TFINAL', '0.08') + ' nDTout=1']
runpy.run_path(shutil.which('parun'), run_name='__main__')
