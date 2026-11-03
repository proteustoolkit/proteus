#!/usr/bin/env python
"""NCLS edge-based (STAB=2 smoothness indicator, the default, or STAB=1 entropy viscosity) disk rotation of
test/LS_with_edgeBased_EV, to t=TFINAL (0.08), for the partition-independence test.

Run under mpiexec by test_ncls_edge_partition_independence.py in an empty directory, with PETSC_OPTIONS tightening
the KSP of the mass matrix solves; leaves u.csv there (owned nodes, sorted by x, y). The mesh is unstructured and
the partition is pinned to a cut x + 0.37 y >= 0.62 through the interface. NOSCAT=1 turns off the owner-to-ghost
scatter of psi and of the entropy residual, as a control. REDIST=True / COUPEZ=True turn on the redistancing and the
Coupez term; NOPSISCAT=1 turns off the scatter of psi in the redistancing residual, as a control. (Entropy viscosity needs a shorter TFINAL: it has a
discontinuous branch, etaMin == etaMax, that amplifies round-off differences step by step.)
"""
import os, shutil, sys, runpy
import numpy as np
from proteus import Comm, MeshTools as MT
from mpi4py import MPI

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LS_with_edgeBased_EV", "NCLS")
sys.path.insert(0, os.getcwd())
for name in ("ncls.py", "ncls_p.py", "ncls_n.py"):
    text = open(os.path.join(SRC, name)).read().replace("from .ncls", "from ncls")
    if name == "ncls.py":
        text = text.replace("parallel = False", "parallel = True")
    if name == "ncls_n.py":
        # converge the mass matrix solves far enough for a comparison to 1e-8
        text += "\nnl_atol_res = 1.0e-13; l_atol_res = 1.0e-15; linTolFac = 0.0; tolFac = 0.0\n"
    open(name, "w").write(text)

_partition = MT.Mesh.partitionMesh
def partitionMesh(self, nLayersOfOverlap=1, parallelPartitioningType=MT.MeshParallelPartitioningTypes.node,
                  nodeOwners=None, elementOwners=None, nodeWeights=None):
    if (nodeOwners is None and Comm.get().size() > 1
            and parallelPartitioningType == MT.MeshParallelPartitioningTypes.node):
        nodeOwners = (self.nodeArray[:, 0] + 0.37 * self.nodeArray[:, 1] >= 0.62).astype(np.intc)
    return _partition(self, nLayersOfOverlap=nLayersOfOverlap, parallelPartitioningType=parallelPartitioningType,
                      nodeOwners=nodeOwners, elementOwners=elementOwners, nodeWeights=nodeWeights)
MT.Mesh.partitionMesh = partitionMesh

from proteus.mprans import NCLS
if os.environ.get("NOPSISCAT"):
    # control: the redistancing residual reads ghost psi from truncated rows again
    _scatter = NCLS.LevelModel.scatterDOFArray
    NCLS.LevelModel.scatterDOFArray = lambda self, a: (
        None if a is getattr(self, "_redistancing_psi_dof", None) else _scatter(self, a))
if os.environ.get("NOSCAT"):
    NCLS.LevelModel.scatterDOFArray = lambda self, a: None
from proteus import NumericalSolution as NS
_calculateSolution = NS.NS_base.calculateSolution
def calculateSolution(self, *args, **kwargs):
    result = _calculateSolution(self, *args, **kwargs)
    model = self.modelList[0].levelModelList[-1]
    nOwned = model.mesh.nNodes_owned
    comm = Comm.get().comm.tompi4py()
    # scatterDOFArray must not keep a vector per array it was handed (it kept 1080 per rank)
    nVecs = comm.allreduce(len(model._dofVecs), op=MPI.MAX)
    gathered = comm.gather(np.column_stack([model.mesh.nodeArray[:nOwned, :2], model.u[0].dof[:nOwned]]), root=0)
    if comm.rank == 0:
        a = np.vstack(gathered)
        np.savetxt("u.csv", a[np.lexsort((a[:, 1], a[:, 0]))], delimiter=",", header="x,y,u")
        open("dofvecs.txt", "w").write("%d\n" % nVecs)
    return result
NS.NS_base.calculateSolution = calculateSolution

sys.argv = ['parun', 'ncls_p.py', 'ncls_n.py', '-l1', '-C',
            'problem=1 STABILIZATION_TYPE=' + os.environ.get('STAB', '2') + ' DO_REDISTANCING=' + os.environ.get('REDIST', 'False') + ' COUPEZ=' + os.environ.get('COUPEZ', 'False') + ' unstructured=True refinement=2 T=' + os.environ.get('TFINAL', '0.08') + ' nDTout=1']
runpy.run_path(shutil.which('parun'), run_name='__main__')
