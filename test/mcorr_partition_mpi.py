#!/usr/bin/env python
"""Edge-based mass correction with FCT (NCLS + VOF + MCorr, test/LS_with_edgeBased_EV/MCorr) to the end of the first
steps, for the partition-independence test.

Run under mpiexec by test_mcorr_partition_independence.py in an empty directory, with PETSC_OPTIONS tightening the
KSPs; leaves final.npz there (the owned dofs of the three models, sorted by x, y). PAR=False uses the LU solvers (serial
only), PAR=True the KSP ones. The mesh is unstructured and the partition is pinned to a cut x + 0.37 y >= 0.62.
NOFIX=1 turns off the owner-to-ghost scatters of MCorr, as a control.
"""
import os, shutil, sys
import numpy as np
from proteus import Comm, MeshTools as MT, default_s

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LS_with_edgeBased_EV", "MCorr")
os.makedirs("mcpkg", exist_ok=True)
for name in os.listdir(SRC):
    if (name.startswith("thelper") and name.endswith(".py")) or name == "__init__.py":
        text = open(os.path.join(SRC, name)).read()
        if name == "thelper_cons_ls.py":
            text = text.replace("parallel = False", "parallel = " + os.environ.get("PAR", "True"))
        open(os.path.join("mcpkg", name), "w").write(text)
sys.path.insert(0, os.getcwd())

import proteus

_partition = MT.Mesh.partitionMesh
def partitionMesh(self, nLayersOfOverlap=1, parallelPartitioningType=MT.MeshParallelPartitioningTypes.node,
                  nodeOwners=None, elementOwners=None, nodeWeights=None):
    if (nodeOwners is None and Comm.get().size() > 1
            and parallelPartitioningType == MT.MeshParallelPartitioningTypes.node):
        nodeOwners = (self.nodeArray[:, 0] + 0.37 * self.nodeArray[:, 1] >= 0.62).astype(np.intc)
    return _partition(self, nLayersOfOverlap=nLayersOfOverlap, parallelPartitioningType=parallelPartitioningType,
                      nodeOwners=nodeOwners, elementOwners=elementOwners, nodeWeights=nodeWeights)
MT.Mesh.partitionMesh = partitionMesh

from mcpkg import thelper_cons_ls
# NCLS with the smoothness indicator, no redistancing and no Coupez term: the redistancing loop has its own
# parallel differences, which are not what is tested here
thelper_cons_ls.ct.STABILIZATION_TYPE_ncls = 2
thelper_cons_ls.ct.DO_REDISTANCING = False
thelper_cons_ls.ct.COUPEZ = False
thelper_cons_ls.ct.STABILIZATION_TYPE_vof = 3
thelper_cons_ls.ct.unstructured = True
thelper_cons_ls.ct.refinement = 1
from mcpkg import (thelper_cons_ls_so, thelper_ncls_p, thelper_ncls_n, thelper_vof_p, thelper_vof_n,
                   thelper_MCorr_p, thelper_MCorr_n)
from proteus.mprans import MCorr
if os.environ.get("NOFIX"):
    MCorr.LevelModel.scatterDOFArray = lambda self, a: None

from proteus import NumericalSolution as NS
_calculateSolution = NS.NS_base.calculateSolution
def calculateSolution(self, *args, **kwargs):
    result = _calculateSolution(self, *args, **kwargs)
    comm = Comm.get().comm.tompi4py()
    out = {}
    for index, model in enumerate(self.modelList):
        level = model.levelModelList[-1]
        nOwned = level.mesh.nNodes_owned
        gathered = comm.gather(np.column_stack([level.mesh.nodeArray[:nOwned, :2], level.u[0].dof[:nOwned]]), root=0)
        if comm.rank == 0:
            a = np.vstack(gathered)
            out["m%d" % index] = a[np.lexsort((a[:, 1], a[:, 0]))]
    if comm.rank == 0:
        np.savez("final.npz", **out)
    return result
NS.NS_base.calculateSolution = calculateSolution

_init = NS.NS_base.__init__
def init(self, so, pList, nList, *args, **kwargs):
    # converge every solve far enough for a comparison to 1e-8
    for n in nList:
        n.nl_atol_res, n.l_atol_res, n.l_rtol_res, n.linTolFac, n.tolFac = 1.0e-13, 1.0e-15, 0.0, 0.0, 0.0
    return _init(self, so, pList, nList, *args, **kwargs)
NS.NS_base.__init__ = init

from proteus.iproteus import opts
pList, nList, sList = [], [], []
for p, n in ((thelper_ncls_p, thelper_ncls_n), (thelper_vof_p, thelper_vof_n), (thelper_MCorr_p, thelper_MCorr_n)):
    pList.append(p)
    if p.name is None:
        p.name = p.__name__
    nList.append(n)
    sList.append(default_s)
so = thelper_cons_ls_so
so.name += "_partition"
ns = proteus.NumericalSolution.NS_base(so, pList, nList, sList, opts)
ns.calculateSolution('mcorr')
