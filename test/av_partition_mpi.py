#!/usr/bin/env python
"""damBreak (TwoPhaseFlow, ARTIFICIAL_VISCOSITY=3) to t=0.1, for the partition-independence test.

Run under mpiexec by test_av_partition_independence.py in an empty directory; leaves pressure.csv there.
The mesh is the stored one and the partition is pinned: node owners x >= XCUT, a cut through the region the
interface sweeps. The solvers are converged far enough that two runs can be compared to 1e-8.
NOFIX=1 turns off the owner-to-ghost scatter of the smoothness indicator, as a control.
"""
import os, shutil, sys, runpy
import numpy as np
from proteus import Comm, MeshTools as MT

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "TwoPhaseFlow")
for name in os.listdir(SRC):
    if name.startswith("meshDambreak") or name == "__init__.py":
        shutil.copy(os.path.join(SRC, name), name)
case = open(os.path.join(SRC, "damBreak.py")).read()
case += ("\n[(setattr(mm.n, 'nl_atol_res', 1.0e-10), setattr(mm.n, 'l_atol_res', 1.0e-14), "
         "(setattr(mm.n, 'linTolFac', 1.0e-14) if hasattr(mm.n, 'linTolFac') else None)) "
         "for mm in myTpFlowProblem.SystemPhysics.modelDict.values()]\n")
open("damBreak.py", "w").write(case)

if os.environ.get("NOFIX"):
    from proteus.mprans import RANS3PF
    _scatter = RANS3PF.LevelModel.scatterDOFArray
    RANS3PF.LevelModel.scatterDOFArray = lambda self, a: None if a is self.uStar_psi_dof else _scatter(self, a)

XCUT = 1.6
_partition = MT.Mesh.partitionMesh
def partitionMesh(self, nLayersOfOverlap=1, parallelPartitioningType=MT.MeshParallelPartitioningTypes.node,
                  nodeOwners=None, elementOwners=None, nodeWeights=None):
    if (nodeOwners is None and Comm.get().size() > 1
            and parallelPartitioningType == MT.MeshParallelPartitioningTypes.node):
        nodeOwners = (self.nodeArray[:, 0] >= XCUT).astype(np.intc)
    return _partition(self, nLayersOfOverlap=nLayersOfOverlap, parallelPartitioningType=parallelPartitioningType,
                      nodeOwners=nodeOwners, elementOwners=elementOwners, nodeWeights=nodeWeights)
MT.Mesh.partitionMesh = partitionMesh

sys.argv = ['parun', '--TwoPhaseFlow', 'damBreak.py', '-l1', '-C', 'final_time=0.1 dt_output=0.1 he=0.1']
runpy.run_path(shutil.which('parun'), run_name='__main__')
