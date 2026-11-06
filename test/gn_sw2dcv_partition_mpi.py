#!/usr/bin/env python
"""GN_SW2DCV circular_damBreak (sw_model=1, lumped mass matrix) to t=0.5, for the partition-independence test.

Unstructured mesh (he=0.4) with a pinned slanted cut. Run under mpiexec by test_sw2dcv_partition_independence.py in an empty directory; leaves u.csv there (owned nodes,
sorted by x, y: x, y, then the six unknowns). NOSCAT=1 turns off the owner-to-ghost scatters of the
low-order solution of the limiter and the limited solution; OLDVOL=1 sets the domain size to the volume of the subdomain. Both are controls.
"""
import os, shutil, sys, runpy
import numpy as np
from proteus import Comm
import rank_staging

# every rank stages the same files: atomically (rank_staging)
rank_staging.write("circular_damBreak.py", open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "SWFlow",
                                                          "circular_damBreak.py")).read()
                   + "\nmySWFlowProblem.GN_swe_parameters['LUMPED_MASS_MATRIX'] = 1\n")
rank_staging.write("__init__.py", "")

from proteus.mprans import GN_SW2DCV as SW2DCV
if os.environ.get("NOSCAT"):
    SW2DCV.LevelModel.scatterDOFArray = lambda self, a: None
if os.environ.get("OLDVOL"):
    _init = SW2DCV.LevelModel.initDataStructures
    def init(self):
        _init(self)
        self.size_of_domain = self.mesh.volume
    SW2DCV.LevelModel.initDataStructures = init

from proteus import MeshTools as MT
_partition = MT.Mesh.partitionMesh
def partitionMesh(self, nLayersOfOverlap=1, parallelPartitioningType=MT.MeshParallelPartitioningTypes.node,
                  nodeOwners=None, elementOwners=None, nodeWeights=None):
    # a slanted cut through the dam, which the graph partitioner would not reproduce from run to run
    if (nodeOwners is None and Comm.get().size() > 1
            and parallelPartitioningType == MT.MeshParallelPartitioningTypes.node):
        nodeOwners = (self.nodeArray[:, 0] + 0.37 * self.nodeArray[:, 1] >= 10.3).astype(np.intc)
    return _partition(self, nLayersOfOverlap=nLayersOfOverlap, parallelPartitioningType=parallelPartitioningType,
                      nodeOwners=nodeOwners, elementOwners=elementOwners, nodeWeights=nodeWeights)
MT.Mesh.partitionMesh = partitionMesh

if os.environ.get("GHOSTCFL"):
    # garbage in the edge based cfl of the ghost dofs, which the time step must not read
    class _Proxy(object):
        def __init__(self, kernel, model):
            self._kernel, self._model = kernel, model
        def __getattr__(self, name):
            return getattr(self._kernel, name)
        def calculateEdgeBasedCFL(self, args):
            result = self._kernel.calculateEdgeBasedCFL(args)
            self._model.edge_based_cfl[self._model.mesh.nNodes_owned:] = 100.0
            return result
    _init = SW2DCV.LevelModel.initializeTimeIntegration if False else None
    _initDataStructures = SW2DCV.LevelModel.initDataStructures
    def initDataStructures(self):
        _initDataStructures(self)
        if not isinstance(self.dsw_2d, _Proxy):
            self.dsw_2d = _Proxy(self.dsw_2d, self)
    SW2DCV.LevelModel.initDataStructures = initDataStructures
if os.environ.get("OLDMAX"):
    SW2DCV.RKEV.ownedMaxCFL = lambda self: self.edge_based_cfl.max()

from proteus import NumericalSolution as NS
_calc = NS.NS_base.calculateSolution
def calc(self, *a, **k):
    r = _calc(self, *a, **k)
    m = self.modelList[0].levelModelList[-1]
    nOwned = m.mesh.nNodes_owned
    comm = Comm.get().comm.tompi4py()
    cols = [m.mesh.nodeArray[:nOwned, :2]] + [m.u[i].dof[:nOwned, None] for i in range(6)]
    gathered = comm.gather(np.hstack(cols), root=0)
    if comm.rank == 0:
        a = np.vstack(gathered)
        np.savetxt("u.csv", a[np.lexsort((a[:, 1], a[:, 0]))], delimiter=",", header="x,y,h,hu,hv,heta,hw,hbeta")
    return r
NS.NS_base.calculateSolution = calc

sys.argv = ['parun', '--SWEs', 'circular_damBreak.py', '-l1', '-C', 'sw_model=1 final_time=0.5 dt_output=0.5 structured=False he=0.4']
runpy.run_path(shutil.which('parun'), run_name='__main__')
