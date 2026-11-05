#!/usr/bin/env python
"""TADR edge-based (STAB=3 smoothness indicator, the default, STAB=2 entropy viscosity, STAB=4, or STAB=5 implicit
entropy viscosity with FCT after the Newton solve) rotation of the disk
of test/TADR, to t=TFINAL (0.08), for the partition-independence test.

Run under mpiexec by test_tadr_partition_independence.py in an empty directory, with PETSC_OPTIONS tightening the KSP
of the mass matrix solves; leaves u.csv there (owned nodes, sorted by x, y). The mesh is unstructured and the partition
is pinned to a cut x + 0.37 y >= 0.62 through the disk (on more ranks, equal strips along x + 0.37 y). LUMPED=1 uses the lumped mass matrix. NOSCAT=1 turns off every
owner-to-ghost scatter, as a control. GHOSTCFL=1 puts garbage in the edge based cfl of the ghost dofs; NOREDUCE=1 keeps each rank's own entropy normalization (STAB=1, a control). OLDMAX=1 makes
the time step read it again (a control). POROUS=1 makes the porosity vary in space; NOTHETASCAT=1 keeps the partial
nodal porosity at the ghosts (a control).
"""
import os, shutil, sys, runpy
import numpy as np
from proteus import Comm, MeshTools as MT

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TADR")
sys.path.insert(0, os.getcwd())
for name in ("thelper_tadr.py", "thelper_tadr_p.py", "thelper_tadr_n.py"):
    text = open(os.path.join(SRC, name)).read()
    if name == "thelper_tadr.py":
        text = text.replace("parallel = False", "parallel = True")
    if name == "thelper_tadr_n.py":
        # converge the mass matrix solves far enough for a comparison to 1e-8
        text += "\nnl_atol_res = 1.0e-13; l_atol_res = 1.0e-15; linTolFac = 0.0; tolFac = 0.0\n"
        # STAB=5 (ImplicitEV) is implicit: backward Euler and Newton, with FCT applied after the solve
        text += ("\nif ct.STABILIZATION_TYPE == 5:\n    levelNonlinearSolver = Newton\n    fullNewtonFlag = True\n"
                 "    updateJacobian = True\n    timeIntegration = BackwardEuler_cfl\n    nl_atol_res = 1.0e-10\n"
                 "    l_atol_res = 1.0e-12\n    maxNonlinearIts = 100\n")
    open(name, "w").write(text)

_partition = MT.Mesh.partitionMesh
def partitionMesh(self, nLayersOfOverlap=1, parallelPartitioningType=MT.MeshParallelPartitioningTypes.node,
                  nodeOwners=None, elementOwners=None, nodeWeights=None):
    if (nodeOwners is None and Comm.get().size() > 1
            and parallelPartitioningType == MT.MeshParallelPartitioningTypes.node):
        cut = self.nodeArray[:, 0] + 0.37 * self.nodeArray[:, 1]
        if Comm.get().size() == 2:
            nodeOwners = (cut >= 0.62).astype(np.intc)
        else:  # strips along the same direction, equal node counts
            edges = np.quantile(cut, np.linspace(0.0, 1.0, Comm.get().size() + 1)[1:-1])
            nodeOwners = np.searchsorted(edges, cut, side="right").astype(np.intc)
    return _partition(self, nLayersOfOverlap=nLayersOfOverlap, parallelPartitioningType=parallelPartitioningType,
                      nodeOwners=nodeOwners, elementOwners=elementOwners, nodeWeights=nodeWeights)
MT.Mesh.partitionMesh = partitionMesh

from proteus.mprans import TADR
if os.environ.get("POROUS"):
    # a porosity that varies in space, so the nodal projection of it differs from dof to dof
    def _theta(x):
        return 0.6 + 0.3 * np.sin(4.0 * x[..., 0]) * np.cos(3.0 * x[..., 1])
    _init = TADR.LevelModel.__init__
    def __init__(self, *args, **kwargs):
        _init(self, *args, **kwargs)
        c = self.coefficients
        c.q_porosity = _theta(self.q['x'])
        c.q_porosity_old = c.q_porosity.copy()
        c.ebqe_porosity = _theta(self.ebqe['x'])
    TADR.LevelModel.__init__ = __init__
if os.environ.get("NOTHETASCAT"):
    # control: keep the partial nodal porosity at the ghosts. In getResidual and FCTStep the scatter of it is the
    # first scatter after the first kernel call (the phase -1 call), so skip that one.
    class _Kernel(object):
        def __init__(self, kernel, model):
            self._kernel, self._model = kernel, model
        def __getattr__(self, name):
            return getattr(self._kernel, name)
        def calculateResidual(self, args):
            self._model._kernelCalls += 1
            return self._kernel.calculateResidual(args)
        def FCTStep(self, args):
            self._model._kernelCalls += 1
            return self._kernel.FCTStep(args)
    _scatter = TADR.LevelModel.scatterDOFArray
    def scatterDOFArray(self, a):
        if getattr(self, "_kernelCalls", 0) == 1 and not self._skipped:
            self._skipped = True
            return
        return _scatter(self, a)
    TADR.LevelModel.scatterDOFArray = scatterDOFArray
    def _counting(method):
        def wrapped(self, *args):
            if not isinstance(self.adr, _Kernel):
                self.adr = _Kernel(self.adr, self)
            self._kernelCalls, self._skipped = 0, False
            try:
                return method(self, *args)
            finally:
                self._kernelCalls = 0
        return wrapped
    TADR.LevelModel.getResidual = _counting(TADR.LevelModel.getResidual)
    TADR.LevelModel.FCTStep = _counting(TADR.LevelModel.FCTStep)
if os.environ.get("NOSCAT"):
    TADR.LevelModel.scatterDOFArray = lambda self, a: None
if os.environ.get("GHOSTCFL"):
    _getResidual = TADR.LevelModel.getResidual
    def getResidual(self, u, r):
        out = _getResidual(self, u, r)
        nOwned = self.mesh.nNodes_owned
        self.edge_based_cfl[nOwned:] = 3.0 * self.edge_based_cfl[:nOwned].max()
        return out
    TADR.LevelModel.getResidual = getResidual
if os.environ.get("NOREDUCE"):
    # each rank normalizes the Taylor-Galerkin entropy viscosity by its own subdomain (a control)
    from proteus.mprans import EntropyViscosity
    EntropyViscosity.reduce_numDiff = lambda *args: None
if os.environ.get("OLDMAX"):
    TADR.RKEV.ownedMaxCFL = lambda self: self.cfl.max()

from proteus import NumericalSolution as NS
_calculateSolution = NS.NS_base.calculateSolution
def calculateSolution(self, *args, **kwargs):
    result = _calculateSolution(self, *args, **kwargs)
    model = self.modelList[0].levelModelList[-1]
    nOwned = model.mesh.nNodes_owned
    comm = Comm.get().comm.tompi4py()
    gathered = comm.gather(np.column_stack([model.mesh.nodeArray[:nOwned, :2], model.u[0].dof[:nOwned]]), root=0)
    if comm.rank == 0:
        a = np.vstack(gathered)
        np.savetxt("u.csv", a[np.lexsort((a[:, 1], a[:, 0]))], delimiter=",", header="x,y,u")
    return result
NS.NS_base.calculateSolution = calculateSolution

sys.argv = ['parun', 'thelper_tadr_p.py', 'thelper_tadr_n.py', '-l1', '-C',
            'problem=1 nd=2 unstructured=True refinement=2 nDTout=1 ENTROPY_TYPE=1 cE=1.0 '
            + 'FCT=' + ('False' if os.environ.get('STAB') == '1' else 'True') + ' '
            'STABILIZATION_TYPE=' + os.environ.get('STAB', '3') + ' T=' + os.environ.get('TFINAL', '0.08')
            + ' LUMPED_MASS_MATRIX=' + ('True' if os.environ.get('LUMPED') else 'False')
            + ' ' + os.environ.get('OPTS', '')]
runpy.run_path(shutil.which('parun'), run_name='__main__')
