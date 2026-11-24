#!/usr/bin/env python
"""SW2DCV circular_damBreak (SWEs, lumped mass matrix) to t = 0.3 with one of the switches no deck sets, for test_sw2dcv_flags.py.

VARIANT names the switch (VARIANTS below): the numerical flags LOW_ORDER_ONLY, NO_STAR_STATES, WALL_RIEMANN, NODAL_H_BOUNDS,
BOUNDS_RELAXATION=0 with BOUNDS_EXCLUDE_SELF=1, LINEAR_FRICTION, a tailwater stage on the right edge, a walls function that leaves
the right edge open, a porosity with a Darcy-Forchheimer drag law, the consistent mass matrix, the dispersive model (gn, gn_consistent: sw_model=1). Unstructured mesh (he = 0.6) with the slanted pinned cut of
sw2dcv_partition_mpi.py. Run in an empty directory, serially or under mpiexec; rank 0 leaves u.csv (the owned nodes, sorted by x, y:
x, y, h, hu, hv, ML) and total.txt (the water volume at the start and at the end).
"""
import os, shutil, sys, runpy
import numpy as np
from proteus import Comm
import rank_staging

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANT = os.environ["VARIANT"]
VARIANTS = {
    "default": "",
    "low_order_only": "mySWFlowProblem.swe_parameters['LOW_ORDER_ONLY'] = 1",
    "no_star_states": "mySWFlowProblem.swe_parameters['NO_STAR_STATES'] = 1",
    "wall_riemann": "mySWFlowProblem.swe_parameters['WALL_RIEMANN'] = 1",
    "nodal_h_bounds": "mySWFlowProblem.swe_parameters['NODAL_H_BOUNDS'] = 1",
    "no_relaxation": "mySWFlowProblem.swe_parameters['BOUNDS_RELAXATION'] = 0\n"
                     "mySWFlowProblem.swe_parameters['BOUNDS_EXCLUDE_SELF'] = 1",
    "linear_friction": "mySWFlowProblem.physical_parameters['LINEAR_FRICTION'] = 1\n"
                       "mySWFlowProblem.physical_parameters['mannings'] = 0.05",
    "tailwater": "boundaryConditions['tailwater'] = {'where': lambda X: X[0] > 19.99,\n"
                 "                                   'stage': lambda x, y, t: 0.9 + 0.0 * x}",
    "walls_function": "boundaryConditions['walls'] = lambda X: X[0] < 19.99",
    "consistent": "mySWFlowProblem.swe_parameters['LUMPED_MASS_MATRIX'] = 0",
    "gn": "mySWFlowProblem.GN_swe_parameters['LUMPED_MASS_MATRIX'] = 1",
    "gn_consistent": "mySWFlowProblem.GN_swe_parameters['LUMPED_MASS_MATRIX'] = 0",
    "drag_law": "mySWFlowProblem.physical_parameters['porosity'] = 0.8\n"
                "mySWFlowProblem.physical_parameters['drag_alpha'] = 0.5\n"
                "mySWFlowProblem.physical_parameters['drag_beta'] = 0.2\n"
                "mySWFlowProblem.physical_parameters['drag_law'] = lambda s: 0.2 + 0.1 * np.minimum(s['age'], 1.0)",
}
EXTRA = VARIANTS[VARIANT]
# the mySWFlowProblem line of the deck is the last statement but the physical parameter line: the extras go before the problem is built
# when they set boundaryConditions, after it otherwise; one place serves both because the problem reads boundaryConditions by reference
_deck = open(os.path.join(HERE, "SWFlow", "circular_damBreak.py")).read()
if "boundaryConditions[" in EXTRA:
    _marker = "# ********************************** #\n# ***** Create mySWFlowProblem ***** #"
    assert _deck.count(_marker) == 1
    _deck = _deck.replace(_marker, EXTRA + "\n" + _marker)
    EXTRA_AFTER = ""
else:
    EXTRA_AFTER = EXTRA
rank_staging.write("circular_damBreak.py", _deck + "\nmySWFlowProblem.swe_parameters['LUMPED_MASS_MATRIX'] = 1\n" + EXTRA_AFTER + "\n")
rank_staging.write("__init__.py", "")

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

from proteus import NumericalSolution as NS
_init = NS.NS_base.__init__
_volumes = []
def init(self, so, pList, nList, *args, **kwargs):
    for n in nList:
        n.nl_atol_res, n.l_atol_res, n.l_rtol_res, n.linTolFac, n.tolFac = 1.0e-13, 1.0e-15, 0.0, 0.0, 0.0
    return _init(self, so, pList, nList, *args, **kwargs)
NS.NS_base.__init__ = init

def volume(m):
    nOwned = m.mesh.nNodes_owned
    return Comm.get().globalSum(float(np.dot(m.ML[:nOwned], m.u[0].dof[:nOwned])))

_calc = NS.NS_base.calculateSolution
def calc(self, *a, **k):
    m = self.modelList[0].levelModelList[-1]
    r = _calc(self, *a, **k)
    nOwned = m.mesh.nNodes_owned
    X = m.mesh.nodeArray[:nOwned]
    h0 = 1.0 + 0.8 * (((X[:, 0] - 10.0) ** 2 + (X[:, 1] - 10.0) ** 2) <= 4.0)   # the deck's initial depth
    v0 = Comm.get().globalSum(float(np.dot(m.ML[:nOwned], h0)))
    comm = Comm.get().comm.tompi4py()
    cols = [m.mesh.nodeArray[:nOwned, :2]] + [m.u[i].dof[:nOwned, None] for i in range(3)] + [m.ML[:nOwned, None]]
    gathered = comm.gather(np.hstack(cols), root=0)
    v1 = volume(m)
    if comm.rank == 0:
        a = np.vstack(gathered)
        np.savetxt("u.csv", a[np.lexsort((a[:, 1], a[:, 0]))], delimiter=",", header="x,y,h,hu,hv,ML")
        with open("total.txt", "w") as f:
            f.write("%.17g %.17g\n" % (v0, v1))
    if comm.size == 1:
        # serial: the Newton Jacobian the explicit SSP scheme never asks for (an implicit time integration would): the
        # model's own getJacobian, whose rows are the (consistent) mass matrix of every unknown
        import json
        m.getJacobian(m.jacobian)
        rowptr, colind, nz = m.jacobian.getCSRrepresentation()
        rows = np.repeat(np.arange(len(rowptr) - 1), np.diff(rowptr))
        J = dict(zip(zip(rows.tolist(), colind.tolist()), nz.tolist()))
        asym = max(abs(v - J.get((j, i), 0.0)) for (i, j), v in J.items())
        with open("jacobian.json", "w") as f:
            json.dump(dict(n=len(rowptr) - 1, nnz=int(len(nz)), total=float(nz.sum()), area=float(m.ML.sum()),
                           nc=int(m.nc), asymmetry=float(asym), finite=bool(np.isfinite(nz).all())), f)
    return r
NS.NS_base.calculateSolution = calc

sys.argv = ['parun', '--SWEs', 'circular_damBreak.py', '-l1', '-C', 'sw_model=%d final_time=0.3 dt_output=0.3 structured=False he=0.6' % (1 if VARIANT.startswith('gn') else 0)]
runpy.run_path(shutil.which('parun'), run_name='__main__')
