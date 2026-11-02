#!/usr/bin/env python
"""Solve vpp_poisson_2d on one or more ranks and write what the velocity post-processing produced.

Run by test_velocity_postprocessing.py in an empty directory, serially or under mpiexec. The variant comes from
VPP_VARIANT, the space from VPP_ORDER (1 or 2). On more than one rank the node partition is pinned to the cut
x + 0.37 y >= 0.62, so serial and parallel runs are comparable element by element (VPP_FREE_PARTITION=1 leaves it to
the partitioner). VPP_NO_STAR_EXCHANGE=1 and VPP_NO_CUT_FACE_EXCHANGE=1 turn off one of the two parallel exchanges
of the PWL variants, as controls.

Writes elements.csv on rank 0: one row per owned element, sorted by centroid, with
  cx, cy, the residual VPP stores (conservationResidual), the residual recomputed here (source integral minus the
  outward flux of the post-processed velocity through the element's faces), the three face fluxes ordered by face
  midpoint, and the velocity at the element quadrature points ordered by position.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from proteus import Comm, MeshTools as MT, runtime

runtime.start()
comm = Comm.get()

_partition = MT.Mesh.partitionMesh


def partitionMesh(self, nLayersOfOverlap=1, parallelPartitioningType=MT.MeshParallelPartitioningTypes.node,
                  nodeOwners=None, elementOwners=None, nodeWeights=None):
    if (nodeOwners is None and Comm.get().size() > 1
            and parallelPartitioningType == MT.MeshParallelPartitioningTypes.node
            and not os.environ.get("VPP_FREE_PARTITION")):
        nodeOwners = (self.nodeArray[:, 0] + 0.37 * self.nodeArray[:, 1] >= 0.62).astype(np.intc)
    return _partition(self, nLayersOfOverlap=nLayersOfOverlap, parallelPartitioningType=parallelPartitioningType,
                      nodeOwners=nodeOwners, elementOwners=elementOwners, nodeWeights=nodeWeights)


MT.Mesh.partitionMesh = partitionMesh

from proteus import NumericalSolution, default_s
from proteus import PostProcessingTools

# controls: run without one of the two parallel exchanges of the PWL variants
if os.environ.get("VPP_NO_STAR_EXCHANGE"):
    PostProcessingTools.VPP_PWL_RT0.sendStarSolutionsToGhosts = lambda self, ci: None
if os.environ.get("VPP_NO_CUT_FACE_EXCHANGE"):
    PostProcessingTools.VPP_PWL_RT0.takeCutFaceVelocitiesFromInteriorRanks = lambda self, ci: None
from proteus.defaults import load_physics, load_numerics, System_base
from petsc4py import PETSc

opts_db = PETSc.Options()
opts_db.setValue("ksp_type", "cg")
opts_db.setValue("pc_type", "lu" if comm.size() == 1 else "asm")
opts_db.setValue("sub_pc_type", "lu")
opts_db.setValue("ksp_rtol", 1e-14)
opts_db.setValue("ksp_atol", 1e-40)
opts_db.setValue("ksp_max_it", 5000)

p = load_physics("vpp_poisson_2d_p", HERE)
n = load_numerics("vpp_poisson_2d_n", HERE)
so = System_base()
so.name = p.name = "vpp_poisson_2d"
so.sList = [default_s]


ns = NumericalSolution.NS_base(so, [p], [n], so.sList, runtime.RunOptions())
ns.calculateSolution("vpp")

model = ns.modelList[0].levelModelList[-1]
vpp = model.velocityPostProcessor.vpp_algorithms[0]
mesh = model.mesh
nOwned = mesh.nElements_owned
q, ebq = model.q, model.ebq

# the source integral of each element: f is the coefficient's r with the sign of -div(a grad u) + r = 0
f = -q[('r', 0)]
dV = q[('dV_u', 0)] if ('dV_u', 0) in q else q['abs(det(J))'] * model.elementQuadratureWeights[('u', 0)]
source = (f * dV).sum(axis=1)

vn = (ebq[('velocity', 0)] * ebq['n']).sum(axis=-1)            # (nE, nFaces, nqb)
face_flux = (vn * ebq[('dS_u', 0)]).sum(axis=-1)               # (nE, nFaces)
recomputed = source - face_flux.sum(axis=1)

rows = []
for eN in range(nOwned):
    xq = q['x'][eN, :, :2]
    order = np.lexsort((xq[:, 1], xq[:, 0]))
    mid = ebq['x'][eN, :, :, :2].mean(axis=1)
    forder = np.lexsort((mid[:, 1], mid[:, 0]))
    rows.append(np.concatenate([xq.mean(axis=0),
                                [q[('conservationResidual', 0)][eN], recomputed[eN]],
                                face_flux[eN, forder],
                                q[('velocity', 0)][eN, order, :2].ravel()]))

mpi = comm.comm.tompi4py()
gathered = mpi.gather(np.array(rows), root=0)
if mpi.rank == 0:
    a = np.vstack(gathered)
    a = a[np.lexsort((a[:, 1], a[:, 0]))]
    np.savetxt("elements.csv", a, delimiter=",",
               header="cx,cy,stored_residual,recomputed_residual,face_fluxes...,velocity_at_quadrature...")
