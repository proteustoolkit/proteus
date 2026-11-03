#!/usr/bin/env python
"""Solve vpp_poisson_2d on one or more ranks and write what the velocity post-processing produced.

Run by test_velocity_postprocessing.py in an empty directory, serially or under mpiexec. The variant comes from
VPP_VARIANT, the space from VPP_ORDER (1 or 2). On more than one rank the node partition is pinned (two ranks: the cut x + 0.37 y >= 0.62;
more: parallel stripes in x + 0.37 y), so serial and parallel runs are comparable element by element (VPP_FREE_PARTITION=1 leaves it to
the partitioner). VPP_NO_STAR_EXCHANGE=1 and VPP_NO_CUT_FACE_EXCHANGE=1 turn off one of the two parallel exchanges
of the PWL variants, as controls.

Writes elements.csv on rank 0: one row per owned element, sorted by centroid, with
  cx, cy, the residual VPP stores (conservationResidual), the residual recomputed here (source integral minus the
  outward flux of the post-processed velocity through the element's faces), the three face fluxes ordered by face
  midpoint, and the velocity at the element quadrature points ordered by position.
And summary.json on rank 0: the linear solver's converged reason (negative: it stopped without converging, and then
no variant can match serial), the L2 error of the velocity at the element quadrature points and the largest face-flux
error, both against the exact -a grad u (smooth problem only), and the largest flux through a face of the low-K block
(VPP_PROBLEM=lowk). Discontinuous spaces (VPP_SPACE=dg or ldg) are partitioned by element, pinned the same way.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from proteus import Comm, MeshTools as MT, runtime

runtime.start()
comm = Comm.get()

_partition = MT.Mesh.partitionMesh


def _owners(x, size):
    t = x[:, 0] + 0.37 * x[:, 1]
    if size == 2:
        return (t >= 0.62).astype(np.intc)
    return np.minimum(size - 1, np.floor(size * t / 1.37)).astype(np.intc)  # parallel stripes across the square


def partitionMesh(self, nLayersOfOverlap=1, parallelPartitioningType=MT.MeshParallelPartitioningTypes.node,
                  nodeOwners=None, elementOwners=None, nodeWeights=None):
    size = Comm.get().size()
    if size > 1 and not os.environ.get("VPP_FREE_PARTITION"):
        if parallelPartitioningType == MT.MeshParallelPartitioningTypes.node and nodeOwners is None:
            nodeOwners = _owners(self.nodeArray, size)
        elif parallelPartitioningType != MT.MeshParallelPartitioningTypes.node and elementOwners is None:
            elementOwners = _owners(self.nodeArray[self.elementNodesArray].mean(axis=1), size)
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
from mpi4py import MPI

opts_db = PETSc.Options()
if os.environ.get("VPP_SPACE", "c0") != "c0":
    # CG/ASM stops at the iteration limit on the interior-penalty systems (2 and 4 pinned ranks); solve them directly
    opts_db.setValue("ksp_type", "preonly")
    opts_db.setValue("pc_type", "lu")
    if comm.size() > 1:
        opts_db.setValue("pc_factor_mat_solver_type", "mumps")
else:
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
ksp = ns.modelList[0].solver.solverList[-1].linearSolver.ksp
converged_reason = ksp.getConvergedReason()
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
summary = {"converged_reason": int(converged_reason)}
if p.analyticalSolutionVelocity is not None:
    def exact(x):
        return -np.stack([(x[..., 0] + 1.0) * 2.0 * x[..., 0], (x[..., 1] + 2.0) * 2.0 * x[..., 1]], axis=-1)
    dv = q[('velocity', 0)][:nOwned, :, :2] - exact(q['x'][:nOwned])
    summary["l2_velocity_error"] = float(np.sqrt(mpi.allreduce(((dv ** 2).sum(axis=-1) * dV[:nOwned]).sum())))
    exact_flux = ((exact(ebq['x'][:nOwned]) * ebq['n'][:nOwned, :, :, :2]).sum(axis=-1)
                  * ebq[('dS_u', 0)][:nOwned]).sum(axis=-1)
    summary["max_face_flux_error"] = float(mpi.allreduce(np.abs(face_flux[:nOwned] - exact_flux).max(), op=MPI.MAX))
if p.problem == "lowk":
    mid = ebq['x'][:nOwned, :, :, :2].mean(axis=2)                     # (nOwned, nFaces, 2) face midpoints
    on_block = (np.abs(mid - 1.0 / 3).min(axis=-1) < 1e-8) | (np.abs(mid - 2.0 / 3).min(axis=-1) < 1e-8)
    on_block &= ((mid > 1.0 / 3 - 1e-8) & (mid < 2.0 / 3 + 1e-8)).all(axis=-1)
    block_face = np.abs(face_flux[:nOwned])[on_block]
    summary["max_block_face_flux"] = float(mpi.allreduce(block_face.max() if block_face.size else 0.0, op=MPI.MAX))
    summary["n_block_faces"] = int(mpi.allreduce(int(on_block.sum())))
gathered = mpi.gather(np.array(rows), root=0)
if mpi.rank == 0:
    a = np.vstack(gathered)
    a = a[np.lexsort((a[:, 1], a[:, 0]))]
    np.savetxt("elements.csv", a, delimiter=",",
               header="cx,cy,stored_residual,recomputed_residual,face_fluxes...,velocity_at_quadrature...")
    with open("summary.json", "w") as f:
        json.dump(summary, f)
