#!/usr/bin/env python
"""The velocity PostProcessingTools produces, converted to the H(div) spaces of FemTools; run by
test_hdiv_vpp.py, serially and under mpiexec.

Runs test/post_processing/vpp_partition_mpi.py unchanged (the heterogeneous Poisson problem vpp_poisson_2d,
variant and space from VPP_VARIANT, VPP_ORDER, VPP_SPACE, VPP_OVERLAP, partition pinned), then takes the
post-processor of the solved model and

* samples its element-wise velocity (``evaluateElementVelocityField``) at the dof points of
  ``Hdiv_AffineOnSimplex`` (RT0, or BDM1 for the -bdm variants) on the elements this rank owns and turns the
  samples into dofs (``dofsFromElementValues``). Each face shared by two owned elements gets a number from
  each: their difference is how far the post-processed velocity is from being single valued in its normal
  component;
* evaluates the FemTools field at random points of every owned element and compares it with the
  post-processor's own evaluation there;
* compares the outward face fluxes and |K| div v with the fluxes the model integrates on its faces and with
  the element's source integral;
* gathers the dofs of all ranks into one global vector (ranks must agree where they overlap, every global dof
  must be covered) and writes it with the points that identify each dof, for the serial-vs-parallel
  comparison.

Rank 0 prints one line ``RESULT {json}``.
"""
import json
import os
import runpy

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
g = runpy.run_path(os.path.join(HERE, "post_processing", "vpp_partition_mpi.py"))

from proteus import FemTools
from mpi4py import MPI

comm, model, vpp, mesh, nOwned = g["comm"], g["model"], g["vpp"], g["mesh"], g["nOwned"]
mpi = comm.comm.tompi4py()
variant = os.environ.get("VPP_VARIANT", "pwl")
family = "bdm1" if "bdm" in variant else "rt0"
nd = 2

fn = vpp.velocityFunction(0)            # the post-processor's own H(div) field (C10b), no conversion here
space, dof, known, defect = fn.femSpace, fn.dof, fn.known, fn.normalDefect
dm = space.dofMap
owned = np.arange(nOwned)
samples = vpp.evaluateElementVelocityField(np.ascontiguousarray(dm.elementDofPoints), 0)

# where the two elements of a face disagree: split the faces by whether one of their elements has a node on
# the boundary of the domain
lfs = space.referenceFiniteElement.localFunctionSpace
local = dm.sign * np.einsum("eid,eid->ei", dm.elementOutwardNormals[:, lfs.space.dof_face, :], samples)
lo, hi = np.full(dm.nDOF, np.inf), np.full(dm.nDOF, -np.inf)
np.minimum.at(lo, dm.l2g[owned], local[owned])
np.maximum.at(hi, dm.l2g[owned], local[owned])
jump = np.where(known, hi - lo, 0.0)
on_boundary = np.zeros(mesh.nNodes_global, dtype=bool)
on_boundary[np.asarray(mesh.elementBoundaryNodesArray)[np.asarray(mesh.exteriorElementBoundariesArray)].ravel()] = True
touches = on_boundary[np.asarray(mesh.elementNodesArray)].any(axis=1)
dof_touches = np.zeros(dm.nDOF, dtype=bool)
dof_touches[dm.l2g[touches]] = True
jump_away = float(jump[~dof_touches].max()) if (~dof_touches).any() else 0.0
dofs_with_jump = int((jump > 1.0e-10).sum())

# the same field: values at random points of the owned elements
xi = np.random.RandomState(7).dirichlet(np.ones(nd + 1), size=5)[:, 1:]
X, en = np.asarray(mesh.nodeArray), np.asarray(mesh.elementNodesArray)
xq = np.ascontiguousarray(np.einsum("qv,evc->eqc", np.column_stack([1.0 - xi.sum(axis=1), xi]), X[en]))
theirs = vpp.evaluateElementVelocityField(xq, 0)[:nOwned]
mine = space.evaluate(dof, xi)[:nOwned]
field_difference = float(np.abs(mine - theirs).max()) if nOwned else 0.0
field_scale = float(np.abs(theirs).max()) if nOwned else 0.0

# fluxes and divergence against what the model integrates
face_flux = g["face_flux"][:nOwned]                       # (nOwned, 3), local face = opposite local vertex
flux_difference = float(np.abs(space.elementFluxes(dof)[:nOwned] - face_flux).max()) if nOwned else 0.0
volume = np.abs(np.linalg.det(space.elementJacobians())) / 2.0
balance = float(np.abs(volume[:nOwned] * space.divergence(dof)[:nOwned] - g["source"][:nOwned]).max()) if nOwned else 0.0
residual = float(np.abs(g["recomputed"][:nOwned]).max()) if nOwned else 0.0

# one global vector
s2g = np.asarray(dm.subdomain2global)
keys = np.hstack([dm.dofPoints, dm.dofFaceCentroids])
parts = mpi.gather((s2g[known], dof[known], keys[known]), root=0)
result = dict(variant=variant, family=family, ranks=comm.size(), order=int(os.environ.get("VPP_ORDER", "1")),
              nDOF=int(dm.nDOF_all_processes),
              normal_defect=float(mpi.allreduce(defect, op=MPI.MAX)),
              normal_jump_away_from_boundary=float(mpi.allreduce(jump_away, op=MPI.MAX)),
              dofs_with_jump=int(mpi.allreduce(dofs_with_jump)),
              field_difference=float(mpi.allreduce(field_difference, op=MPI.MAX)),
              field_scale=float(mpi.allreduce(field_scale, op=MPI.MAX)),
              flux_difference=float(mpi.allreduce(flux_difference, op=MPI.MAX)),
              balance=float(mpi.allreduce(balance, op=MPI.MAX)),
              residual=float(mpi.allreduce(residual, op=MPI.MAX)),
              nElements=int(mpi.allreduce(int(nOwned))),
              converged_reason=int(g["converged_reason"]))
if mpi.rank == 0:
    n = int(dm.nDOF_all_processes)
    value = np.full(n, np.nan)
    point = np.full((n, 6), np.nan)
    cross = 0.0
    for idx, val, key in parts:
        seen = ~np.isnan(value[idx])
        if seen.any():
            cross = max(cross, float(np.abs(value[idx[seen]] - val[seen]).max()))
        value[idx] = val
        point[idx] = key
    result["cross_rank_defect"] = cross
    result["dofs_covered"] = int((~np.isnan(value)).sum())
    np.savez("hdiv_dofs.npz", value=value, point=point)
    print("RESULT " + json.dumps(result), flush=True)
