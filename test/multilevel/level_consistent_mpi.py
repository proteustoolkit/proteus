"""Helper run under mpiexec by test_level_consistent_partitioning.py; not collected.

Builds a multilevel mesh and reports, for the prolongation between consecutive
levels, how local it is:

* ``missing`` -- parents of a fine node this rank owns that are not in the
  coarse subdomain mesh on this rank (neither owned nor ghost). Zero is the
  property level-consistent partitioning exists for: every prolongation row
  needs only data the rank already holds.
* ``offproc`` -- the fraction of prolongation entries whose coarse column is
  owned by another rank, i.e. what PETSc has to communicate.
* ``imbalance`` -- largest owned-node count over the mean, per level.

Usage: level_consistent_mpi.py KIND LEVELS CONSISTENT [node|element [graph|slabs]]   (KIND tri, tet, quad or hex)

With CONSISTENT 0, ``graph`` (the default) cuts every level with the graph partitioner,
a draw that changes from run to run and can nest by luck; ``slabs`` pins each level to
slabs across an axis that turns with the level (pinned_partitions.py), which do not nest.

With element partitioning ``missing`` counts the fine elements this rank owns whose
parent element is not in the coarse subdomain, and ``offproc`` is not measured.
The element lookup the subdomain hierarchy uses is also checked against matching
elements by their nodes, which is independent of how the partitioner numbers them.
"""
import mpi_failfast  # noqa: F401  (an uncaught exception aborts every rank)
import sys

import numpy as np
from proteus import Comm
comm = Comm.init()
from mpi4py import MPI
from proteus import MeshTools
#this tests the global hierarchy builder, not hierarchies generated in pieces
MeshTools.MultilevelMesh.regularGrid = 'global'

rank, size = comm.rank(), comm.size()
mpi = comm.comm.tompi4py()
kind = sys.argv[1]
nLevels = int(sys.argv[2])
consistent = bool(int(sys.argv[3]))
by = sys.argv[4] if len(sys.argv) > 4 else 'node'
independentOwners = sys.argv[5] if len(sys.argv) > 5 else 'graph'
MeshTools.MultilevelMesh.levelConsistentPartitioning = consistent
if not consistent and independentOwners == 'slabs':
    from pinned_partitions import pin_independent_levels
    pin_independent_levels(2 if kind in ('tri', 'quad') else 3)
part = (MeshTools.MeshParallelPartitioningTypes.node if by == 'node'
        else MeshTools.MeshParallelPartitioningTypes.element)
if kind == 'tri':
    ml = MeshTools.MultilevelTriangularMesh(5, 5, 1, refinementLevels=nLevels,
                                            nLayersOfOverlap=1, parallelPartitioningType=part)
elif kind == 'tet':
    ml = MeshTools.MultilevelTetrahedralMesh(4, 4, 4, refinementLevels=nLevels,
                                             nLayersOfOverlap=1, parallelPartitioningType=part)
elif kind == 'quad':
    ml = MeshTools.MultilevelQuadrilateralMesh(5, 5, 1, refinementLevels=nLevels,
                                               nLayersOfOverlap=1, parallelPartitioningType=part)
else:
    ml = MeshTools.MultilevelHexahedralMesh(4, 4, 4, refinementLevels=nLevels,
                                            nLayersOfOverlap=1, parallelPartitioningType=part)
worst_missing = 0
off = 0
tot = 0
imbalance = 1.0
elementDisagreements = 0
for l in range(nLevels - 1):
    cm, fm = ml.meshList[l], ml.meshList[l + 1]
    if by == 'node':
        rowptr, colind, nzval = ml.partitionedVertexInterpolation(l)
        coff = np.asarray(cm.nodeOffsets_subdomain_owned)
        foff = np.asarray(fm.nodeOffsets_subdomain_owned)
        inCoarseSubdomain = set(np.asarray(cm.nodeNumbering_subdomain2global).tolist())
        for r in range(foff[rank], foff[rank + 1]):
            cols = colind[rowptr[r]:rowptr[r + 1]]
            worst_missing += sum(1 for c in cols if int(c) not in inCoarseSubdomain)
            o = int(((cols < coff[rank]) | (cols >= coff[rank + 1])).sum())
            off += o
            tot += cols.size
    else:
        parents = np.asarray(ml.elementParentsArrayList[l + 1])
        foldOf = np.asarray(fm.elementNumbering_global_old2new)
        coldOf = np.asarray(cm.elementNumbering_global_old2new)
        inCoarseSubdomain = set(np.asarray(cm.elementNumbering_subdomain2global).tolist())
        flo, fhi = (int(v) for v in np.asarray(fm.elementOffsets_subdomain_owned)[rank:rank + 2])
        fineNewToOld = np.empty_like(foldOf)
        fineNewToOld[foldOf] = np.arange(foldOf.size)
        for newE in range(flo, fhi):
            if int(coldOf[parents[fineNewToOld[newE]]]) not in inCoarseSubdomain:
                worst_missing += 1
    # the table lookup the subdomain hierarchy uses against matching through nodes
    for level in (l, l + 1):
        mesh = ml.meshList[level]
        table = ml.globalElementsOfSubdomain(level)
        saved = mesh.elementNumbering_global_old2new
        mesh.elementNumbering_global_old2new = None
        try:
            if getattr(mesh, 'nodeNumbering_global_old2new', None) is not None:
                elementDisagreements += int((ml.globalElementsOfSubdomain(level) != table).sum())
        finally:
            mesh.elementNumbering_global_old2new = saved
for mesh in ml.meshList:
    owned = np.diff(np.asarray(mesh.nodeOffsets_subdomain_owned))
    imbalance = max(imbalance, owned.max() / owned.mean())
missing = mpi.allreduce(worst_missing, op=MPI.SUM)
off = mpi.allreduce(off, op=MPI.SUM)
tot = mpi.allreduce(tot, op=MPI.SUM)
disagreements = mpi.allreduce(elementDisagreements, op=MPI.SUM)
if rank == 0:
    print("RESULT kind=%s levels=%d consistent=%d by=%s ranks=%d missing=%d offproc=%.4f imbalance=%.3f lookup_disagreements=%d"
          % (kind, nLevels, consistent, by, size, missing, off / max(tot, 1), imbalance,
             disagreements))
    sys.stdout.flush()
