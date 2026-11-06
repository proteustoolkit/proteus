"""Independent level partitions that are known not to nest; imported by the MPI helpers, not collected.

The controls in test_level_consistent_partitioning.py show that a finer level which
does not inherit the coarser one's partition misses parents. Cut by the graph
partitioner, an independent level is whatever the partitioner gives that run: the cut
changes from run to run and from build to build, and so does what it shares with the
level above. One CI run found no missing parent at all on the 4-rank hex mesh, and
-mat_partitioning_type current (the initial, contiguous partition) gives exactly that.
Pinned partitions make the control independent of the partitioner and its options.

``pin_independent_levels()`` replaces the graph partitioner for every level with
slabs across one coordinate axis, the axis turning with the level: level 0 is cut
across x, level 1 across y, level 2 across x again (z joins in 3D). Each rank owns a
near-equal, contiguous run of nodes (or elements, by barycenter) in the order of that
coordinate, so no rank is empty. Two consecutive levels are cut across different
axes, so a fine node at the far end of its slab has parents in another rank's slab,
beyond any one layer of overlap: the partitions do not nest on any of the test meshes,
whatever the build.

``PROTEUS_TEST_INDEPENDENT_PARTITIONS=slabs`` selects them in the helpers that take
that switch; any other non-empty value keeps the graph partitioner.
"""
import os

import numpy as np
from proteus import MeshTools

__all__ = ['pin_independent_levels', 'slab_owners', 'requested']


def requested():
    """Whether the environment asks for pinned (slab) independent partitions."""
    return os.environ.get('PROTEUS_TEST_INDEPENDENT_PARTITIONS', '') == 'slabs'


def slab_owners(points, axis, size):
    """The owner of each point: ``size`` near-equal runs in the order of coordinate ``axis``.

    Ties are broken by index (a stable sort), so the owners depend only on the points.
    """
    order = np.argsort(np.asarray(points)[:, axis], kind='stable')
    owners = np.empty(len(order), dtype=np.intc)
    starts = (np.arange(size + 1) * len(order)) // size
    for r in range(size):
        owners[order[starts[r]:starts[r + 1]]] = r
    return owners


def pin_independent_levels(dim):
    """Partition every level of the next multilevel meshes independently, by turning slabs.

    Patches Mesh.partitionMesh so that a call without prescribed owners gets slab owners
    instead of the graph partitioner; the n-th such call (the n-th level) is cut across
    axis n mod dim. Turns level-consistent partitioning off. Returns the list the
    patched method appends each level's axis to.
    """
    from proteus import Comm
    MeshTools.MultilevelMesh.levelConsistentPartitioning = False
    original = MeshTools.Mesh.partitionMesh
    axes = []

    def partitionMesh(self, nLayersOfOverlap=1,
                      parallelPartitioningType=MeshTools.MeshParallelPartitioningTypes.node,
                      nodeOwners=None, elementOwners=None, nodeWeights=None):
        size = Comm.get().size()
        if size > 1 and nodeOwners is None and elementOwners is None:
            axis = len(axes) % dim
            axes.append(axis)
            nodes = np.asarray(self.nodeArray)
            if parallelPartitioningType == MeshTools.MeshParallelPartitioningTypes.node:
                nodeOwners = slab_owners(nodes, axis, size)
            else:
                elementNodes = np.asarray(self.elementNodesArray)
                elementOwners = slab_owners(nodes[elementNodes].mean(axis=1), axis, size)
            nodeWeights = None
        return original(self, nLayersOfOverlap=nLayersOfOverlap,
                        parallelPartitioningType=parallelPartitioningType,
                        nodeOwners=nodeOwners, elementOwners=elementOwners,
                        nodeWeights=nodeWeights)

    MeshTools.Mesh.partitionMesh = partitionMesh
    return axes
