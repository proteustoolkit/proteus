"""A partition whose cut leaves a corner's strong-Dirichlet face on one rank and its other face on another.

Used by the strong_dirichlet_*_mpi.py helpers. For each corner given, the element with a boundary face whose
flag is in ``carrier_flags`` (the face that makes the corner a Dirichlet dof) goes to rank 0 and the element
with the corner's other boundary face goes to rank 1; the rest is dealt out in contiguous blocks to the ranks
except 1. Rank 1 then holds the corner without the Dirichlet condition. Corners that have only one element, or
whose two faces are both carriers or both not, are left alone (the mesh decides how many are trapped).
"""
import numpy as np

from proteus import Comm, MeshTools


def trap_owners(mesh, nranks, carrier_flags, corners):
    others = [r for r in range(nranks) if r != 1] or [0]
    owners = np.array([others[min(len(others) - 1, e * len(others) // mesh.nElements_global)]
                       for e in range(mesh.nElements_global)], dtype=np.intc)
    exterior = set(mesh.exteriorElementBoundariesArray.tolist())
    trapped = 0
    for corner in corners:
        i = np.where(np.abs(mesh.nodeArray[:, :2] - corner).max(1) < 1e-9)[0]
        if len(i) == 0:
            continue
        i = int(i[0])
        carrier, other = None, None
        for e in range(mesh.nElements_global):
            if i not in mesh.elementNodesArray[e]:
                continue
            for f in mesh.elementBoundariesArray[e]:
                if f in exterior and i in mesh.elementBoundaryNodesArray[f]:
                    if mesh.elementBoundaryMaterialTypes[f] in carrier_flags:
                        carrier = e
                    else:
                        other = e
        if carrier is not None and other is not None and carrier != other:
            owners[carrier] = 0
            owners[other] = 1
            trapped += 1
    return owners, trapped


def install(carrier_flags, corners, report=None):
    """Replace Mesh.partitionMesh so that, with more than one rank, the element owners are the trap's."""
    original = MeshTools.Mesh.partitionMesh

    def partitionMesh(self, nLayersOfOverlap=1, parallelPartitioningType=MeshTools.MeshParallelPartitioningTypes.node,
                      nodeOwners=None, elementOwners=None, nodeWeights=None):
        if elementOwners is None and Comm.get().size() > 1:
            elementOwners, trapped = trap_owners(self, Comm.get().size(), carrier_flags, corners)
            if report is not None:
                report(trapped)
        return original(self, nLayersOfOverlap=nLayersOfOverlap, parallelPartitioningType=parallelPartitioningType,
                        nodeOwners=nodeOwners, elementOwners=elementOwners, nodeWeights=nodeWeights)
    MeshTools.Mesh.partitionMesh = partitionMesh
