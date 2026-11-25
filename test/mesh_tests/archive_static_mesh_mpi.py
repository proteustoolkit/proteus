"""Archive a mesh over several frames from however many ranks this runs on; the driver for test_archive_static_mesh.py.

Usage: archive_static_mesh_mpi.py PREFIX SYNC DRIVER MOTION FRAMES OUTDIR

PREFIX   a ymf mesh (``PREFIX.ymf``), triangles or tetrahedra
SYNC     global: one global grid (global_sync); per-rank: one grid per rank
DRIVER   the archive's driver: mpio, serial or per-rank
MOTION   static:   the same mesh every frame
         moving:   the nodes move between frames (geometry), the connectivity does not
         adapted:  from frame FRAMES//2 on, another mesh object with another connectivity (topology), as PUMI
                   adaptivity builds one
         forced:   static, but every frame is written with meshChanged=True (the writer's behaviour before
                   2026-11-24, the "before" of the size measurement)
FRAMES   number of frames (the first is init)
OUTDIR   where the archive ``mesh.h5`` goes

Each frame also writes a nodal field ``u`` (the x coordinate), so a frame's grid has a field to read back. Rank 0
prints ``nodes=<global node count of each frame, comma separated>``.
"""
import sys

import numpy as np

from proteus import Archiver, Comm, MeshTools

Comm.init()


def _partitioned(prefix, comm, refine=False):
    triangles = MeshTools.ymfMeshTopology(prefix) == "Triangle"
    mesh = MeshTools.TriangularMesh() if triangles else MeshTools.TetrahedralMesh()
    mesh.generateFromYmfFile(prefix)
    if refine:
        # another mesh of the same domain, with another connectivity: what an adapted mesh is to the archive
        fine = MeshTools.TriangularMesh() if triangles else MeshTools.TetrahedralMesh()
        lo, hi = mesh.nodeArray.min(axis=0), mesh.nodeArray.max(axis=0)
        if triangles:
            fine.generateTriangularMeshFromRectangularGrid(6, 6, hi[0] - lo[0], hi[1] - lo[1])
        else:
            fine.generateTetrahedralMeshFromRectangularGrid(4, 4, 4, hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])
        fine.nodeArray[:] += lo
        mesh = fine
    n = mesh.nNodes_global
    size = comm.size()
    nodeOwners = (np.arange(n) % size).astype(np.intc)        # scattered: the partitioner renumbers
    mesh.partitionMesh(nLayersOfOverlap=1, nodeOwners=nodeOwners)
    return mesh


def main(prefix, sync, driver, motion, frames, outdir):
    comm = Comm.get()
    frames = int(frames)
    mesh = _partitioned(prefix, comm)
    ar = Archiver.XdmfArchive(outdir, "mesh", global_sync=(sync == "global"), driver=driver)
    nodes = []
    for tCount in range(frames):
        if motion == "adapted" and tCount == frames // 2:
            mesh = _partitioned(prefix, comm, refine=True)
        sub = mesh.subdomainMesh
        if motion == "moving" and tCount > 0:
            sub.nodeArray[:, 1] += 0.01 * sub.nodeArray[:, 0]       # every rank moves its nodes, ghosts too
        init = tCount == 0
        sub.writeMeshXdmf(ar, "Spatial_Domain", t=0.1 * tCount, init=init,
                          meshChanged=(motion == "forced"), tCount=tCount)
        u = np.ascontiguousarray(sub.nodeArray[:, 0])
        if ar.global_sync:
            owned = mesh.nodeOffsets_subdomain_owned
            ar.write_field(sub.arGrid, "u", u, tCount, dimensions=[mesh.nNodes_global],
                           sync_offsets=owned, sync_data=u[:owned[comm.rank() + 1] - owned[comm.rank()]])
        else:
            ar.write_field(sub.arGrid, "u", u, tCount, dimensions=[sub.nNodes_global])
        nodes.append(mesh.nNodes_global)
        ar.sync()
    ar.close()
    if comm.rank() == 0:
        print("nodes=%s" % ",".join(str(n) for n in nodes))
        sys.stdout.flush()


if __name__ == "__main__":
    main(*sys.argv[1:7])
