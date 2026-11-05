"""Archive a ymf mesh from however many ranks this runs on; the driver for test_archive_input_order.py.

Usage: archive_input_order_mpi.py PREFIX OWNERS SYNC OUTDIR

PREFIX   a ymf mesh (``PREFIX.ymf``), triangles or tetrahedra
OWNERS   contiguous: node i to rank i * size // n (the partitioner keeps the input order)
         scatter:    node i to rank i % size (the partitioner renumbers the nodes)
         reversed:   contiguous, last block to rank 0 (renumbers nodes and elements)
SYNC     global: one global grid (global_sync); per-rank: one grid per rank
OUTDIR   where the archive ``mesh.h5`` goes

Rank 0 prints ``fingerprint=<the global mesh's inputFingerprint()>`` and ``renumbered=<0|1>``.
"""
import os
import sys

import numpy as np

from proteus import Archiver, Comm, MeshTools

Comm.init()


def main(prefix, owners, sync, outdir):
    comm = Comm.get()
    size, rank = comm.size(), comm.rank()
    triangles = MeshTools.ymfMeshTopology(prefix) == "Triangle"
    mesh = MeshTools.TriangularMesh() if triangles else MeshTools.TetrahedralMesh()
    mesh.generateFromYmfFile(prefix)
    n = mesh.nNodes_global
    nodeOwners = {"scatter": np.arange(n) % size,
                  "contiguous": np.arange(n) * size // n,
                  "reversed": size - 1 - np.arange(n) * size // n}[owners].astype(np.intc)
    mesh.partitionMesh(nLayersOfOverlap=1, nodeOwners=nodeOwners)
    ar = Archiver.XdmfArchive(outdir, "mesh", global_sync=(sync == "global"))
    mesh.subdomainMesh.writeMeshXdmf(ar, "Spatial_Domain", init=True, meshChanged=True)
    ar.sync()
    ar.close()
    try:
        fingerprint = mesh.inputFingerprint()
    except ImportError:                       # a ymf before cekees/ymf#11
        fingerprint = "none"
    renumbered = int(any(m is not None for m in MeshTools.inputNumbering(mesh)))
    if rank == 0:
        print("fingerprint=%s renumbered=%d" % (fingerprint, renumbered))
        sys.stdout.flush()


if __name__ == "__main__":
    main(*sys.argv[1:5])
