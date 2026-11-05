"""Write each rank's ymf subdomain view of a partitioned ymf mesh; the driver for test_subdomain_view.py.

Usage: subdomain_view_mpi.py PREFIX OWNERS OUTDIR

OWNERS is scatter (node i to rank i % size) or reversed (contiguous blocks, the last to rank 0), as in
archive_input_order_mpi.py. Each rank partitions the mesh with those owners through partitionMesh
(the MPI path), writes ``rank<r>.json`` (its subdomainYmfGrid) and ``rank<r>.npz`` (its subdomain
mesh's arrays, with the -1 the partitioner marked).
"""
import json
import os
import sys

import numpy as np

from proteus import Comm, MeshTools

Comm.init()

ARRAYS = ("nodeArray", "elementNodesArray", "elementBoundaryNodesArray", "exteriorElementBoundariesArray",
          "elementBoundaryMaterialTypes", "nodeMaterialTypes", "elementMaterialTypes")


def owners(kind, n, size):
    return {"scatter": np.arange(n) % size,
            "reversed": size - 1 - np.arange(n) * size // n}[kind].astype(np.intc)


def main(prefix, kind, outdir):
    comm = Comm.get()
    size, rank = comm.size(), comm.rank()
    mesh = MeshTools._meshFromYmf(prefix)
    mesh.partitionMesh(nLayersOfOverlap=1, nodeOwners=owners(kind, mesh.nNodes_global, size))
    grid = MeshTools.subdomainYmfGrid(mesh)
    with open(os.path.join(outdir, "rank%d.json" % rank), "w") as f:
        json.dump(grid, f)
    sub = mesh.subdomainMesh
    np.savez(os.path.join(outdir, "rank%d.npz" % rank),
             marked=bool(sub.__dict__.get("rankInterfacesMarked", False)),
             nNodes_owned=sub.nNodes_owned, nElements_owned=sub.nElements_owned,
             **dict((name, np.asarray(getattr(sub, name))) for name in ARRAYS))


if __name__ == "__main__":
    main(*sys.argv[1:4])
