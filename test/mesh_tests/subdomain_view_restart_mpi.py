"""Write a per-rank archive of a partitioned ymf mesh, read its subdomain views back, compare; the driver for
test_subdomain_view_restart.py.

Usage: subdomain_view_restart_mpi.py PREFIX OWNERS DOCUMENT OUTDIR

OWNERS scatter (node i to rank i % size) or reversed (contiguous blocks, the last to rank 0); DOCUMENT
global (layout B) or per-rank (layout C). Each rank partitions with partitionMesh, writes the mesh to a
per-rank archive, then builds the mesh from the archive's views (MeshTools.subdomainViewMesh, by path)
and prints one line per rank: ``rank=<r> arrays=<n compared> differ=<names> maps=<...> scalars=<...>``.
"""
import sys

import numpy as np

from proteus import Archiver, Comm, MeshTools

Comm.init()


def owners(kind, n, size):
    return {"scatter": np.arange(n) % size,
            "reversed": size - 1 - np.arange(n) * size // n}[kind].astype(np.intc)


def main(prefix, kind, document, outdir):
    comm = Comm.get()
    mesh = MeshTools._meshFromYmf(prefix)
    mesh.partitionMesh(nLayersOfOverlap=1, nodeOwners=owners(kind, mesh.nNodes_global, comm.size()))
    ar = Archiver.XdmfArchive(outdir, "mesh", driver="per-rank", document=document)
    mesh.subdomainMesh.writeMeshXdmf(ar, "Spatial_Domain", init=True, meshChanged=True)
    ar.sync()
    ar.close()
    comm.barrier()
    built = MeshTools.subdomainViewMesh(outdir + "/mesh")
    sub, mine = mesh.subdomainMesh, built.subdomainMesh
    names = sorted(k for k, v in vars(sub).items() if isinstance(v, np.ndarray) and hasattr(mine, k))
    differ = [k for k in names if not np.array_equal(np.asarray(getattr(sub, k)), np.asarray(getattr(mine, k)))]
    scalars = sorted(k for k, v in vars(sub).items()
                     if isinstance(v, (int, float, np.integer, np.floating)) and not k.startswith("_"))
    differ += [k for k in scalars if getattr(mine, k, None) != getattr(sub, k)]
    maps = [k for k in MeshTools._PARTITION_MAPS if not np.array_equal(getattr(mesh, k), getattr(built, k))]
    globals_ = [k for k in MeshTools._VIEW_GLOBAL_SCALARS if getattr(built, k, None) != getattr(mesh, k, None)]
    marked = (sub.__dict__.get("rankInterfacesMarked"), mine.__dict__.get("rankInterfacesMarked"))
    cut = int(np.count_nonzero(np.asarray(mine.elementBoundaryMaterialTypes) == MeshTools.RANK_INTERFACE_FLAG))
    print("rank=%d arrays=%d scalars=%d differ=%s maps=%s globals=%s marked=%s cut=%d"
          % (comm.rank(), len(names), len(scalars), ",".join(differ) or "-", ",".join(maps) or "-",
             ",".join(globals_) or "-", "%s/%s" % marked, cut))
    sys.stdout.flush()


if __name__ == "__main__":
    main(*sys.argv[1:5])
