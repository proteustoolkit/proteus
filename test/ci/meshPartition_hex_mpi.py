#!/usr/bin/env python
"""Helper run under mpiexec by test_meshPartitionCharacterisation.py; not collected.

Partitions a generated hexahedral grid (the stored meshes are simplices) and
reports, for the subdomain mesh on every rank, how many rows of
``edgeNodesArray`` are not an edge of the mesh: an edge of a hexahedron joins
two nodes that differ in exactly one grid index, so a row is bad when its two
nodes' coordinates differ in more than one axis or not at all, i.e. is (0, 0)
or a face or body diagonal. It also reports a digest of the edges the rank
holds, by coordinates (so it does not depend on the numbering).

Usage: meshPartition_hex_mpi.py PATH OVERLAP NX
  PATH     element | node   (partitionElements | partitionNodes, ``-mat_partitioning_type current``)

Prints ``RESULT ranks=<n> nEdges=<per rank> bad=<per rank> digest=<edge set per rank> all=<digest of the union>``.
"""
import hashlib
import sys

import numpy as np
from proteus import Comm
comm = Comm.init()
from proteus import MeshTools

path, overlap, nx = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
mpi = comm.comm.tompi4py()
mesh = MeshTools.HexahedralMesh()
mesh.generateHexahedralMeshFromRectangularGrid(nx, nx, nx, 1.0, 1.0, 1.0)
if comm.size() > 1:
    mesh.partitionMesh(nLayersOfOverlap=overlap,
                       parallelPartitioningType=(MeshTools.MeshParallelPartitioningTypes.element
                                                 if path == 'element' else
                                                 MeshTools.MeshParallelPartitioningTypes.node))
    sub = mesh.subdomainMesh
else:
    sub = mesh
x = np.asarray(sub.nodeArray)
edges = np.asarray(sub.edgeNodesArray).reshape(-1, 2)[:sub.nEdges_global]
h = 1.0 / nx
bad = 0
keys = []
for a, b in edges:
    differ = np.count_nonzero(np.abs(x[a] - x[b]) > 0.5 * h)
    if differ != 1:
        bad += 1
    ka = tuple(np.round(x[a] / h).astype(int))
    kb = tuple(np.round(x[b] / h).astype(int))
    keys.append((min(ka, kb), max(ka, kb)))
keys = sorted(set(keys))
digest = hashlib.sha1(repr(keys).encode()).hexdigest()[:10]
everything = mpi.gather((sub.nEdges_global, bad, digest, keys), root=0)
if comm.rank() == 0:
    union = sorted(set(k for _, _, _, ks in everything for k in ks))
    print('RESULT ranks=%d nEdges=%s bad=%s digest=%s all=%s'
          % (comm.size(), ','.join(str(e[0]) for e in everything), ','.join(str(e[1]) for e in everything),
             ','.join(e[2] for e in everything), hashlib.sha1(repr(union).encode()).hexdigest()[:10]))
