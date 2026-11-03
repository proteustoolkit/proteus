#!/usr/bin/env python
"""Helper run under mpiexec by test_meshPartitionCharacterisation.py; not collected.

Partitions one of the stored unstructured meshes in meshPartition/ and writes
every array the partitioner produced on this rank to OUTDIR/rank<r>.npz, so
the test can check them against the serial mesh and against each other.

Usage: meshPartition_mpi.py MESH PATH OWNERS OVERLAP OUTDIR

MESH     tri  meshPartition/pslg.{node,ele,edge}      (Triangle, 154 nodes)
         tet  meshPartition/plc.{node,ele,face,edge}  (TetGen, 131 nodes)
PATH     serial   no partitioning: the global mesh as the reader builds it
         node     partitionNodes on the mesh read into memory
         element  partitionElements on the mesh read into memory
         files    partitionNodesFromTriangleFiles / partitionNodesFromTetgenFiles
                  (``parun -F``): the partitioner reads the files itself
OWNERS   scatter  prescribed owners, a deterministic hash of the index
         graph    none: MatPartitioning decides; run it with
                  -mat_partitioning_type current to get the default contiguous
                  chunks, which is what makes it reproducible
OVERLAP  layers of overlap (the files path supports only 1)

Prints ``RESULT ... calls=<partitioner>:<owners given> petsc=<0|1>``. calls is
which cpartitioning entry point ran and whether it was handed owners: the proof
that the case executed the branch it is named for.
"""
import os
import sys

import numpy as np
from proteus import Comm
comm = Comm.init()
from mpi4py import MPI
from proteus import MeshTools, FemTools, cmeshTools, cpartitioning

HERE = os.path.dirname(os.path.abspath(__file__))
rank, size = comm.rank(), comm.size()
mpi = comm.comm.tompi4py()
meshKind, path, ownersKind, overlap, outdir = sys.argv[1:6]
overlap = int(overlap)
filebase = os.path.join(HERE, 'meshPartition', {'tri': 'pslg', 'tet': 'plc'}[meshKind])
nd = {'tri': 2, 'tet': 3}[meshKind]

# Record which partitioner entry point ran and whether it was given owners.
calls = []
for name in ('partitionNodes', 'partitionElements',
             'partitionNodesFromTriangleFiles', 'partitionNodesFromTetgenFiles'):
    def wrap(f, name=name):
        def wrapped(*args, **kwargs):
            owners = args[4] if len(args) > 4 and name in ('partitionNodes', 'partitionElements') else None
            calls.append('%s:%d' % (name, owners is not None))
            return f(*args, **kwargs)
        return wrapped
    setattr(cpartitioning, name, wrap(getattr(cpartitioning, name)))


def scatterOwners(n):
    i = np.arange(n, dtype=np.int64)
    return (((i * 2654435761) >> 7) % size).astype('i')


#the order of cmeshTools.buildPythonMeshInterface
INTERFACE = ('nElements_global nNodes_global nNodes_element nNodes_elementBoundary '
             'nElementBoundaries_element nElementBoundaries_global '
             'nInteriorElementBoundaries_global nExteriorElementBoundaries_global '
             'max_nElements_node nEdges_global max_nNodeNeighbors_node elementNodesArray '
             'nodeElementsArray nodeElementOffsets elementNeighborsArray elementBoundariesArray '
             'elementBoundaryNodesArray elementBoundaryElementsArray '
             'elementBoundaryLocalElementBoundariesArray interiorElementBoundariesArray '
             'exteriorElementBoundariesArray edgeNodesArray nodeStarArray nodeStarOffsets '
             'elementMaterialTypes elementBoundaryMaterialTypes nodeMaterialTypes nodeArray').split()


def meshArrays(prefix, mesh):
    out = {}
    for name, value in zip(INTERFACE, cmeshTools.buildPythonMeshInterface(mesh.cmesh)):
        if value is not None:
            out[prefix + name] = np.array(value, copy=True)
    #what _markRankInterfaces left: flags with -1 on the faces cut by the partition
    out[prefix + 'elementBoundaryMaterialTypes'] = np.array(mesh.elementBoundaryMaterialTypes, copy=True)
    return out


MAPS = ('elementOffsets_subdomain_owned', 'elementNumbering_subdomain2global',
        'nodeOffsets_subdomain_owned', 'nodeNumbering_subdomain2global',
        'elementBoundaryOffsets_subdomain_owned', 'elementBoundaryNumbering_subdomain2global',
        'edgeOffsets_subdomain_owned', 'edgeNumbering_subdomain2global',
        'nodeNumbering_global_old2new', 'elementNumbering_global_old2new')

nodeType = MeshTools.MeshParallelPartitioningTypes.node
elementType = MeshTools.MeshParallelPartitioningTypes.element
Mesh = {'tri': MeshTools.TriangularMesh, 'tet': MeshTools.TetrahedralMesh}[meshKind]
arrays = {}
if path == 'files':
    assert ownersKind == 'graph', 'the files path has no prescribed owners'
    mesh = Mesh()
    Multilevel = {'tri': MeshTools.MultilevelTriangularMesh, 'tet': MeshTools.MultilevelTetrahedralMesh}[meshKind]
    mlMesh = Multilevel(0, 0, 0, skipInit=True, nLayersOfOverlap=overlap, parallelPartitioningType=nodeType)
    generate = getattr(mlMesh, {'tri': 'generatePartitionedMeshFromTriangleFiles',
                                'tet': 'generatePartitionedMeshFromTetgenFiles'}[meshKind])
    generate(filebase, 1, mesh, 1, nLayersOfOverlap=overlap, parallelPartitioningType=nodeType)
    mesh = mlMesh.meshList[0]
else:
    mesh = Mesh()
    if meshKind == 'tri':
        mesh.generateFromTriangleFiles(filebase=filebase, base=1)
    else:
        mesh.generateFromTetgenFiles(filebase, 1, parallel=size > 1)
    if path == 'serial':
        assert size == 1
        arrays.update(meshArrays('global.', mesh))
    else:
        n = mesh.nNodes_global if path == 'node' else mesh.nElements_global
        owners = {'scatter': lambda: scatterOwners(n), 'graph': lambda: None}[ownersKind]()
        mesh.partitionMesh(nLayersOfOverlap=overlap,
                           parallelPartitioningType=nodeType if path == 'node' else elementType,
                           nodeOwners=owners if path == 'node' else None,
                           elementOwners=owners if path == 'element' else None)

if path != 'serial':
    for name in MAPS:
        if getattr(mesh, name, None) is not None:
            arrays['map.' + name] = np.array(getattr(mesh, name), copy=True)
    sub = mesh.subdomainMesh
    arrays.update(meshArrays('sub.', sub))
    #the quadratic and DG numberings built on top of the partition
    q = FemTools.QuadraticLagrangeDOFMap(sub, FemTools.QuadraticOnSimplexWithNodalBasis(nd), nd)
    arrays['quadratic.l2g'] = np.array(q.l2g, copy=True)
    arrays['quadratic.subdomain2global'] = np.array(q.subdomain2global, copy=True)
    arrays['quadratic.dof_offsets_subdomain_owned'] = np.array(q.dof_offsets_subdomain_owned, copy=True)
    arrays['quadratic.counts'] = np.array([q.nDOF_all_processes, q.nDOF_subdomain, q.max_dof_neighbors])
    arrays['quadratic.lagrangeNodesArray'] = np.array(q.lagrangeNodesArray, copy=True)
    #a DG space needs every rank to own elements, and checks that collectively:
    #decide on every rank at once, or a rank that skips it deadlocks the rest
    if mpi.allreduce(sub.nElements_owned, op=MPI.MIN) > 0:
        dg =FemTools.DiscontinuousGalerkinDOFMap(sub, FemTools.LinearOnSimplexWithNodalBasis(nd))
        arrays['dg.l2g'] = np.array(dg.l2g, copy=True)
        arrays['dg.subdomain2global'] = np.array(dg.subdomain2global, copy=True)
        arrays['dg.dof_offsets_subdomain_owned'] = np.array(dg.dof_offsets_subdomain_owned, copy=True)
        arrays['dg.counts'] = np.array([dg.nDOF_all_processes, dg.nDOF_subdomain, dg.max_dof_neighbors])

os.makedirs(outdir, exist_ok=True)
np.savez(os.path.join(outdir, 'rank%d.npz' % rank), **arrays)
petsc = int(cpartitioning.petscInitialized())
petsc = mpi.allreduce(petsc, op=MPI.MAX)
allCalls = mpi.gather(','.join(calls), root=0)
if rank == 0:
    print("RESULT mesh=%s path=%s owners=%s overlap=%d ranks=%d calls=%s petsc=%d"
          % (meshKind, path, ownersKind, overlap, size, allCalls[0] or 'none', petsc))
    sys.stdout.flush()
