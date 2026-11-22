#!/usr/bin/env python
"""Helper run under mpiexec by test_minus_f_stored_mesh.py; not collected.

Meshes a stored tetgen PLC mesh through ``MeshTools.generateMesh`` as a deck does that reads the mesh it
finds (genMesh False), with ``parun -F`` (``generatePartitionedMeshFromFiles``) or without it, and
prints one ``RESULT`` line per rank.

Usage: minus_f_stored_mesh_mpi.py PREFIX OPTIONS F|memory

PREFIX   the stored mesh: PREFIX.poly (the domain), PREFIX.{node,ele,face,edge}, in the working directory
OPTIONS  ``none``: triangleOptions stays unset (what a deck that sets it only under ``if genMesh:`` leaves);
         otherwise the string to set as triangleOptions, whatever it says about how the files were made
F        parun -F: the partitioner reads the files itself
memory   no -F: rank 0 reads the whole mesh and it is partitioned in memory (the reference)

The RESULT line says whether the branch under test ran: ``req`` counts calls of
``_requireTetgenFilesToPartition`` and ``files`` of ``cpartitioning.partitionNodesFromTetgenFiles``.
Nothing is written here: every rank runs this module, and the files were staged by the test.
"""
import sys
import types

from proteus import Comm
comm = Comm.init()
from proteus import Domain, MeshTools, cpartitioning

prefix, options, mode = sys.argv[1:4]

calls = {"req": 0, "files": 0}


def count(owner, name, key):
    original = getattr(owner, name)

    def counted(*args, **kwargs):
        calls[key] += 1
        return original(*args, **kwargs)
    setattr(owner, name, counted)


count(MeshTools, "_requireTetgenFilesToPartition", "req")
count(cpartitioning, "partitionNodesFromTetgenFiles", "files")

domain = Domain.PiecewiseLinearComplexDomain(fileprefix=prefix)
meshOptions = MeshTools.getMeshOptions(domain)
meshOptions.genMesh = False
if options != "none":
    meshOptions.triangleOptions = options
physics = types.SimpleNamespace(domain=domain, genMesh=False)
numerics = types.SimpleNamespace()
try:
    mlMesh = MeshTools.generateMesh(physics, numerics, mode == "F")
except SystemExit as exit:
    print("RESULT rank=%d exit=%s" % (comm.rank(), exit.code), flush=True)
    raise
mesh = mlMesh.meshList[0]
nodes, elements = mesh.nodeOffsets_subdomain_owned, mesh.elementOffsets_subdomain_owned
rank = comm.rank()
print("RESULT rank=%d ownedNodes=%d ownedElements=%d globalNodes=%d globalElements=%d req=%d files=%d" % (
    rank, nodes[rank + 1] - nodes[rank], elements[rank + 1] - elements[rank], nodes[-1], elements[-1],
    calls["req"], calls["files"]), flush=True)
