#!/usr/bin/env python
"""Partitioning an empty global mesh says the mesh is empty.

Seen 2026-11-03 as "every Triangle/PSLG mesh fails in cpartitioning.partitionNodes
with an IndexError" in an environment run without activation: no ``triangle`` on
PATH, so MeshTools.writeMeshFiles' call failed unnoticed, the reader reported
"cannot open file" and returned a mesh with no nodes, and partitionNodes failed
on ``&owners[0]`` of an empty owner array.
"""
import numpy as np
import pytest
from proteus import Comm, MeshTools, cmeshTools, cpartitioning

comm = Comm.init()


@pytest.mark.parametrize('partition', ['partitionNodes', 'partitionElements'])
@pytest.mark.parametrize('owners', [None, 'prescribed'])
def test_empty_global_mesh_is_a_clear_error(partition, owners):
    owners = np.zeros(0, dtype=np.intc) if owners else None
    with pytest.raises(ValueError, match='global mesh is empty'):
        getattr(cpartitioning, partition)(comm.comm.tompi4py(), 1, cmeshTools.CMesh(), cmeshTools.CMesh(), owners)


def test_mesh_from_missing_triangle_files(tmp_path):
    """The path that was hit: the files were never written. Since cekees/proteus#181 the reader says so
    itself, naming the file, instead of returning an empty mesh for partitionNodes to fail on."""
    mesh = MeshTools.TriangularMesh()
    with pytest.raises(OSError, match='never_written.node'):
        mesh.generateFromTriangleFiles(filebase=str(tmp_path / 'never_written'), base=1)
