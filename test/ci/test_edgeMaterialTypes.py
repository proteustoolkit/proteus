#!/usr/bin/env python
"""Every mesh carries edge flags (Mesh.edgeMaterialTypes), one per edge: the
.edge file's marker when a Triangle or TetGen .edge file described the edges
(-1 read as 0), and 0 ("no tag") otherwise, the rule node flags follow
(decisions 1 and 12 of 2026-11-06). In process, no MPI launch: the readers
and the generators that build their own edges."""
import os
import shutil

import numpy as np
import pytest

from proteus import MeshTools

HERE = os.path.dirname(os.path.abspath(__file__))
STORED = os.path.join(HERE, 'meshPartition')


def _zeros(mesh):
    flags = mesh.edgeMaterialTypes
    assert flags is not None
    assert len(flags) == mesh.nEdges_global == len(mesh.edgeNodesArray) > 0
    assert not np.asarray(flags).any()


@pytest.mark.parametrize('kind', ['tri', 'tet', 'quad', 'hex', 'edge'])
def test_generated_meshes_have_zero_edge_flags(kind):
    """Rectangular-grid generators build their edges themselves: every flag 0."""
    if kind == 'tri':
        m = MeshTools.TriangularMesh()
        m.generateTriangularMeshFromRectangularGrid(4, 3, 1.0, 1.0)
    elif kind == 'tet':
        m = MeshTools.TetrahedralMesh()
        m.generateTetrahedralMeshFromRectangularGrid(3, 3, 3, 1.0, 1.0, 1.0)
    elif kind == 'quad':
        m = MeshTools.QuadrilateralMesh()
        m.generateQuadrilateralMeshFromRectangularGrid(4, 3, 1.0, 1.0)
    elif kind == 'hex':
        m = MeshTools.HexahedralMesh()
        m.generateHexahedralMeshFromRectangularGrid(3, 3, 3, 1.0, 1.0, 1.0)
    else:
        m = MeshTools.EdgeMesh()
        m.generateEdgeMeshFromRectangularGrid(5, 1.0)
    _zeros(m)


def test_mesh_from_arrays_has_zero_edge_flags():
    nodes = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])
    elements = np.array([[0, 1, 2], [1, 3, 2]], dtype='i')
    m = MeshTools.TriangularMesh()
    m.generateFromArrays(nodes, elements)
    _zeros(m)


@pytest.mark.parametrize('mesh', ['tri', 'tet'])
def test_files_without_an_edge_file_have_zero_edge_flags(tmp_path, mesh):
    """Triangle/TetGen files read without a .edge file (triangle/tetgen run
    without -e): the edges come from the elements, every flag 0."""
    stored = os.path.join(STORED, {'tri': 'pslg', 'tet': 'plc'}[mesh])
    base = str(tmp_path / 'noedge')
    for ext in ('node', 'ele') + (('face',) if mesh == 'tet' else ()):
        shutil.copy(stored + '.' + ext, base + '.' + ext)
    m = MeshTools.TriangularMesh() if mesh == 'tri' else MeshTools.TetrahedralMesh()
    if mesh == 'tri':
        m.generateFromTriangleFiles(base, 1)
    else:
        m.generateFromTetgenFiles(base, 1)
    _zeros(m)


@pytest.mark.parametrize('mesh', ['tri', 'tet'])
def test_files_with_a_marked_edge_file_have_its_markers(tmp_path, mesh):
    """The in-memory readers take each edge's marker from the .edge file (here
    k % 6 on edge k, and -1 on every seventh of TetGen's, read as 0)."""
    stored = os.path.join(STORED, {'tri': 'pslg', 'tet': 'plc'}[mesh])
    base = str(tmp_path / 'marked')
    for ext in ('node', 'ele') + (('face',) if mesh == 'tet' else ()):
        shutil.copy(stored + '.' + ext, base + '.' + ext)
    with open(stored + '.edge') as f:
        lines = [l.split() for l in f.read().splitlines() if l.strip() and not l.lstrip().startswith('#')]
    out, expected = ['%s 1' % lines[0][0]], {}
    for k, l in enumerate(lines[1:]):
        marker = -1 if (mesh == 'tet' and k % 7 == 0) else k % 6
        out.append(' '.join(l[:3] + [str(marker)]))
        expected[tuple(sorted((int(l[1]) - 1, int(l[2]) - 1)))] = 0 if marker == -1 else marker
    with open(base + '.edge', 'w') as f:
        f.write('\n'.join(out) + '\n')
    m = MeshTools.TriangularMesh() if mesh == 'tri' else MeshTools.TetrahedralMesh()
    if mesh == 'tri':
        m.generateFromTriangleFiles(base, 1)
    else:
        m.generateFromTetgenFiles(base, 1)
    got = {tuple(sorted(e)): f for e, f in zip(np.asarray(m.edgeNodesArray).tolist(),
                                               np.asarray(m.edgeMaterialTypes).tolist())}
    assert got == expected
