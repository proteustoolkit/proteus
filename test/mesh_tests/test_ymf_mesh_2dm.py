"""A 2DM mesh is converted to a ymf mesh, and the ymf mesh is what a run reads.

The reference is the C++ 2DM reader, the path this replaces: a mesh read
through ymf must equal the one it reads, array for array.
"""
import os
import numpy as np
import pytest
from proteus import Domain, MeshTools

MESHDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'poisson_2d')
SQUARE = os.path.join(MESHDIR, 'square4x4')

ARRAYS = ['nodeArray', 'elementNodesArray', 'nodeMaterialTypes',
          'elementMaterialTypes', 'elementBoundariesArray',
          'elementNeighborsArray', 'elementBoundaryElementsArray',
          'elementBoundaryNodesArray', 'nodeElementsArray', 'nodeElementOffsets']


def test_converted_mesh_equals_2dm_mesh(tmp_path):
    prefix = str(tmp_path / 'square')
    MeshTools.convert2DMToYmf(SQUARE, prefix)
    reference = MeshTools.TriangularMesh()
    reference.generateFrom2DMFile(SQUARE)
    mesh = MeshTools.TriangularMesh()
    mesh.generateFromYmfFile(prefix)
    assert mesh.nElements_global == reference.nElements_global == 8
    assert mesh.nNodes_global == reference.nNodes_global == 9
    for name in ARRAYS:
        assert np.array_equal(getattr(mesh, name), getattr(reference, name)), name
    # the node tags came from square4x4.bc
    assert list(mesh.nodeMaterialTypes) == [1, 2, 2, 1, 3, 3, 3, 2, 1]


def test_ymf_mesh_is_read_without_the_2dm_file(tmp_path):
    prefix = str(tmp_path / 'square')
    MeshTools.convert2DMToYmf(SQUARE, prefix)
    info = MeshTools.readMeshYmf(prefix)
    assert info.elementTopologyName == 'Triangle'
    assert info.nodeArray.shape == (9, 3)
    assert info.elementNodesArray.shape == (8, 3)
    assert sorted(os.listdir(tmp_path)) == ['square.h5', 'square.ymf']


def test_write_read_round_trip(tmp_path):
    nodes = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])
    elements = np.array([[0, 1, 2], [1, 3, 2]])
    prefix = str(tmp_path / 'two')
    MeshTools.writeMeshYmf(prefix, nodes, elements,
                           nodeMaterialTypes=[3, 0, 0, 4], elementMaterialTypes=[7, 8])
    info = MeshTools.readMeshYmf(prefix)
    assert np.array_equal(info.nodeArray, nodes)
    assert np.array_equal(info.elementNodesArray, elements)
    assert list(info.nodeMaterialTypes) == [3, 0, 0, 4]
    assert list(info.elementMaterialTypes) == [7, 8]


def test_a_mesh_domain_is_read_through_ymf(tmp_path, monkeypatch):
    """The Mesh2DMDomain branch converts and reads the ymf mesh, not the 2DM file."""
    monkeypatch.chdir(tmp_path)
    reads = []
    original = MeshTools.TriangularMesh.generateFromYmfFile

    def spy(self, prefix):
        reads.append(prefix)
        return original(self, prefix)

    def forbidden(self, *args, **kwargs):
        raise AssertionError("the 2DM reader was called")

    monkeypatch.setattr(MeshTools.TriangularMesh, 'generateFromYmfFile', spy)
    monkeypatch.setattr(MeshTools.TriangularMesh, 'generateFrom2DMFile', forbidden)
    domain = Domain.Mesh2DMDomain(SQUARE)
    mlMesh = MeshTools._generateMesh(domain, MeshTools.getMeshOptions(domain))
    assert reads == ['square4x4_2dm']
    assert mlMesh.meshList[0].nElements_global == 8
    assert os.path.exists('square4x4_2dm.ymf') and os.path.exists('square4x4_2dm.h5')


def test_element_boundary_tags_are_refused(tmp_path):
    base = str(tmp_path / 'tri')
    with open(base + '.3dm', 'w') as f:
        f.write("MESH2D\nE3T 1 1 2 3 1\nND 1 0. 0. 0.\nND 2 1. 0. 0.\nND 3 0. 1. 0.\n")
    with open(base + '.bc', 'w') as f:
        f.write("FCS 1 1 5\n")
    with pytest.raises(NotImplementedError, match="FCS"):
        MeshTools.convert2DMToYmf(base)


def test_non_triangle_elements_are_refused(tmp_path):
    base = str(tmp_path / 'quad')
    with open(base + '.3dm', 'w') as f:
        f.write("MESH2D\nE4Q 1 1 2 3 4 1\nND 1 0. 0. 0.\nND 2 1. 0. 0.\nND 3 1. 1. 0.\nND 4 0. 1. 0.\n")
    with pytest.raises(NotImplementedError, match="E4Q"):
        MeshTools.convert2DMToYmf(base)


def test_array_mesh_rejects_bad_connectivity():
    from proteus import cmeshTools
    with pytest.raises(ValueError, match="outside"):
        cmeshTools.generateFromArrays(cmeshTools.CMesh(), np.zeros((3, 3)), [[0, 1, 3]])
