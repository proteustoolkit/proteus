"""MeshYmfDomain: a mesh a converter wrote, read as ymf (C7.1 step E).

proteus reads explicit meshes only as ymf; SMS/ADH files are converted by ymf-import-sms
(proteustoolkit/ymf-import-sms). The fixtures in ``ymf_meshes/`` were written by it from
proteus's own SMS fixtures: ``square4x4`` from ``test/poisson_2d/square4x4.3dm`` and ``.bc``
(NDS node tags), ``square4x4_tagged`` from the same mesh with EGS edge strings tagging its four
sides 1-4 (``square4x4_tagged.bc``), and ``cube_3x3x3`` from the 3DM file of the importer
tests. Before the in-tree SMS readers were removed, each read was bit-equal to the in-tree
reader's (the commit that added this file); the reference now is the source file's text.
"""
import os

import numpy as np
import pytest

from proteus import Comm, Domain, MeshTools

Comm.init()

HERE = os.path.dirname(os.path.abspath(__file__))
YMF = os.path.join(HERE, "ymf_meshes")
SQUARE_2DM = os.path.join(os.path.dirname(HERE), "poisson_2d", "square4x4")
#: The node tags square4x4.bc gives with NDS cards (1-based node, tag).
SQUARE_NODE_TAGS = [1, 2, 2, 1, 3, 3, 3, 2, 1]


def _sms_text(filebase):
    """Nodes, zero-based triangles and (material - 1) straight from an SMS file's E3T/ND cards."""
    nodes, cells, materials = [], [], []
    for words in (line.split() for line in open(filebase + ".3dm")):
        if words and words[0] == "E3T":
            cells.append([int(w) - 1 for w in words[2:5]])
            materials.append(int(words[5]) - 1)
        elif words and words[0] == "ND":
            nodes.append([float(w) for w in words[2:5]])
    return np.array(nodes), np.array(cells), np.array(materials)


def test_a_converted_2d_mesh_is_the_source_file():
    mesh = MeshTools.TriangularMesh()
    mesh.generateFromYmfFile(os.path.join(YMF, "square4x4"))
    nodes, cells, materials = _sms_text(SQUARE_2DM)
    np.testing.assert_array_equal(mesh.nodeArray, nodes)
    np.testing.assert_array_equal(mesh.elementNodesArray, cells)
    np.testing.assert_array_equal(mesh.elementMaterialTypes, materials)
    assert mesh.nodeMaterialTypes.tolist() == SQUARE_NODE_TAGS
    assert (mesh.nElementBoundaries_global, mesh.nExteriorElementBoundaries_global) == (16, 8)
    # untagged: the construction's tags, 1 outside and 0 inside
    assert set(mesh.elementBoundaryMaterialTypes[mesh.exteriorElementBoundariesArray]) == {1}
    assert not mesh.elementBoundaryMaterialTypes[mesh.interiorElementBoundariesArray].any()


def test_a_converted_3d_mesh_is_the_mesh_the_3dm_reader_builds():
    mesh = MeshTools.TetrahedralMesh()
    mesh.generateFromYmfFile(os.path.join(YMF, "cube_3x3x3"))
    source = MeshTools.TetrahedralMesh()
    source.generateTetrahedralMeshFromRectangularGrid(3, 3, 3, 1.0, 1.0, 1.0)
    np.testing.assert_array_equal(mesh.nodeArray, source.nodeArray)
    np.testing.assert_array_equal(mesh.elementNodesArray, source.elementNodesArray)
    np.testing.assert_array_equal(mesh.elementMaterialTypes, np.arange(48) % 3 + 1)
    assert (mesh.nElementBoundaries_global, mesh.nExteriorElementBoundaries_global,
            mesh.nEdges_global) == (120, 48, 98)


def test_a_ymf_mesh_holds_the_topology_its_reader_expects():
    with pytest.raises(ValueError, match="holds a Tetrahedron mesh, not triangles"):
        MeshTools.TriangularMesh().generateFromYmfFile(os.path.join(YMF, "cube_3x3x3"))


def test_facet_tags_become_element_boundary_tags():
    pytest.importorskip("ymf.mesh")
    mesh = MeshTools.TriangularMesh()
    mesh.generateFromYmfFile(os.path.join(YMF, "square4x4_tagged"))
    tags = mesh.elementBoundaryMaterialTypes
    exterior = mesh.exteriorElementBoundariesArray
    # each side its own tag, from the edge strings; interior boundaries untouched
    sides = {}
    for ebN in exterior:
        mid = mesh.nodeArray[mesh.elementBoundaryNodesArray[ebN]].mean(axis=0)
        lo, hi = mesh.nodeArray.min(axis=0), mesh.nodeArray.max(axis=0)
        side = (1 if np.isclose(mid[1], lo[1]) else 2 if np.isclose(mid[0], hi[0]) else
                3 if np.isclose(mid[1], hi[1]) else 4)
        sides.setdefault(side, set()).add(int(tags[ebN]))
    assert sides == {1: {1}, 2: {2}, 3: {3}, 4: {4}}
    assert not tags[mesh.interiorElementBoundariesArray].any()
    # the other arrays are those of the untagged file
    untagged = MeshTools.TriangularMesh()
    untagged.generateFromYmfFile(os.path.join(YMF, "square4x4"))
    np.testing.assert_array_equal(mesh.elementBoundaryNodesArray, untagged.elementBoundaryNodesArray)


def test_a_facet_that_is_not_a_boundary_is_refused(tmp_path):
    pytest.importorskip("ymf.mesh")
    import shutil
    import h5py
    for suffix in (".ymf", ".h5"):
        shutil.copy(os.path.join(YMF, "square4x4_tagged" + suffix), tmp_path)
    with h5py.File(tmp_path / "square4x4_tagged.h5", "r+") as f:
        facets = f["facets"][()]
        facets[0] = [0, 8]        # opposite corners: no element has this edge
        f["facets"][...] = facets
    with pytest.raises(ValueError, match="is not a face"):
        MeshTools.TriangularMesh().generateFromYmfFile(str(tmp_path / "square4x4_tagged"))


@pytest.mark.parametrize("name, nd, counts", [("square4x4", 2, (9, 8)), ("cube_3x3x3", 3, (27, 48))])
def test_a_ymf_mesh_domain_is_meshed_and_partitioned(name, nd, counts):
    """Through _generateMesh, on however many ranks the test runs on; nd comes from the file."""
    domain = Domain.MeshYmfDomain(os.path.join(YMF, name))
    assert domain.nd == nd
    mlMesh = MeshTools._generateMesh(domain, MeshTools.MeshOptions(nd=nd))
    mesh = mlMesh.meshList[0]
    assert (mesh.nNodes_global, mesh.nElements_global) == counts
    rank = Comm.get().rank()
    owned = mesh.elementOffsets_subdomain_owned[rank + 1] - mesh.elementOffsets_subdomain_owned[rank]
    assert Comm.get().comm.tompi4py().allreduce(int(owned)) == counts[1]


def test_an_old_style_meshfile_is_a_ymf_mesh():
    """A p module's meshfile (setDomainFromPhysics) names a ymf mesh; an SMS file is converted first."""
    domain = MeshTools._meshFileDomain(os.path.join(YMF, "square4x4"), 2)
    assert isinstance(domain, Domain.MeshYmfDomain) and domain.nd == 2
    with pytest.raises(ValueError, match="names no ymf mesh .* ymf-import-sms"):
        MeshTools._meshFileDomain(SQUARE_2DM, 2)


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


def test_array_mesh_rejects_bad_connectivity():
    from proteus import cmeshTools
    with pytest.raises(ValueError, match="outside"):
        cmeshTools.generateFromArrays(cmeshTools.CMesh(), np.zeros((3, 3)), [[0, 1, 3]])


def test_a_ymf_mesh_domain_writes_and_reads_as_a_file_domain():
    from proteus import DomainYMF
    domain = Domain.MeshYmfDomain(os.path.join(YMF, "square4x4"))
    block = DomainYMF.to_ymf(domain)
    assert (block["kind"], block["format"], block["dimension"]) == ("file", "ymf", 2)
    back = DomainYMF.from_ymf(block)
    assert isinstance(back, Domain.MeshYmfDomain) and back.meshfile == domain.meshfile
