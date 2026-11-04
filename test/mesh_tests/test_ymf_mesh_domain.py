"""MeshYmfDomain: a mesh a converter wrote, read without the in-tree format readers (C7.1 step E).

The fixtures in ``ymf_meshes/`` were written by ymf-import-sms (proteustoolkit/ymf-import-sms)
from proteus's own SMS fixtures: ``square4x4`` from ``test/poisson_2d/square4x4.3dm`` and
``.bc`` (NDS node tags), ``square4x4_tagged`` from the same mesh with EGS edge strings tagging
its four sides 1-4 (``square4x4_tagged.bc``), and ``cube_3x3x3`` from the 3DM file of the
importer tests. Each is compared with what the in-tree reader builds from the source file.
"""
import os
import warnings

import numpy as np
import pytest

from proteus import Comm, Domain, MeshTools

Comm.init()

HERE = os.path.dirname(os.path.abspath(__file__))
YMF = os.path.join(HERE, "ymf_meshes")
SQUARE_2DM = os.path.join(os.path.dirname(HERE), "poisson_2d", "square4x4")
#: The node tags square4x4.bc gives with NDS cards (1-based node, tag).
SQUARE_NODE_TAGS = [1, 2, 2, 1, 3, 3, 3, 2, 1]


def _in_tree_square():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        mesh = MeshTools.TriangularMesh()
        mesh.generateFrom2DMFile(SQUARE_2DM)
    mesh.nodeMaterialTypes[:] = SQUARE_NODE_TAGS
    return mesh


ARRAYS = ("nodeArray", "elementNodesArray", "elementMaterialTypes", "nodeMaterialTypes",
          "elementBoundaryNodesArray", "elementBoundaryElementsArray",
          "exteriorElementBoundariesArray", "edgeNodesArray", "elementBoundaryMaterialTypes")


def test_a_converted_2d_mesh_is_the_mesh_the_2dm_reader_builds():
    mesh = MeshTools.TriangularMesh()
    mesh.generateFromYmfFile(os.path.join(YMF, "square4x4"))
    in_tree = _in_tree_square()
    for key in ARRAYS:
        np.testing.assert_array_equal(getattr(mesh, key), getattr(in_tree, key), err_msg=key)


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


def test_a_ymf_mesh_domain_writes_and_reads_as_a_file_domain():
    from proteus import DomainYMF
    domain = Domain.MeshYmfDomain(os.path.join(YMF, "square4x4"))
    block = DomainYMF.to_ymf(domain)
    assert (block["kind"], block["format"], block["dimension"]) == ("file", "ymf", 2)
    back = DomainYMF.from_ymf(block)
    assert isinstance(back, Domain.MeshYmfDomain) and back.meshfile == domain.meshfile
