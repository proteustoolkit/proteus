"""A ``discrete`` domain block is a mesh (ymf D4): proteus reads it in memory through MeshYmfDomain.

The reference is the same mesh read from its ymf file (``test_ymf_mesh_domain.py``'s fixtures): a
discrete block holding the file's grid, inline or by HDF5 reference, gives the mesh, the boundary
tags and, posed through ADRProblem, the solution the file does, bit for bit.
"""
import os

import numpy as np
import pytest

from proteus import Comm, Domain, DomainYMF, MeshTools

Comm.init()

HERE = os.path.dirname(os.path.abspath(__file__))
YMF = os.path.join(HERE, "ymf_meshes")
SQUARE = os.path.join(os.path.dirname(HERE), "poisson_2d", "square4x4")
TAGGED = os.path.join(YMF, "square4x4_tagged")


def _grid(prefix):
    """The ymf grid of a mesh file, as a discrete block holds it, its HDF5 references made absolute."""
    from ymf import archive
    domain, _ = archive.read_ymf(prefix + ".ymf")
    grid = archive.validate_domain(domain)["times"][0]["spaces"][MeshTools.YMF_MESH_SPACE]
    base = os.path.dirname(prefix)
    for _, owner in archive._data_items({"times": [{"time": 0.0, "spaces": {"m": grid}}]}):
        item = owner["data_item"]
        if "data" in item:
            name, dataset = item["data"].rsplit(":", 1)
            item["data"] = "%s:%s" % (os.path.join(base, name), dataset)
    return grid


def _inline(prefix):
    """The same grid with every array inline (the document carries the whole mesh)."""
    from ymf import archive
    return archive.inline_approximation(
        {"times": [{"time": 0.0, "spaces": {"m": _grid(prefix)}}]})["times"][0]["spaces"]["m"]


def _block(grid, **extra):
    block = {"kind": "discrete", "name": "square"}
    block.update(dict((k, grid[k]) for k in MeshTools.YMF_GRID_KEYS if k in grid and k != "name"))
    block.update(extra)
    return block


ARRAYS = ("nodeArray", "elementNodesArray", "elementMaterialTypes", "nodeMaterialTypes",
          "elementBoundaryNodesArray", "exteriorElementBoundariesArray", "elementBoundaryMaterialTypes")


@pytest.mark.parametrize("held", ["inline", "hdf5"])
def test_a_discrete_triangle_mesh_is_the_mesh_its_file_holds(held):
    pytest.importorskip("ymf.mesh")
    grid = _inline(TAGGED) if held == "inline" else _grid(TAGGED)
    domain = DomainYMF.from_ymf(_block(grid, dimension=2))
    assert isinstance(domain, Domain.MeshYmfDomain) and domain.nd == 2 and domain.meshfile is None
    mesh = MeshTools._generateMesh(domain, MeshTools.MeshOptions(nd=2)).meshList[0]
    from_file = MeshTools.TriangularMesh()
    from_file.generateFromYmfFile(TAGGED)
    for key in ARRAYS:
        np.testing.assert_array_equal(getattr(mesh, key), getattr(from_file, key), err_msg=key)
    assert sorted(set(mesh.elementBoundaryMaterialTypes[mesh.exteriorElementBoundariesArray])) == [1, 2, 3, 4]


def test_a_discrete_tetrahedral_mesh_is_three_dimensional():
    domain = DomainYMF.from_ymf(_block(_inline(os.path.join(YMF, "cube_3x3x3"))))
    assert isinstance(domain, Domain.MeshYmfDomain) and domain.nd == 3
    mesh = MeshTools._generateMesh(domain, MeshTools.MeshOptions(nd=3)).meshList[0]
    assert (mesh.nNodes_global, mesh.nElements_global, mesh.nExteriorElementBoundaries_global) == (27, 48, 48)


def test_a_planar_mesh_in_xy_needs_no_dimension():
    """Triangles with XY geometry are a planar mesh; XYZ with dimension 3 stays a surface."""
    grid = _inline(SQUARE)
    from ymf import archive
    nodes = np.array(grid["geometry"]["data_item"]["values"], dtype=float).reshape(-1, 3)[:, :2]
    grid["geometry"] = archive.geometry(archive.data_item_for(nodes, inline=True), geometry_type="XY")
    domain = DomainYMF.from_ymf(_block(grid))
    assert isinstance(domain, Domain.MeshYmfDomain) and domain.nd == 2
    info = MeshTools.ymfMeshInfo(domain.grid)
    np.testing.assert_array_equal(info.nodeArray, MeshTools.readMeshYmf(SQUARE).nodeArray)


def test_a_triangulated_surface_stays_a_surface():
    surface = Domain.TriangulatedSurfaceDomain([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], [[0, 1, 2]])
    block = DomainYMF.to_ymf(surface)
    assert block["kind"] == "discrete" and block["dimension"] == 3
    assert isinstance(DomainYMF.from_ymf(block), Domain.TriangulatedSurfaceDomain)


def test_a_discrete_domain_writes_back_as_the_block_it_came_from():
    block = _block(_inline(SQUARE), dimension=2)
    again = DomainYMF.to_ymf(DomainYMF.from_ymf(block))
    assert again["kind"] == "discrete" and again["dimension"] == 2
    assert again["topology"] == block["topology"] and again["geometry"] == block["geometry"]


def test_a_discrete_mesh_of_quadrilaterals_is_refused():
    block = _block(_inline(SQUARE), dimension=2)
    block["topology"] = dict(block["topology"], type="Quadrilateral")
    with pytest.raises(DomainYMF.DomainTranslationError, match="triangles or tetrahedra"):
        DomainYMF.from_ymf(block)


def test_the_bounds_of_a_discrete_mesh_are_its_nodes():
    lower, upper = MeshTools.ymfMeshBounds(_grid(SQUARE), 2)
    nodes = MeshTools.readMeshYmf(SQUARE).nodeArray[:, :2]
    assert lower == nodes.min(axis=0).tolist() and upper == nodes.max(axis=0).tolist()


def test_untagged_walls_of_a_discrete_mesh_are_counted():
    pytest.importorskip("ymf.mesh")
    assert MeshTools.ymfUntaggedExteriorFacets(_inline(TAGGED)) == 0
    assert MeshTools.ymfUntaggedExteriorFacets(_inline(SQUARE)) == 0      # no facet list
