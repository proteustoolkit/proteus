"""proteus.Domain <-> a YMF domain block.

Phase 5b of the YMF semantics campaign. A YMF domain says what *kind* of
region it is -- a box, a boundary representation, an already-discretized
mesh, or a reference to a file in another language -- and Proteus's
fourteen Domain classes fall into exactly those four.

What is translated is the region and how its boundary is named. Meshing
policy (everything on MeshTools.getMeshOptions(domain): he, use_gmsh,
use_plex, regionConstraints) and runtime state (shape_list,
auxiliaryVariables, BC) are not: the first is how a region becomes a mesh
rather than what the region is, and a spec states it under
discretizations[].mesh; the second is not data at all.

So the translation is lossy with respect to the *object* and lossless with
respect to the *domain*. Where a class carries region state the format
cannot hold, it refuses rather than quietly dropping it -- which is the
property these tests are mostly about.
"""

import pytest

from proteus import Domain

ymf_domain = pytest.importorskip("ymf.domain")
from proteus.DomainYMF import DomainTranslationError, from_ymf, to_ymf


def roundtrip(domain):
    """block -> Domain -> block, with the block validated on the way."""
    block = to_ymf(domain)
    ymf_domain.validate_domain_block(block)
    return block, to_ymf(from_ymf(block))


# --------------------------------------------------------------------------
# the four kinds
# --------------------------------------------------------------------------


def test_a_rectangular_domain_is_a_box():
    block, back = roundtrip(Domain.RectangularDomain(L=[2.0, 3.0], x=[0.0, 1.0]))
    assert block["kind"] == "box"
    assert block["lower"] == [0.0, 1.0] and block["upper"] == [2.0, 4.0]
    assert back == block


def test_a_pslg_is_a_boundary_representation():
    block, back = roundtrip(Domain.PlanarStraightLineGraphDomain(
        vertices=[[0., 0.], [1., 0.], [1., 1.], [0., 1.]],
        segments=[[0, 1], [1, 2], [2, 3], [3, 0]],
        vertexFlags=[1, 1, 2, 2], segmentFlags=[1, 2, 3, 4]))
    assert block["kind"] == "piecewise_linear" and block["dimension"] == 2
    assert block["segments"] == [[0, 1], [1, 2], [2, 3], [3, 0]]
    assert back == block


def test_a_piecewise_linear_complex_is_the_same_kind_in_3d():
    block, back = roundtrip(Domain.PiecewiseLinearComplexDomain(
        vertices=[[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
                  [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.]],
        facets=[[[0, 1, 2, 3]], [[4, 5, 6, 7]]], facetFlags=[1, 2]))
    assert block["kind"] == "piecewise_linear" and block["dimension"] == 3
    assert "facets" in block and "segments" not in block
    assert back == block


def test_a_triangulated_surface_is_already_discrete():
    """The one class XDMF could hold directly, so it uses that vocabulary."""
    block, back = roundtrip(Domain.TriangulatedSurfaceDomain(
        [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], [[0, 1, 2]]))
    assert block["kind"] == "discrete"
    assert block["topology"]["type"] == "Triangle"
    assert block["topology"]["number_of_elements"] == 1
    assert back == block


@pytest.mark.parametrize("cls,fmt", [
    (Domain.MeshHexDomain, "hex"),
    (Domain.MeshTetgenDomain, "tetgen"),
])
def test_a_mesh_file_domain_is_a_path_and_a_format(cls, fmt):
    """Half the hierarchy is already a path; the format says which."""
    block, back = roundtrip(cls("flume"))
    assert block["kind"] == "file"
    assert block["format"] == fmt and block["path"] == "flume"
    assert back == block


def test_boundary_tags_survive():
    """They are how a condition refers to a boundary, so they must."""
    block, back = roundtrip(Domain.RectangularDomain(L=[1.0, 1.0], x=[0.0, 0.0]))
    assert block["boundary_tags"]["left"] == 4      # 2-D RectangularDomain
    assert back["boundary_tags"] == block["boundary_tags"]


# --------------------------------------------------------------------------
# what it refuses, and why
# --------------------------------------------------------------------------


def test_a_domain_holding_a_live_object_refuses():
    with pytest.raises(DomainTranslationError, match="live object"):
        to_ymf(Domain.PUMIDomain())


def test_a_bathymetry_domain_refuses_rather_than_dropping_the_bathymetry():
    """Writing it as a plain PSLG would lose the point of the class."""
    with pytest.raises(DomainTranslationError, match="bathy"):
        to_ymf(Domain.InterpolatedBathymetryDomain(
            vertices=[[0., 0.], [1., 0.], [1., 1.]],
            segments=[[0, 1], [1, 2], [2, 0]]))


@pytest.mark.parametrize("dimension, cls", [(3, Domain.PiecewiseLinearComplexDomain),
                                            (2, Domain.PlanarStraightLineGraphDomain)])
def test_a_gmsh_geometry_is_a_use_gmsh_domain_that_round_trips(dimension, cls):
    """A gmsh_geo block builds the domain whose .geo file is the geometry and which gmsh meshes (C7.1
    step J); he is meshing policy, so it stays behind."""
    from proteus import MeshTools
    block = {"kind": "file", "format": "gmsh_geo", "path": "flume.geo", "dimension": dimension,
             "name": "flume", "units": "m"}
    domain = from_ymf(block)
    assert type(domain) is cls
    assert domain.geofile == "flume" and domain.vertices == []
    assert MeshTools.getMeshOptions(domain).use_gmsh is True
    assert MeshTools.meshFiles(domain, MeshTools.getMeshOptions(domain))["generator"] == "gmsh"
    assert to_ymf(domain) == block


def test_a_gmsh_3d_domain_is_deprecated_and_writes_the_same_block():
    with pytest.warns(DeprecationWarning, match="GMSH_3D_Domain is deprecated"):
        legacy = Domain.GMSH_3D_Domain(geofile="flume", he=0.1)
    block = to_ymf(legacy)
    assert block["kind"] == "file" and block["format"] == "gmsh_geo"
    assert block["path"] == "flume.geo"
    assert "he" not in block
    assert from_ymf(block).geofile == "flume"


def test_a_rescaled_gmsh_geometry_refuses_rather_than_dropping_the_mapping():
    """length_scale and permute_dims map the file into the domain."""
    with pytest.warns(DeprecationWarning):
        rescaled = Domain.GMSH_3D_Domain(geofile="flume", length_scale=0.001)
        permuted = Domain.GMSH_3D_Domain(geofile="flume", permute_dims=[0, 2, 1])
    with pytest.raises(DomainTranslationError, match="permutes its axes"):
        to_ymf(rescaled)
    with pytest.raises(DomainTranslationError, match="permutes its axes"):
        to_ymf(permuted)


def test_an_unknown_file_format_says_which_it_knows():
    with pytest.raises(DomainTranslationError, match="known: hex, tetgen, ymf"):
        from_ymf({"kind": "file", "format": "step", "path": "part.step"})


def test_a_discrete_domain_proteus_has_no_class_for_is_refused():
    """Triangles and tetrahedra are a MeshYmfDomain (test/mesh_tests/test_discrete_domain.py); a
    triangulated surface is a TriangulatedSurfaceDomain; other cells are refused."""
    with pytest.raises(DomainTranslationError, match="triangles or tetrahedra"):
        from_ymf({"kind": "discrete", "dimension": 3,
                  "topology": {"type": "Hexahedron", "values": []},
                  "geometry": {"type": "XYZ", "values": []}})


# --------------------------------------------------------------------------
# what is deliberately not carried
# --------------------------------------------------------------------------


def test_meshing_policy_is_not_part_of_the_domain():
    """MeshOptions is how the region becomes a mesh, not what it is."""
    domain = Domain.RectangularDomain(L=[1.0, 1.0], x=[0.0, 0.0])
    domain.MeshOptions.he = 0.05
    block = to_ymf(domain)
    assert not any("he" in str(k) or "mesh" in str(k).lower() for k in block)


def test_runtime_state_is_not_part_of_the_domain():
    domain = Domain.RectangularDomain(L=[1.0, 1.0], x=[0.0, 0.0])
    domain.auxiliaryVariables = [object()]
    domain.shape_list = [object()]
    block = to_ymf(domain)
    assert "auxiliaryVariables" not in block and "shape_list" not in block
