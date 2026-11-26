"""C49 R4: the parts of proteus.SpatialTools that no deck or test builds.

The geometry helpers (rotation2D/3D, extrude2Dto3D, getInfoFromSTL, getGmshPhysicalGroups), the shapes that the
existing tests never construct (Sphere, Cylinder, Circle, ShapeSTL), the Shape methods that move or relate shapes,
and the geometry assembled into a Domain when shapes share vertices or contain each other.  Every test compares with
a value computed independently (a rotation by hand, the closed-surface identities of a mesh, a volume by the
divergence theorem), not only that the call returns.
"""
import numpy as np
import numpy.testing as npt
import pytest

from proteus import Comm
from proteus.Domain import (PiecewiseLinearComplexDomain,
                            PlanarStraightLineGraphDomain)
from proteus import SpatialTools as st
from proteus.SpatialTools import (Circle, Cuboid, Cylinder, Rectangle, Shape,
                                  ShapeSTL, Sphere, CustomShape,
                                  assembleDomain, extrude2Dto3D,
                                  getGmshPhysicalGroups, getInfoFromSTL,
                                  rotation2D, rotation3D, unique_rows)

Comm.init()


def _volume_by_divergence(vertices, triangles):
    """volume of a surface that is star-shaped about the origin: the sum of the pyramids on its triangles (the
    orientation of Sphere's triangles is not consistent, so each pyramid is taken as a positive volume)"""
    v = np.asarray(vertices)
    t = np.asarray(triangles).reshape(-1, 3)
    return np.abs(np.einsum('ij,ij->i', v[t[:, 0]], np.cross(v[t[:, 1]], v[t[:, 2]]))).sum() / 6.


def _edge_use(triangles):
    """number of triangles that use each undirected edge"""
    use = {}
    for tri in np.asarray(triangles).reshape(-1, 3):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            e = tuple(sorted((int(tri[a]), int(tri[b]))))
            use[e] = use.get(e, 0) + 1
    return use


# ---------------------------------------------------------------------------------------------------------------------
# rotations
# ---------------------------------------------------------------------------------------------------------------------

def test_rotation2D_quarter_turn_about_origin_and_a_pivot():
    pts = np.array([[1., 0.], [0., 2.]])
    # the convention is the row-vector one of the module: a positive angle turns counter-clockwise
    r = rotation2D(pts, np.pi / 2.)
    # derive the sign from a point on the axis, then check the rest against it
    sign = np.sign(r[0, 1])
    assert abs(abs(r[0, 1]) - 1.) < 1e-14 and abs(r[0, 0]) < 1e-14
    npt.assert_allclose(r, [[0., sign], [-2. * sign, 0.]], atol=1e-14)
    # a pivot is a fixed point; a single point (1-D array) takes the other branch of the function
    npt.assert_allclose(rotation2D([3., 4.], 1.3, pivot=(3., 4.)), [3., 4.], atol=1e-14)
    single = rotation2D([2., 1.], np.pi, pivot=(1., 1.))
    npt.assert_allclose(single, [0., 1.], atol=1e-14)
    # rotations preserve distance to the pivot
    big = np.random.RandomState(0).rand(7, 2)
    rb = rotation2D(big, 0.7, pivot=(0.3, -0.2))
    npt.assert_allclose(np.linalg.norm(rb - [0.3, -0.2], axis=1),
                        np.linalg.norm(big - [0.3, -0.2], axis=1), atol=1e-14)


@pytest.mark.parametrize('axis', [(0., 0., 1.), (1., 0., 0.), (0., 1., 0.), (1., 1., 1.)])
def test_rotation3D_is_a_rigid_rotation_about_the_axis(axis):
    rng = np.random.RandomState(1)
    pts = rng.rand(6, 3)
    pivot = np.array([0.2, -0.1, 0.4])
    rot = 0.9
    out = rotation3D(pts, rot, axis=axis, pivot=pivot)
    a = np.array(axis, dtype=float)
    a /= np.linalg.norm(a)
    # distances to the pivot and the component along the axis are invariant
    npt.assert_allclose(np.linalg.norm(out - pivot, axis=1), np.linalg.norm(pts - pivot, axis=1), atol=1e-13)
    npt.assert_allclose((out - pivot) @ a, (pts - pivot) @ a, atol=1e-13)
    # the angle between the perpendicular parts is `rot`
    p0 = (pts - pivot) - np.outer((pts - pivot) @ a, a)
    p1 = (out - pivot) - np.outer((out - pivot) @ a, a)
    cosang = np.einsum('ij,ij->i', p0, p1) / (np.linalg.norm(p0, axis=1) * np.linalg.norm(p1, axis=1))
    npt.assert_allclose(cosang, np.cos(rot), atol=1e-13)
    # a single point (1-D array) agrees with the same point in a 2-D array
    npt.assert_allclose(rotation3D(pts[0], rot, axis=axis, pivot=pivot), out[0], atol=1e-14)


# ---------------------------------------------------------------------------------------------------------------------
# Shape methods
# ---------------------------------------------------------------------------------------------------------------------

def test_shape_setPosition_translate_rotate_move_every_attribute_in_3D():
    d = PiecewiseLinearComplexDomain()
    c = Cuboid(d, dim=(1., 2., 3.), coords=(1., 1., 1.))
    c.setHoles([[1., 1., 1.]])
    assert c.holes.shape == (1, 3)
    v0, r0 = c.vertices.copy(), c.regions.copy()
    c.setPosition(np.array([2., 3., 4.]))
    npt.assert_allclose(c.barycenter, [2., 3., 4.])
    npt.assert_allclose(c.vertices, v0 + [1., 2., 3.])
    npt.assert_allclose(c.regions, r0 + [1., 2., 3.])
    npt.assert_allclose(c.coords, [2., 3., 4.])
    npt.assert_allclose(c.holes, [[2., 3., 4.]])
    npt.assert_allclose(c.getPosition(), [2., 3., 4.])
    # rotation about z through the barycenter: the barycenter, region and hole (all on the pivot) stay put
    c.rotate(np.pi / 2., axis=(0., 0., 1.))
    npt.assert_allclose(c.barycenter, [2., 3., 4.], atol=1e-13)
    npt.assert_allclose(c.holes, [[2., 3., 4.]], atol=1e-13)
    npt.assert_allclose(c.regions, [[2., 3., 4.]], atol=1e-13)
    npt.assert_allclose(c.coords, [2., 3., 4.], atol=1e-13)
    # the local coordinate system is orthonormal and the face normals turned with it
    cs = c.getRotation()
    npt.assert_allclose(cs @ cs.T, np.eye(3), atol=1e-13)
    npt.assert_allclose(np.linalg.norm(c.b_or, axis=1), 1., atol=1e-13)
    # box size is preserved
    extent = np.ptp(c.vertices, axis=0)
    npt.assert_allclose(sorted(extent), [1., 2., 3.], atol=1e-13)
    # rotating about a pivot off the barycenter moves the barycenter on a circle around it
    pivot = np.array([0., 0., 0.])
    before = np.linalg.norm(c.barycenter[:2] - pivot[:2])
    c.rotate(0.4, axis=(0., 0., 1.), pivot=pivot)
    npt.assert_allclose(np.linalg.norm(c.barycenter[:2] - pivot[:2]), before, atol=1e-13)
    # setBarycenter changes the barycenter and nothing else
    v1 = c.vertices.copy()
    c.setBarycenter([9., 8., 7.])
    npt.assert_allclose(c.barycenter, [9., 8., 7.])
    npt.assert_allclose(c.vertices, v1)


def test_shape_methods_in_2D_with_holes_regions_and_barycenter_of_two_entries():
    d = PlanarStraightLineGraphDomain()
    r = Rectangle(d, dim=(2., 1.), coords=(1., 1.))
    r.setHoles([[1., 1.]])
    r.setBarycenter([1., 1.])           # two entries: only the in-plane part is set
    npt.assert_allclose(r.barycenter, [1., 1., 0.])
    r.setPosition(np.array([4., 5.]))
    npt.assert_allclose(r.barycenter, [4., 5., 0.])
    npt.assert_allclose(r.holes, [[4., 5.]])
    npt.assert_allclose(r.coords, [4., 5.])
    r.rotate(np.pi / 2.)
    # a 2 x 1 rectangle turned by 90 degrees is 1 x 2
    npt.assert_allclose(sorted(np.ptp(r.vertices, axis=0)), [1., 2.], atol=1e-13)
    npt.assert_allclose(r.barycenter, [4., 5., 0.], atol=1e-13)
    npt.assert_allclose(r.holes, [[4., 5.]], atol=1e-13)
    cs = r.getRotation()
    npt.assert_allclose(cs @ cs.T, np.eye(2), atol=1e-13)


def test_setRegions_checks_its_flags():
    d = PlanarStraightLineGraphDomain()
    r = Rectangle(d, dim=(1., 1.), coords=(0., 0.))
    r.setRegions([[0., 0.], [5., 5.]], [1, 2])
    npt.assert_allclose(r.regions, [[0., 0.], [5., 5.]])
    npt.assert_array_equal(r.regionFlags, [1, 2])
    with pytest.raises(AssertionError):
        r.setRegions([[0., 0.]], [2])           # tags start at 1
    with pytest.raises(AssertionError):
        r.setRegions([[0., 0.], [1., 1.]], [1, 3])  # no gap
    with pytest.raises(AssertionError):
        r.setRegions([[0., 0.], [1., 1.]], [1])  # lengths differ


@pytest.mark.xfail(strict=True, raises=(UnboundLocalError, NameError, TypeError),
                   reason='Shape.setRegions: `regionFLags = self.regionFlags` is a misspelling of regionFlags, so '
                          'setRegions(regions) without flags fails (found by C49 R4; kernel territory: C0)')
def test_setRegions_without_flags_keeps_the_existing_flags():
    d = PlanarStraightLineGraphDomain()
    r = Rectangle(d, dim=(1., 1.), coords=(0., 0.))
    r.setRegions([[0.3, 0.3]])
    npt.assert_array_equal(r.regionFlags, [1])


@pytest.mark.xfail(strict=True, raises=TypeError,
                   reason="Shape._checkListOfLists builds its message with 'str' + int, so a list of the wrong "
                          "dimension raises TypeError instead of its AssertionError (found by C49 R4; C0)")
def test_a_region_of_the_wrong_dimension_is_an_assertion_error():
    d = PlanarStraightLineGraphDomain()
    r = Rectangle(d, dim=(1., 1.), coords=(0., 0.))
    with pytest.raises(AssertionError):
        r.setRegions([[0., 0., 0.]], [1])


def test_flag_hack_99_and_tag_checks():
    d = PlanarStraightLineGraphDomain()
    r = Rectangle(d, dim=(1., 1.), coords=(0., 0.))
    r._checkFlags([1, 2, 3])
    r._checkFlags([1, 99])                      # the SWFlow hack ends the check
    with pytest.raises(AssertionError):
        r._checkFlags([2, 3])
    r._checkNd([0., 0.])
    r._checkNd([0., 0., 0.])
    with pytest.raises(AssertionError):
        r._checkNd([0.])


def test_child_and_parent_shapes_accumulate_per_index():
    d = PlanarStraightLineGraphDomain()
    outer = Rectangle(d, dim=(4., 4.), coords=(0., 0.))
    a = Rectangle(d, dim=(1., 1.), coords=(0., 0.))
    b = Rectangle(d, dim=(1., 1.), coords=(1., 1.))
    c = Rectangle(d, dim=(1., 1.), coords=(-1., -1.))
    outer.setChildShape(a)
    outer.setChildShape(b)         # second child of the same index appends
    c.setParentShape(outer, ind=1)
    d2 = Rectangle(d, dim=(1., 1.), coords=(-1., 1.))
    d2.setParentShape(outer, ind=1)  # second parent entry of an index appends too
    assert outer.children[0] == [a, b]
    assert outer.children[1] == [c, d2]


def test_shape_with_the_wrong_dimension_exits():
    d = PlanarStraightLineGraphDomain()
    with pytest.raises(SystemExit):
        Cuboid(d, dim=(1., 1., 1.), coords=(0., 0., 0.))   # a 3D shape in a 2D domain
    with pytest.raises(SystemExit):
        Shape(d, nd=3)


# ---------------------------------------------------------------------------------------------------------------------
# Sphere, Cylinder, Circle
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('nSectors', [2, 4])
def test_sphere_is_a_closed_surface_of_the_requested_radius(nSectors):
    d = PiecewiseLinearComplexDomain()
    radius = 0.7
    centre = np.array([1., -2., 3.])
    s = Sphere(d, radius, coords=centre, nSectors=nSectors)
    npt.assert_allclose(np.linalg.norm(s.vertices - centre, axis=1), radius, atol=1e-13)
    assert len(s.facets) == 16 * nSectors ** 2 == len(s.facetFlags)
    # closed 2-manifold: every edge is shared by exactly two triangles, Euler's formula V - E + F = 2
    use = _edge_use(s.facets)
    assert set(use.values()) == {2}
    assert len(s.vertices) - len(use) + len(s.facets) == 2
    # the cube-sphere mapping approximates the sphere: a polyhedron inscribed in it: volume 19% low at nSectors=2, 5% at 4 (see the convergence test)
    vol = _volume_by_divergence(s.vertices - centre, s.facets)
    exact = 4. / 3. * np.pi * radius ** 3
    assert abs(vol - exact) / exact < (0.2 if nSectors == 2 else 0.055)
    assert s.volumes == [[list(range(len(s.facets)))]]
    npt.assert_allclose(s.barycenter, centre)
    assert s.boundaryTags == {'sphere': 1}
    assert list(s.BC) == ['sphere']
    npt.assert_allclose(s.regions, [centre])
    # an explicit barycenter is kept
    s2 = Sphere(d, radius, coords=centre, barycenter=(0., 0., 0.), nSectors=2)
    npt.assert_allclose(s2.barycenter, 0.)


def test_sphere_volume_converges_with_nSectors():
    d = PiecewiseLinearComplexDomain()
    errs = []
    for n in (2, 4, 8):
        s = Sphere(d, 1., coords=(0., 0., 0.), nSectors=n)
        errs.append(abs(_volume_by_divergence(s.vertices, s.facets) - 4. / 3. * np.pi))
    # the inscribed polyhedron converges at second order: the error falls by about 4 per doubling
    assert errs[0] / errs[1] > 3. and errs[1] / errs[2] > 3.


def test_cylinder_geometry_and_tags():
    d = PiecewiseLinearComplexDomain()
    radius, height, n = 0.5, 2., 16
    cyl = Cylinder(d, radius, height, n, coords=(1., 2., 3.), barycenter=(1., 2., 3.))
    assert cyl.vertices.shape == (2 * n, 3)
    npt.assert_allclose(np.linalg.norm(cyl.vertices[:, :2] - [1., 2.], axis=1), radius, atol=1e-13)
    npt.assert_allclose(sorted(set(np.round(cyl.vertices[:, 2], 12))), [2., 4.])    # centred on coords
    assert len(cyl.facets) == 2 + n
    npt.assert_array_equal(cyl.facetFlags, [1, 2] + [3] * n)
    assert cyl.boundaryTags == {'z-': 1, 'z+': 2, 'cylinder': 3}
    assert len(cyl.segments) == 2 * n
    # the segments close each circle
    seg = cyl.segments
    for k in range(n):
        assert seg[k][1] == (seg[k][0] + 1) % n
        assert seg[n + k][1] == n + (seg[k][0] + 1) % n
    # the side facets are planar quads (all four z values form two levels)
    for f in cyl.facets[2:]:
        assert len(set(np.round(cyl.vertices[f[0], 2], 12))) == 2
    assert [b.name for b in cyl.BC_list] == ['z-', 'z+', 'cylinder']
    # area of the polygonal cap -> pi r^2 as n grows
    cap = cyl.vertices[:n, :2] - [1., 2.]
    area = 0.5 * abs(np.sum(cap[:, 0] * np.roll(cap[:, 1], -1) - np.roll(cap[:, 0], -1) * cap[:, 1]))
    assert abs(area - 0.5 * n * radius ** 2 * np.sin(2 * np.pi / n)) < 1e-14


@pytest.mark.xfail(strict=True, raises=ValueError,
                   reason='Cylinder(..., barycenter=None) stores np.array(None) (an object array) instead of '
                          'defaulting to coords, so the shape cannot be moved (found by C49 R4; kernel territory: C0)')
def test_cylinder_default_barycenter_is_its_coords():
    d = PiecewiseLinearComplexDomain()
    cyl = Cylinder(d, 0.5, 1., 8, coords=(1., 2., 3.))
    cyl.translate(np.array([1., 0., 0.]))
    npt.assert_allclose(cyl.barycenter, [2., 2., 3.])


def test_circle_default_and_tagged():
    d = PlanarStraightLineGraphDomain()
    c = Circle(d, radius=1.5, coords=(1., 2.), barycenter=None, nPoints=12)
    npt.assert_allclose(np.linalg.norm(c.vertices - [1., 2.], axis=1), 1.5, atol=1e-13)
    assert len(c.segments) == 12 and c.segments[-1].tolist() == [11, 0]
    assert c.boundaryTags == {'circle': 1}
    npt.assert_array_equal(c.vertexFlags, 1)
    npt.assert_array_equal(c.segmentFlags, 1)
    npt.assert_allclose(c.barycenter[:2], [1., 2.])
    npt.assert_allclose(c.It, np.pi * 1.5 ** 4 / 2.)
    assert c.facets.shape == (1, 1, 12)
    d2 = PlanarStraightLineGraphDomain()
    c2 = Circle(d2, radius=1., coords=(0., 0.), barycenter=(0.2, 0.1), nPoints=8, boundaryTag={'cyl': 3})
    assert c2.boundaryTags == {'cyl': 3}
    npt.assert_array_equal(c2.vertexFlags, 3)
    npt.assert_array_equal(c2.facetFlags, [3])
    npt.assert_array_equal(c2.regionFlags, [3])
    assert list(c2.BC) == ['cyl']
    npt.assert_allclose(c2.barycenter[:2], [0.2, 0.1])


# ---------------------------------------------------------------------------------------------------------------------
# STL, extrusion, gmsh groups
# ---------------------------------------------------------------------------------------------------------------------

_TETRA = """solid body
  facet normal 0 0 -1
    outer loop
      vertex 0 0 0
      vertex 0 1 0
      vertex 1 0 0
    endloop
  endfacet
  facet normal 0 -1 0
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0 0 1
    endloop
  endfacet
endsolid body
solid lid
  facet normal 1 1 1
    outer loop
      vertex 1 0 0
      vertex 0 1 0
      vertex 0 0 1
    endloop
  endfacet
  facet normal -1 0 0
    outer loop
      vertex 0 0 0
      vertex 0 0 1
      vertex 0 1 0
    endloop
  endfacet
endsolid lid
"""


def test_getInfoFromSTL_removes_duplicate_vertices_and_keeps_the_blocks(tmp_path):
    f = tmp_path / 'tetra.stl'
    f.write_text(_TETRA)
    vertices, facets, normals, vflags, fflags, tags = getInfoFromSTL(str(f))
    assert len(vertices) == 4                      # 12 corners -> the 4 vertices of the tetrahedron
    assert sorted(map(tuple, vertices)) == [(0., 0., 0.), (0., 0., 1.), (0., 1., 0.), (1., 0., 0.)]
    assert facets.shape == (4, 1, 3)
    # each facet still names the right coordinates after the renumbering
    expected = [{(0., 0., 0.), (0., 1., 0.), (1., 0., 0.)}, {(0., 0., 0.), (1., 0., 0.), (0., 0., 1.)},
                {(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)}, {(0., 0., 0.), (0., 0., 1.), (0., 1., 0.)}]
    for fac, exp in zip(facets, expected):
        assert {tuple(vertices[i]) for i in fac[0]} == exp
    npt.assert_allclose(normals, [[0, 0, -1], [0, -1, 0], [1, 1, 1], [-1, 0, 0]])
    npt.assert_array_equal(fflags, [1, 1, 2, 2])
    assert tags == {'body': 1, 'lid': 2}
    assert vflags.shape == (4,)
    # closed surface: each edge belongs to two facets
    assert set(_edge_use(facets).values()) == {2}


def test_ShapeSTL_builds_a_shape_with_one_BC_per_block(tmp_path):
    f = tmp_path / 'tetra.stl'
    f.write_text(_TETRA)
    d = PiecewiseLinearComplexDomain()
    s = ShapeSTL(d, str(f))
    assert s.filename == str(f)
    assert sorted(s.BC) == ['body', 'lid'] and len(s.BC_list) == 2
    assert s.volumes == [[[0, 1, 2, 3]]]
    assert s.boundaryTags == {'body': 1, 'lid': 2}
    assert s.name.startswith('STL')


def test_unique_rows_returns_first_occurrence_and_inverse():
    arr = np.array([[1., 2.], [0., 0.], [1., 2.], [3., 3.], [0., 0.]])
    uniq, inverse, indices = unique_rows(arr)
    assert len(uniq) == 3
    npt.assert_allclose(uniq[inverse], arr)
    npt.assert_allclose(arr[indices], uniq)


def test_extrude2Dto3D_counts_and_coordinates():
    vertices = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    # a 2D polygon is placed in 3D by the caller; extrusion adds the translated copy
    v3 = np.hstack((vertices, np.zeros((4, 1))))
    segments = [[0, 1], [1, 2], [2, 3], [3, 0]]
    facets = [[[0, 1, 2, 3]]]
    regions = [np.array([0.5, 0.5, 0.])]
    extrusion = np.array([0., 0., 2.])
    v, s, f, r = extrude2Dto3D(extrusion, v3, segments, facets, regions)
    assert len(v) == 8
    npt.assert_allclose(v[4:], v3 + extrusion)
    assert len(s) == 4 + 4 + 4
    assert s[4:8].tolist() == [[0, 4], [1, 5], [2, 6], [3, 7]]
    assert s[8:].tolist() == [[4, 5], [5, 6], [6, 7], [7, 4]]
    assert len(f) == 1 + 4 + 1
    assert f[-1] == [[4, 5, 6, 7]]
    assert f[4] == [[3, 0, 4, 7]]           # the closing side facet wraps around
    npt.assert_allclose(r[0], [0.5, 0.5, 1.])
    # the original region is not modified
    npt.assert_allclose(regions[0], [0.5, 0.5, 0.])
    # no regions given
    _, _, _, none = extrude2Dto3D(extrusion, v3, segments, facets)
    assert none is None


def test_getGmshPhysicalGroups_reads_names_and_numbers(tmp_path):
    geo = tmp_path / 'tank.geo'
    geo.write_text('Point(1) = {0, 0, 0, 1};\n'
                   'Physical Surface("\'bottom\'", 1) = {1};\n'
                   'Physical Line("\'left\'", 2) = {2, 3};\n'
                   'Physical Volume(7) = {1};\n'
                   'Physical Surface(wall) = {4};\n')
    tags = getGmshPhysicalGroups(str(geo))
    assert tags['bottom'] == 1 and tags['left'] == 2
    assert tags[7] == 7
    assert tags['wall'] == 'wall'


# ---------------------------------------------------------------------------------------------------------------------
# assembling shapes into a domain
# ---------------------------------------------------------------------------------------------------------------------

def test_assemble_two_cuboids_that_touch_share_the_common_vertices():
    d = PiecewiseLinearComplexDomain()
    a = Cuboid(d, dim=(1., 1., 1.), coords=(0.5, 0.5, 0.5))
    b = Cuboid(d, dim=(1., 1., 1.), coords=(1.5, 0.5, 0.5))   # shares the x = 1 face with a
    assembleDomain(d)
    # 16 corners, 4 of them coincide
    assert len(d.vertices) == 12
    assert len(d.facets) == 12
    assert len(d.regions) == 2 and d.regionFlags == [1, 2]
    assert d.boundaryTags[a.name + '_z-'] == 1 and d.boundaryTags[b.name + '_z-'] == 7
    # the domain's box is the union of both
    npt.assert_allclose(d.x, [0., 0., 0.])
    npt.assert_allclose(d.L, [2., 1., 1.])


@pytest.mark.xfail(strict=True, reason='_assembleGeometry renumbers the facets of a shape that shares vertices on a '
                   'numpy array it never uses afterwards (it copies shape.facets first, then edits the original), '
                   'so the facets of the second cuboid name vertices 12-15 of a 12-vertex domain; and it updates '
                   '`verticesFlags` (typo) so vertexFlags keeps 16 entries for 12 vertices (found by C49 R4; C0)')
def test_assemble_two_cuboids_that_touch_gives_valid_facets_and_flags():
    d = PiecewiseLinearComplexDomain()
    Cuboid(d, dim=(1., 1., 1.), coords=(0.5, 0.5, 0.5))
    Cuboid(d, dim=(1., 1., 1.), coords=(1.5, 0.5, 0.5))
    assembleDomain(d)
    assert len(d.vertexFlags) == len(d.vertices)
    for fac in d.facets:
        assert all(0 <= i < len(d.vertices) for i in fac[0])
    # the x- face of the second cuboid is the x+ face of the first
    xm = {tuple(d.vertices[i]) for i in d.facets[6 + 4][0]}
    xp = {tuple(d.vertices[i]) for i in d.facets[2][0]}
    assert xm == xp


def test_assemble_two_rectangles_that_touch_share_vertices_and_segments():
    d = PlanarStraightLineGraphDomain()
    Rectangle(d, dim=(1., 1.), coords=(0.5, 0.5))
    Rectangle(d, dim=(1., 1.), coords=(1.5, 0.5))
    assembleDomain(d)
    assert len(d.vertices) == 6
    assert len(d.segments) == 8
    assert all(0 <= i < len(d.vertices) for seg in d.segments for i in seg)


def test_assemble_a_child_rectangle_in_the_facet_of_its_parent_lists_its_facets_under_the_parent():
    d = PlanarStraightLineGraphDomain()
    outer = Rectangle(d, dim=(4., 4.), coords=(0., 0.))
    inner = Rectangle(d, dim=(1., 1.), coords=(0., 0.))
    inner.setParentShape(outer)
    inner.setHoles([[0., 0.]], indice=[0])
    outer.setHoles([[9., 9.]], indice=[0])
    assembleDomain(d)
    # the domain's facet list is rebuilt: the parent's facet carries the child's as a hole loop
    assert len(d.facets) == 2
    assert d.facets[0][0] == [0, 1, 2, 3]
    assert d.facets[0][1] == [4, 5, 6, 7]
    assert d.holes_ind == [0, 1]
