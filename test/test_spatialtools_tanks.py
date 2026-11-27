"""C49 R4: Tank3D, Tank2D and TankWithObstacles2D of proteus.SpatialTools, geometry checked against the closed-form
geometry of the tank they should describe (a polygon area, a closed hexahedron per cell, the points of the regions).
"""
import itertools

import numpy as np
import numpy.testing as npt
import pytest

from proteus import Comm
from proteus.Domain import (PiecewiseLinearComplexDomain,
                            PlanarStraightLineGraphDomain)
from proteus.SpatialTools import Tank2D, Tank3D, TankWithObstacles2D, assembleDomain

Comm.init()


def _shoelace(points):
    p = np.asarray(points, dtype=float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _inside(poly, pt):
    """even-odd ray casting"""
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            if x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
                inside = not inside
    return inside


# ---------------------------------------------------------------------------------------------------------------------
# Tank3D
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('combo', list(itertools.product([0, 1], repeat=4)))
def test_tank3D_sponge_combinations_are_closed_cells_that_fill_the_extended_box(combo):
    xp, xn, yp, yn = combo
    L, W, H = 4., 3., 2.
    spx_p, spx_n, spy_p, spy_n = 1., 1.5, 2., 0.5
    d = PiecewiseLinearComplexDomain()
    t = Tank3D(d, dim=(L, W, H))
    t.setSponge(x_p=spx_p if xp else None, x_n=spx_n if xn else None,
                y_p=spy_p if yp else None, y_n=spy_n if yn else None)
    nsp = sum(combo)
    assert len(t.regions) == len(t.regionFlags) == len(t.volumes) == 1 + nsp
    assert len(t.facetFlags) == len(t.facets)
    assert len(t.vertexFlags) == len(t.vertices)
    V = t.vertices
    # box of everything: the tank grown outwards by the sponge layers
    lo = np.array([-(spx_n if xn else 0.), -(spy_n if yn else 0.), 0.])
    hi = np.array([L + (spx_p if xp else 0.), W + (spy_p if yp else 0.), H])
    npt.assert_allclose(V.min(axis=0), lo)
    npt.assert_allclose(V.max(axis=0), hi)
    for vi, vol in enumerate(t.volumes):
        # each cell is a closed polyhedron: every edge of its faces is used by exactly two faces
        edges = {}
        for f in vol[0]:
            fac = t.facets[f][0]
            for k in range(len(fac)):
                e = tuple(sorted((int(fac[k - 1]), int(fac[k]))))
                edges[e] = edges.get(e, 0) + 1
        assert set(edges.values()) == {2}, 'cell %d is not closed' % vi
        idx = sorted({i for f in vol[0] for i in t.facets[f][0]})
        assert len(idx) == 8
        P = V[idx]
        # the region point lies inside its cell's box
        r = t.regions[vi]
        assert np.all(r > P.min(axis=0) - 1e-12) and np.all(r < P.max(axis=0) + 1e-12)
    # names of the sponge regions
    assert set(t.regionIndice) == {'tank'} | {k for k, on in zip(('x+', 'x-', 'y+', 'y-'), combo) if on}
    # flags: the walls between a sponge and another sponge are 'wall', the end faces keep their own
    bt = t.boundaryTags
    assert set(t.facetFlags) <= set(bt.values())
    assert (bt['sponge'] in t.facetFlags) == bool(nsp)


def _tiling_params():
    out = []
    for combo in itertools.product([0, 1], repeat=4):
        xp, xn, yp, yn = combo
        marks = []
        if (xp or xn) and (yp or yn):
            marks = [pytest.mark.xfail(strict=True, reason=(
                'Tank3D lists the x sponge cell with the y extent of the whole extended tank (its facet list reaches '
                'the y sponge facet), so it overlaps the y sponge cell in the corner (x_sponge * y_sponge * height); '
                'the volumes list is used by gmsh (found by C49 R4; C0)'))]
        out.append(pytest.param(combo, marks=marks))
    return out


@pytest.mark.parametrize('combo', _tiling_params())
def test_tank3D_sponge_cells_tile_the_extended_box_without_overlap(combo):
    xp, xn, yp, yn = combo
    L, W, H = 4., 3., 2.
    sp = dict(x_p=1. if xp else None, x_n=1.5 if xn else None, y_p=2. if yp else None, y_n=0.5 if yn else None)
    d = PiecewiseLinearComplexDomain()
    t = Tank3D(d, dim=(L, W, H))
    t.setSponge(**sp)
    lo = np.array([-(sp['x_n'] or 0.), -(sp['y_n'] or 0.), 0.])
    hi = np.array([L + (sp['x_p'] or 0.), W + (sp['y_p'] or 0.), H])
    total = 0.
    for vol in t.volumes:
        idx = sorted({i for f in vol[0] for i in t.facets[f][0]})
        total += np.prod(np.ptp(t.vertices[idx], axis=0))
    npt.assert_allclose(total, np.prod(hi - lo))


def test_tank3D_assemble_and_translate_with_sponges():
    d = PiecewiseLinearComplexDomain()
    t = Tank3D(d, dim=(2., 1., 1.))
    t.setSponge(x_p=1., x_n=1.)
    t.translate(np.array([10., 20., 30.]))
    npt.assert_allclose(t.vertices.min(axis=0), [9., 20., 30.])
    npt.assert_allclose(t.vertices.max(axis=0), [13., 21., 31.])
    assembleDomain(d)
    assert len(d.vertices) == len(t.vertices)
    assert d.regionFlags == [1, 2, 3]
    # a tank with coords given is centred there, not at the half-dimensions
    d2 = PiecewiseLinearComplexDomain()
    t2 = Tank3D(d2, dim=(2., 2., 2.), coords=(5., 5., 5.))
    npt.assert_allclose(t2.vertices.min(axis=0), [4., 4., 4.])
    npt.assert_allclose(t2.vertices.max(axis=0), [6., 6., 6.])
    assert t2.from_0 is False


def test_tank3D_zones_for_every_sponge():
    d = PiecewiseLinearComplexDomain()
    t = Tank3D(d, dim=(4., 3., 2.))
    t.setSponge(x_p=1., x_n=1.5, y_p=2., y_n=0.5)
    t.setAbsorptionZones(dragAlpha=10., allSponge=True)
    assert set(t.zones) == {t.regionFlags[t.regionIndice[k]] for k in ('x+', 'x-', 'y+', 'y-')}
    for key, orient in (('x-', [1., 0., 0.]), ('x+', [-1., 0., 0.]), ('y-', [0., 1., 0.]), ('y+', [0., -1., 0.])):
        z = t.zones[t.regionFlags[t.regionIndice[key]]]
        assert z.zone_type == 'absorption'
        npt.assert_allclose(z.orientation, orient)
        assert z.dragAlpha == 10.
        # the zone's centre is the centre of its sponge region
        npt.assert_allclose(z.center, t.regions[t.regionIndice[key]])
        assert z.epsFact_porous == pytest.approx(t.spongeLayers[key] / 2.)


def test_tank3D_generation_zones_set_an_unsteady_inlet():
    from proteus import WaveTools as wt
    waves = wt.MonochromaticWaves(period=0.8, waveHeight=0.03, mwl=0.9, depth=0.9, g=np.array([0., 0., -9.81]),
                                  waveDir=np.array([1., 0., 0.]))
    d = PiecewiseLinearComplexDomain()
    t = Tank3D(d, dim=(4., 3., 2.))
    t.setSponge(x_p=1., x_n=1.5, y_p=2., y_n=0.5)
    t.setGenerationZones(dragAlpha=5., smoothing=0.1, waves=waves, allSponge=True, dragBeta=1., porosity=0.7)
    for key in ('x+', 'x-', 'y+', 'y-'):
        z = t.zones[t.regionFlags[t.regionIndice[key]]]
        assert z.zone_type == 'generation'
        assert (z.dragBeta, z.porosity) == (1., 0.7)
        # the wall of that side now imposes the wave's velocity
        assert t.BC[key].u_dirichlet.uOfXT is not None and t.BC[key].vof_dirichlet.uOfXT is not None


# ---------------------------------------------------------------------------------------------------------------------
# Tank2D and TankWithObstacles2D
# ---------------------------------------------------------------------------------------------------------------------

def test_tank2D_with_sponges_is_one_polygon_per_region():
    d = PlanarStraightLineGraphDomain()
    t = Tank2D(d, dim=(4., 2.), coords=(10., 5.))
    npt.assert_allclose(t.vertices, [[8., 4.], [12., 4.], [12., 6.], [8., 6.]])
    t.setSponge(x_n=1., x_p=2.)
    # three rectangles: the sponges are outside the tank, the shared walls (x- and x+ of the tank) flag 'sponge'
    assert len(t.vertices) == 8
    assert _shoelace(t.vertices[[0, 1, 2, 3]]) == pytest.approx(8.)
    assert len(t.facets) == 3 and len(t.regions) == 3
    npt.assert_array_equal(t.regionFlags, [1, 2, 3])
    assert t.regionIndice == {'tank': 0, 'x-': 1, 'x+': 2}
    bt = t.boundaryTags
    assert t.segmentFlags[3] == bt['sponge'] and t.segmentFlags[1] == bt['sponge']
    # each region point is inside its own rectangle
    rects = {0: [[8, 4], [12, 4], [12, 6], [8, 6]], 1: [[7, 4], [8, 4], [8, 6], [7, 6]],
             2: [[12, 4], [14, 4], [14, 6], [12, 6]]}
    for i, poly in rects.items():
        assert _inside(poly, t.regions[i])
    # the facet loops reference existing vertices
    assert np.all(t.facets < len(t.vertices))


def test_tank2D_zones_use_the_region_of_their_sponge():
    d = PlanarStraightLineGraphDomain()
    t = Tank2D(d, dim=(4., 2.))
    t.setSponge(x_n=1., x_p=1.)
    t.setAbsorptionZones(dragAlpha=3., x_n=True, x_p=True)
    left = t.zones[t.regionFlags[t.regionIndice['x-']]]
    right = t.zones[t.regionFlags[t.regionIndice['x+']]]
    npt.assert_allclose(left.center, [-0.5, 1., 0.])
    npt.assert_allclose(right.center, [4.5, 1., 0.])
    npt.assert_allclose(left.orientation[:2], [1., 0.])
    npt.assert_allclose(right.orientation[:2], [-1., 0.])


def _obstacle_tank(obstacles, dim=(10., 3.), **kw):
    d = PlanarStraightLineGraphDomain()
    return TankWithObstacles2D(d, dim=dim, obstacles=obstacles, **kw)


def _main_polygon(t, n_sponge_vertices=0):
    v = t.vertices if n_sponge_vertices == 0 else t.vertices[:-n_sponge_vertices]
    return v


SILL = [[[3., 0.], [4., 1.], [6., 1.], [7., 0.]]]
RAMP = [[[6., 0.], [10., 1.]]]


def test_tank_with_a_sill_removes_its_area_and_adds_the_floor_corners():
    t = _obstacle_tank(SILL)
    # the tank 10 x 3 minus the trapezoid (bases 4 and 2, height 1)
    assert _shoelace(t.vertices) == pytest.approx(30. - 3.)
    # the segments close one loop through every vertex
    assert [s[0] for s in t.segments] == list(range(len(t.vertices)))
    assert t.segments[-1].tolist() == [len(t.vertices) - 1, 0]
    bt = t.boundaryTags
    assert bt['obstacle1'] == 6
    npt.assert_array_equal(t.segmentFlags, [1, 1, 1, 1, bt['x+'], bt['y+'], bt['x-'], 1])
    # the region point is in the water (a corner of the free domain), not in the sill
    assert _inside(t.vertices.tolist(), t.regions[0])
    assert t.regionIndice == {'tank': 0}


def test_tank_with_a_ramp_ending_on_the_wall():
    t = _obstacle_tank(RAMP)
    assert _shoelace(t.vertices) == pytest.approx(30. - 2.)
    bt = t.boundaryTags
    npt.assert_array_equal(t.segmentFlags, [1, bt['x+'], bt['y+'], bt['x-'], 1])
    assert _inside(t.vertices.tolist(), t.regions[0])


def test_tank_with_a_closed_obstacle_has_its_own_region_and_bottom_segment():
    t = _obstacle_tank(SILL, hole=False, obstacle_regions=[[5., 0.5]])
    bt = t.boundaryTags
    assert t.regionIndice == {'obstacle1': 0, 'tank': 1}
    npt.assert_array_equal(t.regionFlags, [1, 2])
    npt.assert_allclose(t.regions[0], [5., 0.5])
    # the obstacle's own surface carries its tag, the segment that closes it at the bottom is 'y-'
    npt.assert_array_equal(t.vertexFlags[:4], bt['obstacle1'])
    assert t.segments[-1].tolist() == [0, 3] and t.segmentFlags[-1] == bt['y-']
    # one region per obstacle is required
    with pytest.raises(AssertionError):
        _obstacle_tank(SILL, hole=False, obstacle_regions=[])


def test_tank_with_special_boundaries_tags_the_segment_from_that_vertex():
    t = _obstacle_tank(RAMP, special_boundaries={'ramp': [[6., 0.]]})
    bt = t.boundaryTags
    assert bt['ramp'] == 6 and bt['obstacle1'] == 7
    assert 'ramp' in t.BC and t.BC['ramp'] in t.BC_list
    assert t.vertexFlags[0] == bt['ramp'] and t.segmentFlags[0] == bt['ramp']


@pytest.mark.parametrize('sponge', [dict(x_n=2.), dict(x_p=3.), dict(x_n=2., x_p=3.)])
def test_tank_with_obstacles_and_sponges(sponge):
    t = _obstacle_tank(RAMP)
    t.setSponge(**sponge)
    nsp = 2 * len(sponge)
    main = t.vertices[:-nsp]
    assert _shoelace(main) == pytest.approx(28.)
    # a sponge is a rectangle on the wall: its height is the wall height at that wall
    area = 28.
    if 'x_n' in sponge:
        area += sponge['x_n'] * 3.
    if 'x_p' in sponge:
        area += sponge['x_p'] * 2.   # the ramp raises the floor to 1 at x+
    assert len(t.regions) == 1 + len(sponge)
    assert set(t.regionIndice) == {'tank'} | {'x-' if k == 'x_n' else 'x+' for k in sponge}
    bt = t.boundaryTags
    assert bt['sponge'] in t.segmentFlags
    # all segments reference existing vertices; the sponge walls carry the end tags
    assert t.segments.max() == len(t.vertices) - 1
    if 'x_n' in sponge:
        assert bt['x-'] in t.segmentFlags[-3 * len(sponge):]
    # the sponge regions lie outside the tank polygon, on the right side
    if 'x_n' in sponge:
        assert t.regions[t.regionIndice['x-']][0] < 0.
    if 'x_p' in sponge:
        assert t.regions[t.regionIndice['x+']][0] > 10.
    # zones work for both
    t.setAbsorptionZones(dragAlpha=1., x_n='x_n' in sponge, x_p='x_p' in sponge)
    assert len(t.zones) == len(sponge)


def test_tank_with_obstacles_generation_zones():
    from proteus import WaveTools as wt
    waves = wt.MonochromaticWaves(period=0.8, waveHeight=0.03, mwl=0.9, depth=0.9, g=np.array([0., -9.81, 0.]),
                                  waveDir=np.array([1., 0., 0.]))
    t = _obstacle_tank(RAMP)
    t.setSponge(x_n=2., x_p=3.)
    t.setGenerationZones(dragAlpha=1., smoothing=0.1, waves=waves, x_n=True, x_p=True)
    zl = t.zones[t.regionFlags[t.regionIndice['x-']]]
    zr = t.zones[t.regionFlags[t.regionIndice['x+']]]
    assert zl.zone_type == zr.zone_type == 'generation'
    npt.assert_allclose(zl.center, [-1., 1.5, 0.])
    npt.assert_allclose(zr.center, [11.5, 2., 0.])     # the x+ wall spans y = 1..3 (the ramp's end)


def test_tank_with_obstacles_rejects_bad_obstacles():
    with pytest.raises(ValueError):
        _obstacle_tank([[[3., 0.], [4., 1.], [3., 0.]]])      # starts and ends on the same point
    with pytest.raises(ValueError):
        _obstacle_tank([[[3., 0.], [4., 1.], [5., 1.]]])      # ends off the walls


def test_tank_with_two_obstacles_joins_the_corners_between_them():
    d = PlanarStraightLineGraphDomain()
    # a sill on the floor and a ramp that starts on the floor to the right of it and ends on the x+ wall
    t = TankWithObstacles2D(d, dim=(10., 3.),
                            obstacles=[[[2., 0.], [3., 1.], [4., 0.]], [[6., 0.], [10., 1.]]])
    assert _shoelace(t.vertices) == pytest.approx(30. - 1. - 2.)
    assert t.boundaryTags['obstacle2'] == 7
    assert _inside(t.vertices.tolist(), t.regions[0])


@pytest.mark.xfail(strict=True, raises=KeyError,
                   reason="TankWithObstacles2D: an obstacle that starts at a corner of the tank (documented as "
                          "allowed) looks its tag up as boundaryTags['x-y-'], which does not exist (found by C49 "
                          "R4; kernel territory: C0)")
def test_tank_with_an_obstacle_that_starts_in_a_corner():
    t = _obstacle_tank([[[0., 0.], [0., 1.]]])
    assert _shoelace(t.vertices) == pytest.approx(30.)


def test_tank_with_obstacles_on_all_four_walls_joins_them_in_order():
    # floor (y-), x+ wall, ceiling (y+), x- wall, counterclockwise
    obstacles = [[[2., 0.], [3., 0.5], [4., 0.]],
                 [[10., 1.], [9., 1.5], [10., 2.]],
                 [[8., 3.], [7., 2.5], [6., 3.]],
                 [[0., 2.], [1., 1.5], [0., 1.]]]
    t = _obstacle_tank(obstacles)
    area = 30. - 0.5 * 2. * 0.5 - 0.5 * 1. * 1. - 0.5 * 2. * 0.5 - 0.5 * 1. * 1.
    assert _shoelace(t.vertices) == pytest.approx(area)
    assert _inside(t.vertices.tolist(), t.regions[0])
    # the corners between the obstacles are vertices of the loop: (10, 0) after the floor obstacle, ...
    for corner in ([10., 0.], [10., 3.], [0., 3.], [0., 0.]):
        assert corner in t.vertices.tolist()
    assert set(t.segmentFlags) <= set(t.boundaryTags.values())
