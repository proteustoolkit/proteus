"""C49 R4: assembleDomain / assembleAuxiliaryVariables of proteus.SpatialTools, the dimension setters of Cuboid and
Rectangle, and the shapes that contain other shapes (holes and volumes handed to gmsh).
"""
import types

import numpy as np
import numpy.testing as npt
import pytest

from proteus import Comm, Gauges
from proteus import BoundaryConditions as bc
from proteus.Domain import (PiecewiseLinearComplexDomain,
                            PlanarStraightLineGraphDomain)
from proteus.SpatialTools import (Cuboid, Rectangle, Tank2D, Tank3D, CustomShape, Circle, Sphere,
                                  assembleAuxiliaryVariables, assembleDomain, get_unit_vector)
from proteus.mbd import BodyDynamics as bd

Comm.init()


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    # assembleDomain may write mesh files
    monkeypatch.chdir(tmp_path)


def test_get_unit_vector():
    npt.assert_allclose(get_unit_vector([3., 0., 4.]), [0.6, 0., 0.8])
    assert np.linalg.norm(get_unit_vector([1., 2., 3.])) == pytest.approx(1.)


def test_cuboid_and_rectangle_setDimensions_keep_the_centre_and_update_volume():
    d = PiecewiseLinearComplexDomain()
    c = Cuboid(d, dim=(1., 1., 1.), coords=(2., 3., 4.))
    c.setDimensions((2., 4., 6.))
    npt.assert_allclose(c.vertices.min(axis=0), [1., 1., 1.])
    npt.assert_allclose(c.vertices.max(axis=0), [3., 5., 7.])
    npt.assert_allclose(c.vertices.mean(axis=0), c.coords)
    assert c.volume == 48.
    c._setInertiaTensor()
    L, W, H = c.dim
    npt.assert_allclose(c.It, np.diag([(W ** 2 + H ** 2) / 12., (L ** 2 + H ** 2) / 12., (W ** 2 + L ** 2) / 12.]))
    d2 = PlanarStraightLineGraphDomain()
    r = Rectangle(d2, dim=(1., 1.), coords=(2., 3.))
    r.setDimensions((4., 2.))
    npt.assert_allclose(r.vertices, [[0., 2.], [4., 2.], [4., 4.], [0., 4.]])
    assert r.volume == 8.
    r._setInertiaTensor()
    assert r.It == pytest.approx((16. + 4.) / 12.)


def test_shape_gauges_are_attached_once_and_listed_by_model():
    d = PlanarStraightLineGraphDomain()
    r = Rectangle(d, dim=(1., 1.), coords=(0., 0.))
    r.attachPointGauges('twp', gauges=((('p',), ((0., 0., 0.),)),), fileName='p.csv')
    r.attachLineGauges('vof', gauges=((('vof',), (((0., 0., 0.), (1., 0., 0.)),)),), fileName='l.csv')
    r.attachLineIntegralGauges('vof', gauges=((('vof',), (((0., 0., 0.), (1., 0., 0.)),)),), fileName='li.csv')
    assert set(r.auxiliaryVariables) == {'Gauge_twp', 'Gauge_vof'}
    assert len(r.auxiliaryVariables['Gauge_twp']) == 1
    assert len(r.auxiliaryVariables['Gauge_vof']) == 2
    assert isinstance(r.auxiliaryVariables['Gauge_twp'][0], Gauges.Gauges)
    # the same gauge twice is refused, an unknown key is logged and not attached
    r._attachAuxiliaryVariable('Gauge_vof', gauge=r.auxiliaryVariables['Gauge_vof'][0])
    assert len(r.auxiliaryVariables['Gauge_vof']) == 2
    r._attachAuxiliaryVariable('nonsense')
    assert 'nonsense' not in r.auxiliaryVariables
    # and a key that is attached already is left alone
    r._attachAuxiliaryVariable('WallFunction', ['w'])
    r._attachAuxiliaryVariable('WallFunction', ['other'])
    assert r.auxiliaryVariables['WallFunction'] == ['w']


def test_assembleAuxiliaryVariables_collects_walls_gauges_and_zones_with_global_flags():
    d = PlanarStraightLineGraphDomain()
    t1 = Tank2D(d, dim=(4., 2.))
    t1.setSponge(x_n=1.)
    t1.setAbsorptionZones(dragAlpha=2., x_n=True, dragBeta=3., porosity=0.5)
    t1.attachPointGauges('twp', gauges=((('p',), ((1., 1., 0.),)),), fileName='p.csv')
    t1.setTurbulentWall(['wall-a', 'wall-b'])
    t1.setTurbulentKWall(['kwall'])
    t2 = Rectangle(d, dim=(1., 1.), coords=(10., 10.))
    t2.attachPointGauges('vof', gauges=((('vof',), ((10., 10., 0.),)),), fileName='v.csv')
    assembleDomain(d)
    aux = d.auxiliaryVariables
    assert set(aux) == {'dissipation', 'kappa', 'ls', 'ls_consrv', 'moveMesh', 'redist', 'twp', 'vof', 'thp', 'vos',
                        'SWFlow'}
    assert aux['kappa'] == ['kwall']
    # twp: the relaxation-zone generator, the two walls, the point gauge
    kinds = [type(a).__name__ for a in aux['twp']]
    assert kinds.count('RelaxationZoneWaveGenerator') == 1
    assert [a for a in aux['twp'] if isinstance(a, str)] == ['wall-a', 'wall-b']
    assert sum(isinstance(a, Gauges.Gauges) for a in aux['twp']) == 1
    assert len(aux['vof']) == 1 and len(aux['SWFlow']) == 1
    # the zone of the sponge is stored under its flag, and the per-region arrays carry its coefficients
    ri = t1.regionIndice['x-']
    flag = t1.regionFlags[ri]
    assert d.dragAlphaTypes[1 + ri] == 2. and d.dragBetaTypes[1 + ri] == 3. and d.porosityTypes[1 + ri] == 0.5
    assert d.epsFact_porous[1 + ri] == pytest.approx(0.5)
    gen = [a for a in aux['twp'] if type(a).__name__ == 'RelaxationZoneWaveGenerator'][0]
    assert flag in gen.zones
    # the second shape's zones (none) leave the default coefficients of its region untouched
    assert d.dragAlphaTypes[1 + len(t1.regions)] == 0.


def test_assembleAuxiliaryVariables_rejects_an_unknown_gauge_model():
    d = PlanarStraightLineGraphDomain()
    r = Rectangle(d, dim=(1., 1.), coords=(0., 0.))
    r.attachPointGauges('nosuchmodel', gauges=((('p',), ((0., 0., 0.),)),), fileName='p.csv')
    with pytest.raises(ValueError):
        assembleDomain(d)


def test_assembleDomain_with_a_rigid_body_fixes_the_mesh_on_it_and_indexes_its_boundary_flags():
    d = PlanarStraightLineGraphDomain()
    tank = Tank2D(d, dim=(4., 2.))
    body_shape = Rectangle(d, dim=(0.5, 0.5), coords=(2., 1.))
    body = bd.RigidBody(shape=body_shape)
    body_shape._attachAuxiliaryVariable('RigidBody', body)
    # a rigid body leaves a hole at its barycenter
    npt.assert_allclose(body_shape.holes, [[2., 1.]])
    assembleDomain(d)
    assert body in d.auxiliaryVariables['twp']
    # flags of the tank come first: 5 of them (y-, x+, y+, x-, sponge)
    assert body.i_start == len(tank.BC_list) + 1
    assert body.i_end == body.i_start + len(body_shape.BC_list)
    # moving-mesh boundary conditions are attached to every boundary of the body, and they move with it
    body.h[:] = [0.1, -0.2, 0.]
    for bcond in body_shape.BC_list:
        np.testing.assert_allclose([bcond.hx_dirichlet.uOfXT(np.array([2., 1., 0.]), 0.),
                                    bcond.hy_dirichlet.uOfXT(np.array([2., 1., 0.]), 0.)], [0.1, -0.2], atol=1e-14)
    # the domain holds the hole
    npt.assert_allclose(d.holes, [[2., 1.]])


def test_assembleAuxiliaryVariables_for_a_chrono_body():
    d = PlanarStraightLineGraphDomain()
    r = Rectangle(d, dim=(1., 1.), coords=(0., 0.))
    seen = {}

    class Body:
        boundaryFlags = np.array([])

        def hx(self, x, t):
            return 1.

        def hy(self, x, t):
            return 2.

        def hz(self, x, t):
            return 3.

        def setBoundaryFlags(self, flags):
            seen['flags'] = flags

    body = Body()
    r.auxiliaryVariables['ChRigidBody'] = body
    d2 = PlanarStraightLineGraphDomain()
    # a second shape first, so that the flags of the body are offset
    Rectangle(d2, dim=(1., 1.), coords=(5., 5.))
    r2 = Rectangle(d2, dim=(1., 1.), coords=(0., 0.))
    r2.auxiliaryVariables['ChRigidBody'] = body
    assembleDomain(d2)
    assert seen['flags'] == [4 + f for f in (1, 2, 3, 4)]
    for bcond in r2.BC_list:
        assert bcond.hx_dirichlet.uOfXT(np.zeros(3), 0.) == 1.
        assert bcond.hz_dirichlet.uOfXT(np.zeros(3), 0.) == 3.


def test_domain_without_a_ShapeRANS_gets_the_generic_boundary_conditions(monkeypatch):
    # a CustomShape is a ShapeRANS too once the module is imported, so test the branch by the predicate
    import proteus.SpatialTools as st
    d = PlanarStraightLineGraphDomain()
    Rectangle(d, dim=(1., 1.), coords=(0., 0.))
    monkeypatch.setattr(st, 'ShapeRANS', type('NotAShape', (), {}))
    assembleDomain(d)
    assert type(d.bc[0]) is bc.BC_Base
    # the auxiliary variables of the shapes are not assembled on this path
    assert not d.auxiliaryVariables


def test_assemble_3D_child_cuboid_is_listed_as_a_shell_inside_the_parent_volume():
    d = PiecewiseLinearComplexDomain()
    outer = Cuboid(d, dim=(4., 4., 4.), coords=(0., 0., 0.))
    inner = Cuboid(d, dim=(1., 1., 1.), coords=(0., 0., 0.))
    inner.setParentShape(outer)
    inner.setHoles([[0., 0., 0.]], indice=[0])
    assembleDomain(d)
    # the parent's volume lists its own six facets and the child's six as a second shell; the child's volume is
    # the hole (holes_ind), so it is dropped from the volumes
    assert len(d.volumes) == 1
    shells = d.volumes[0]
    assert shells[0] == [0, 1, 2, 3, 4, 5]
    assert shells[1] == [6, 7, 8, 9, 10, 11]
    assert d.holes_ind == [1]
    npt.assert_allclose(d.holes, [[0., 0., 0.]])
    # bounding box of the domain is the outer cuboid
    npt.assert_allclose(d.x, [-2., -2., -2.])
    npt.assert_allclose(d.L, [4., 4., 4.])


def test_assemble_3D_two_touching_children_stay_two_cells_inside_the_parent_shell_list():
    d = PiecewiseLinearComplexDomain()
    outer = Cuboid(d, dim=(8., 4., 4.), coords=(0., 0., 0.))
    a = Cuboid(d, dim=(1., 1., 1.), coords=(-0.5, 0., 0.))
    b = Cuboid(d, dim=(1., 1., 1.), coords=(0.5, 0., 0.))
    for child in (a, b):
        child.setParentShape(outer)
    assembleDomain(d)
    # the parent's volume has its own shell and one inner shell per child; each child is also a volume of its own
    assert d.volumes == [[list(range(0, 6)), list(range(6, 12)), list(range(12, 18))],
                         [list(range(6, 12))], [list(range(12, 18))]]
    assert len(d.vertices) == 8 + 8 + 8 - 4      # the shared face of the children is stored once
    assert len(d.facets) == 18
