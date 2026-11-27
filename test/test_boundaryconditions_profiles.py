"""C49 R4: proteus.BoundaryConditions profiles and relaxation zones that test_boundaryconditions.py leaves out.

Each condition is evaluated at points where a closed-form answer exists: a hydrostatic pressure against the weight of
the two fluid columns, a blend of velocities against the Heaviside the module documents, the displacement of a body
point against the rigid-motion formula, a relaxation zone's signed distance against the projection of the centre
offset on its orientation.
"""
import numpy as np
import numpy.testing as npt
import pytest

from proteus import Comm
from proteus import BoundaryConditions as bc
from proteus import WaveTools as wt
from proteus.ctransportCoefficients import smoothedHeaviside
from proteus.Domain import PiecewiseLinearComplexDomain, PlanarStraightLineGraphDomain
from proteus.SpatialTools import Rectangle, Cuboid

Comm.init()


def _bc(b_or, nd=2):
    return bc.BC_RANS(b_or=np.array([b_or], dtype=float), b_i=0, nd=nd)


def _x(y, nd=2, x0=0.3):
    out = np.zeros(3)
    out[0] = x0
    out[nd - 1 if nd == 3 else 1] = y
    return out


# ---------------------------------------------------------------------------------------------------------------------
# hydrostatic outlet with depth
# ---------------------------------------------------------------------------------------------------------------------

RHO_UP, RHO_DOWN, SEA, REF, SMOOTH = 1.2, 1000., 0.5, 1.0, 0.1
G = np.array([0., -9.81, 0.])


def _hydro(b_or=(1., 0., 0.), **kw):
    b = _bc(b_or)
    b.setHydrostaticPressureOutletWithDepth(seaLevel=SEA, rhoUp=RHO_UP, rhoDown=RHO_DOWN, g=G, refLevel=REF,
                                            smoothing=SMOOTH, **kw)
    return b


def test_hydrostatic_pressure_outlet_with_depth_is_the_weight_of_the_two_columns():
    b = _hydro()
    for y in (-0.2, 0.0, 0.3):                     # below the sea level, outside the smoothing
        expected = 9.81 * (RHO_UP * (REF - SEA) + RHO_DOWN * (SEA - y))
        npt.assert_allclose(b.p_dirichlet.uOfXT(_x(y), 0.), expected, rtol=1e-12)
        npt.assert_allclose(b.pInit_dirichlet.uOfXT(_x(y), 0.), expected, rtol=1e-12)
    for y in (0.7, 0.9):                           # above it: only the air column over the point
        npt.assert_allclose(b.p_dirichlet.uOfXT(_x(y), 0.), 9.81 * RHO_UP * (REF - y), rtol=1e-12)
    # at the reference level the pressure is pRef (0 by default), and pRef shifts it
    npt.assert_allclose(b.p_dirichlet.uOfXT(_x(REF), 0.), 0., atol=1e-12)
    b2 = _hydro(pRef=5.)
    npt.assert_allclose(b2.p_dirichlet.uOfXT(_x(REF), 0.), 5., atol=1e-12)
    # across the smoothing layer the pressure is continuous and monotone in depth
    ys = np.linspace(SEA + 2 * SMOOTH, SEA - 2 * SMOOTH, 41)
    p = np.array([b.p_dirichlet.uOfXT(_x(y), 0.) for y in ys])
    assert np.all(np.diff(p) > 0.)
    # the pressure increment is zero and the level set is the signed distance
    assert b.pInc_dirichlet.uOfXT(_x(0.), 0.) == 0.
    npt.assert_allclose(b.phi_dirichlet.uOfXT(_x(0.8), 0.), 0.3)


def test_hydrostatic_pressure_outlet_with_depth_vof_goes_from_water_to_air_through_the_smoothing():
    b = _hydro(air=1., water=0.)
    assert b.vof_dirichlet.uOfXT(_x(SEA - 2 * SMOOTH), 0.) == 0.
    assert b.vof_dirichlet.uOfXT(_x(SEA + 2 * SMOOTH), 0.) == 1.
    mid = b.vof_dirichlet.uOfXT(_x(SEA), 0.)
    npt.assert_allclose(mid, smoothedHeaviside(SMOOTH, 0.))
    assert 0. < b.vof_dirichlet.uOfXT(_x(SEA + 0.5 * SMOOTH), 0.) < 1.
    # other phase values are mixed with the same weights
    b = _hydro(air=3., water=2.)
    assert b.vof_dirichlet.uOfXT(_x(SEA - 2 * SMOOTH), 0.) == 2.
    assert b.vof_dirichlet.uOfXT(_x(SEA + 2 * SMOOTH), 0.) == 3.


@pytest.mark.parametrize('b_or, fixed, free', [((1., 0., 0.), ('v', 'w'), 'u'),
                                               ((-1., 0., 0.), ('v', 'w'), 'u'),
                                               ((0., 1., 0.), ('u', 'w'), 'v'),
                                               ((0., 0., 1.), ('u', 'v'), 'w')])
def test_hydrostatic_pressure_outlet_with_depth_sets_the_tangential_velocity_to_zero(b_or, fixed, free):
    b = _hydro(b_or=b_or)
    for comp in fixed:
        assert getattr(b, comp + '_dirichlet').uOfXT(_x(0.2), 0.) == 0.
    # the normal component is left free (diffusive zero gradient)
    assert getattr(b, free + '_diffusive').uOfXT(_x(0.2), 0.) == 0.
    # sediment variables carry zero fluxes
    assert b.vos_advective.uOfXT(_x(0.2), 0.) == 0.


def test_hydrostatic_pressure_outlet_with_depth_blends_velocity_and_turbulence_inflow():
    U, Uwind = (0.4, -0.1, 0.), (2., 0.5, 0.)
    b = _hydro(U=U, Uwind=Uwind, kInflow=1e-3, dissipationInflow=1e-4, kInflowAir=1e-5, dissipationInflowAir=1e-6)
    # below the sea level: the water values; above the smoothing: the air values; in it: the Heaviside blend
    low, high = _x(SEA - 0.2), _x(SEA + 2 * SMOOTH)
    for i, comp in enumerate(('u', 'v', 'w')):
        f = getattr(b, comp + '_dirichlet').uOfXT
        assert f(low, 0.) == U[i]
        assert f(high, 0.) == Uwind[i]
    assert b.k_dirichlet.uOfXT(low, 0.) == 1e-3 and b.k_dirichlet.uOfXT(high, 0.) == 1e-5
    assert b.dissipation_dirichlet.uOfXT(low, 0.) == 1e-4 and b.dissipation_dirichlet.uOfXT(high, 0.) == 1e-6
    mid = _x(SEA + 0.5 * SMOOTH)
    H = smoothedHeaviside(SMOOTH / 2., 0.5 * SMOOTH - SMOOTH / 2.)
    npt.assert_allclose(b.u_dirichlet.uOfXT(mid, 0.), H * Uwind[0] + (1 - H) * U[0])
    npt.assert_allclose(b.k_dirichlet.uOfXT(mid, 0.), H * 1e-5 + (1 - H) * 1e-3)
    npt.assert_allclose(b.dissipation_dirichlet.uOfXT(mid, 0.), H * 1e-6 + (1 - H) * 1e-4)
    # with U given and no wind the air is at rest
    b = _hydro(U=U)
    assert b.u_dirichlet.uOfXT(high, 0.) == 0.


def test_hydrostatic_pressure_outlet_with_depth_needs_an_orientation():
    b = bc.BC_RANS(nd=2)
    with pytest.raises(AssertionError):
        b.setHydrostaticPressureOutletWithDepth(seaLevel=SEA, rhoUp=1., rhoDown=2., g=G, refLevel=REF, smoothing=0.1)


# ---------------------------------------------------------------------------------------------------------------------
# two-phase velocity inlet: the branches
# ---------------------------------------------------------------------------------------------------------------------

def test_two_phase_velocity_inlet_in_3D_uses_z_as_vertical_and_blends_with_the_smoothed_heaviside():
    b = bc.BC_RANS(b_or=np.array([[1., 0., 0.]]), b_i=0, nd=3)
    U, Uwind, level, smooth = (1., 0., 0.), (5., 0., 0.), 2., 0.3
    b.setTwoPhaseVelocityInlet(U=U, Uwind=Uwind, waterLevel=level, smoothing=smooth, air=1., water=0.,
                               kInflow=1., dissipationInflow=2., kInflowAir=3., dissipationInflowAir=4.)
    assert b.BC_type == 'TwoPhaseVelocityInlet'

    def at(z):
        return np.array([0.1, 0.2, z])
    # vertical axis is the last one in 3D: z below the water level is water, above it air
    assert b.u_dirichlet.uOfXT(at(1.), 0.) == 1. and b.u_dirichlet.uOfXT(at(3.), 0.) == 5.
    assert b.vof_dirichlet.uOfXT(at(1.), 0.) == 0. and b.vof_dirichlet.uOfXT(at(3.), 0.) == 1.
    inside = at(level + 0.1)
    H = smoothedHeaviside(smooth, 0.1 - smooth / 2.)
    npt.assert_allclose(b.u_dirichlet.uOfXT(inside, 0.), H * 5. + (1 - H) * 1.)
    npt.assert_allclose(b.k_dirichlet.uOfXT(inside, 0.), H * 3. + (1 - H) * 1.)
    npt.assert_allclose(b.dissipation_dirichlet.uOfXT(inside, 0.), H * 4. + (1 - H) * 2.)
    # vof uses its own smoothing about the level
    npt.assert_allclose(b.vof_dirichlet.uOfXT(inside, 0.), smoothedHeaviside(smooth, 0.1))
    npt.assert_allclose(b.vof_dirichlet.uOfXT(at(level - 0.1), 0.), smoothedHeaviside(smooth, -0.1))
    # the advective pressure flux is minus the blended normal velocity
    npt.assert_allclose(b.p_advective.uOfXT(at(1.), 0.), -1.)
    npt.assert_allclose(b.p_advective.uOfXT(at(3.), 0.), -5.)
    npt.assert_allclose(b.p_advective.uOfXT(inside, 0.), -(H * 5. + (1 - H) * 1.))
    npt.assert_allclose(b.phi_dirichlet.uOfXT(at(2.5), 0.), 0.5)


def test_two_phase_velocity_inlet_default_wind_is_at_rest():
    b = _bc((0., 1., 0.))
    b.setTwoPhaseVelocityInlet(U=(0., 1., 0.), waterLevel=0.5, smoothing=0.1)
    assert b.v_dirichlet.uOfXT(_x(0.2), 0.) == 1.
    assert b.v_dirichlet.uOfXT(_x(0.9), 0.) == 0.


# ---------------------------------------------------------------------------------------------------------------------
# atmosphere with turbulence values, turbulent walls, mesh motion
# ---------------------------------------------------------------------------------------------------------------------

def test_atmosphere_takes_turbulence_inflow_values():
    b = _bc((0., 1., 0.))
    b.setAtmosphere(kInflow=0.5, dInflow=0.25, vof_air=1.)
    assert b.k_dirichlet.uOfXT(_x(0.), 0.) == 0.5
    assert b.dissipation_dirichlet.uOfXT(_x(0.), 0.) == 0.25
    assert b.p_dirichlet.uOfXT(_x(0.), 0.) == 0.
    # a boundary normal to z also fixes the z diffusive flux
    b = _bc((0., 0., 1.), nd=3)
    b.setAtmosphere()
    assert b.w_diffusive.uOfXT(_x(0., nd=3), 0.) == 0.
    assert b.k_dirichlet.uOfXT(_x(0., nd=3), 0.) == 1e-20


def test_turbulent_dirichlet_and_zero_gradient():
    b = _bc((1., 0., 0.))
    b.setTurbulentDirichlet(kVal=0.01, dissipationVal=0.02)
    assert b.k_dirichlet.uOfXT(_x(0.), 0.) == 0.01
    assert b.dissipation_dirichlet.uOfXT(_x(0.), 0.) == 0.02
    # the advective and diffusive conditions are reset
    assert b.k_advective.uOfXT is None and b.k_diffusive.uOfXT is None
    b.setTurbulentZeroGradient()
    assert b.k_dirichlet.uOfXT(_x(0.), 0.) == 0. and b.dissipation_dirichlet.uOfXT(_x(0.), 0.) == 0.
    assert b.k_diffusive.uOfXT(_x(0.), 0.) == 0. and b.dissipation_diffusive.uOfXT(_x(0.), 0.) == 0.
    assert b.k_advective.uOfXT is None


def _rot_z(a):
    # row-vector convention of the module: x_new = x_0 . R
    return np.array([[np.cos(a), np.sin(a), 0.], [-np.sin(a), np.cos(a), 0.], [0., 0., 1.]])


def test_setMoveMesh_is_the_rigid_displacement_of_the_point_and_defaults_to_no_rotation():
    b = _bc((1., 0., 0.))
    last, h = np.array([1., 2., 0.]), np.array([0.1, -0.2, 0.3])
    b.setMoveMesh(last_pos=last, h=h)                  # no rotation matrix: identity
    x = np.array([3., 5., 1.])
    hx = [getattr(b, 'h%s_dirichlet' % c).uOfXT(x, 0.) for c in 'xyz']
    npt.assert_allclose(hx, h, atol=1e-14)
    # with a rotation about the last position the point moves by (x - last).R - (x - last), plus the translation
    R = _rot_z(0.3)
    b.setMoveMesh(last_pos=last, h=h, rot_matrix=R)
    expected = (x - last) @ R - (x - last) + h
    hx = [getattr(b, 'h%s_dirichlet' % c).uOfXT(x, 0.) for c in 'xyz']
    npt.assert_allclose(hx, expected, atol=1e-14)
    # a point on the axis of the rotation only translates
    on_axis = [getattr(b, 'h%s_dirichlet' % c).uOfXT(last, 0.) for c in 'xyz']
    npt.assert_allclose(on_axis, h, atol=1e-14)


def test_setRigidBodyMoveMesh_follows_the_body_state_as_it_changes():
    class Body:
        nd = 2
        last_position = np.array([1., 2., 0.])
        rotation_matrix = _rot_z(0.5)
        h = np.array([0.2, 0.1, 0.])

    body = Body()
    b = _bc((1., 0., 0.))
    b.setRigidBodyMoveMesh(body)
    assert b.hz_dirichlet.uOfXT is None       # a 2D body has no vertical displacement
    x = np.array([2., 4., 0.])
    expected = (x - body.last_position) @ body.rotation_matrix - (x - body.last_position) + body.h
    got = [getattr(b, 'h%s_dirichlet' % c).uOfXT(x, 0.) for c in 'xy']
    npt.assert_allclose(got, expected[:2], atol=1e-14)
    # the conditions read the body at evaluation time, so a step of the body changes them
    body.h = np.array([0.3, 0., 0.])
    body.rotation_matrix = np.eye(3)
    got = [getattr(b, 'h%s_dirichlet' % c).uOfXT(x, 0.) for c in 'xy']
    npt.assert_allclose(got, body.h[:2], atol=1e-14)


# ---------------------------------------------------------------------------------------------------------------------
# relaxation zones
# ---------------------------------------------------------------------------------------------------------------------

def _shape2d():
    return Rectangle(PlanarStraightLineGraphDomain(), dim=(1., 1.), coords=(0., 0.))


def _shape3d():
    return Cuboid(PiecewiseLinearComplexDomain(), dim=(1., 1., 1.), coords=(0., 0., 0.))


def _waves():
    return wt.MonochromaticWaves(period=1.2, waveHeight=0.1, mwl=0.5, depth=0.5, g=np.array([0., -9.81, 0.]),
                                 waveDir=np.array([1., 0., 0.]))


def test_relaxation_zone_phi_is_the_distance_to_the_centre_along_the_orientation():
    z = bc.RelaxationZone(shape=_shape2d(), zone_type='absorption', center=np.array([2., 1., 0.]),
                          orientation=np.array([0.6, 0.8, 0.]), epsFact_porous=0.5)
    z.calculate_init()
    for x in ([0., 0., 0.], [3., 1.5, 0.], [2., 1., 0.]):
        d = np.array([2., 1.]) - np.array(x[:2])
        npt.assert_allclose(z.calculate_phi_python(np.array(x)), 0.6 * d[0] + 0.8 * d[1], atol=1e-14)
    # an absorption zone has no velocity of its own
    npt.assert_allclose(z.calculate_vel_python(np.array([0., 0., 0.]), 1.), 0.)


def test_relaxation_zone_phi_in_3D_uses_the_vertical_component_too():
    z = bc.RelaxationZone(shape=_shape3d(), zone_type='absorption', center=np.array([1., 2., 3.]),
                          orientation=np.array([0., 0., 1.]), epsFact_porous=0.5)
    z.calculate_init()
    npt.assert_allclose(z.calculate_phi_python(np.array([0., 0., 1.])), 2.)
    assert z.vert_axis == 2


def test_porous_zone_phi_is_a_large_negative_number_inside_and_has_no_velocity():
    z = bc.RelaxationZone(shape=_shape2d(), zone_type='porous', center=np.array([0., 0., 0.]),
                          orientation=np.array([1., 0., 0.]), epsFact_porous=1., dragAlpha=2., dragBeta=3.,
                          porosity=0.4)
    z.calculate_init()
    assert z.calculate_phi_python(np.array([5., 5., 0.])) == -10000.
    npt.assert_allclose(z.calculate_vel_python(np.array([5., 5., 0.]), 0.), 0.)
    assert (z.dragAlpha, z.dragBeta, z.porosity) == (2., 3., 0.4)


def test_generation_zone_velocity_is_the_wave_velocity_and_blends_to_wind_above_the_crest():
    waves = _waves()
    wind = np.array([4., 0., 0.])
    z = bc.RelaxationZone(shape=_shape2d(), zone_type='generation', center=np.array([0., 0.5, 0.]),
                          orientation=np.array([1., 0., 0.]), epsFact_porous=0.5, waves=waves, wind_speed=wind,
                          smoothing=0.2)
    z.calculate_init()
    t = 0.37
    # below the free surface the particle velocity of the wave itself (WaveTools)
    x = np.array([0.4, 0.2, 0.])
    npt.assert_allclose(z.calculate_vel_python(x, t), waves.u(x, t), atol=1e-12)
    npt.assert_allclose(z.calculate_vel_python(x, t), waves.u(x, t), atol=1e-12)
    # far above: the wind
    npt.assert_allclose(z.calculate_vel_python(np.array([0.4, 2., 0.]), t), wind, atol=1e-12)
    # in the smoothing layer above the crest: the Heaviside blend of the wind and the velocity under the crest
    eta = waves.eta(np.array([0.4, 0., 0.]), t)
    phi = 0.5 + 0.05                                  # height above the instantaneous level
    xs = np.array([0.4, 0.5 + eta + 0.05, 0.])
    H = smoothedHeaviside(0.1, 0.05 - 0.1)
    water = waves.u(np.array([0.4, 0.5 + eta, 0.]), t)
    npt.assert_allclose(z.calculate_vel_python(xs, t), H * wind + (1 - H) * water, atol=1e-12)


def test_relaxation_zone_wave_generator_fills_the_quadrature_arrays_per_material_type():
    shape = _shape2d()
    absorb = bc.RelaxationZone(shape=shape, zone_type='absorption', center=np.array([0., 0., 0.]),
                               orientation=np.array([1., 0., 0.]), epsFact_porous=1.)
    porous = bc.RelaxationZone(shape=shape, zone_type='porous', center=np.array([0., 0., 0.]),
                               orientation=np.array([1., 0., 0.]), epsFact_porous=1.)
    gen = bc.RelaxationZoneWaveGenerator({2: absorb, 4: porous}, nd=2)
    nE, nk = 5, 3
    rng = np.random.RandomState(0)
    qx = rng.rand(nE, nk, 3)
    mtypes = np.array([0, 2, 4, 1, 2], dtype=np.intc)             # 0 and 1 have no zone, 3 is not in the array

    class Coefficients:
        q_phi = np.zeros((nE, nk))
        q_phi_porous = np.zeros((nE, nk))
        q_velocity_porous = np.zeros((nE, nk, 3))

    class Mesh:
        elementMaterialTypes = mtypes

    class TimeIntegration:
        t = 0.

    class LevelModel:
        coefficients = Coefficients()
        mesh = Mesh()
        timeIntegration = TimeIntegration()
        q = {'x': qx}

    class Model:
        levelModelList = [LevelModel()]

    gen.attachModel(Model(), None)
    gen.attachAuxiliaryVariables({})
    gen.calculate_init()
    gen.calculate()
    m = Model.levelModelList[0]
    # absorption cells: phi is the projection of (centre - x) on the orientation = -x[0]; the porous zone: -10000
    for eN, mt in enumerate(mtypes):
        for k in range(nk):
            if mt == 2:
                assert m.coefficients.q_phi_porous[eN, k] == pytest.approx(-qx[eN, k, 0])
            elif mt == 4:
                assert m.coefficients.q_phi_porous[eN, k] == -10000.
            else:
                assert m.coefficients.q_phi_porous[eN, k] == 0.
    npt.assert_allclose(m.coefficients.q_velocity_porous, 0.)
    assert np.shares_memory(np.asarray(m.q['phi_porous']), m.coefficients.q_phi_porous)
    assert np.shares_memory(np.asarray(m.q['velocity_porous']), m.coefficients.q_velocity_porous)
