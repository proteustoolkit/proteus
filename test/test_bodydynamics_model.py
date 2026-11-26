"""C49 R4: proteus.mbd.BodyDynamics driven the way the simulation drives it, through a model that hands the body its
forces and moments.

test_bodydynamics.py tests the pieces; here a body is attached to a stub model (the arrays `netForces_p`,
`netForces_v`, `netMoments`, the step controller's time) and `calculate()` is run, so the force sum, the constraints,
the substepped integration, the spring-damper Runge-Kutta scheme, imposed motion, the CSV record and the Caisson and
Paddle variants run in sequence. Every result is checked against the closed-form motion of the same body: free fall
with the substepped forward-Euler recurrence, the undamped oscillator for the springs, hydrostatic equilibrium.
"""
import csv
import os

import numpy as np
import numpy.testing as npt
import pytest

from proteus import Comm, Profiling
from proteus.Domain import PlanarStraightLineGraphDomain, PiecewiseLinearComplexDomain
from proteus.SpatialTools import Cuboid, Rectangle
from proteus.mbd import BodyDynamics as bd

Comm.init()
Profiling.procID = 0


class _Coefficients:
    def __init__(self, n):
        self.netForces_p = np.zeros((n, 3))
        self.netForces_v = np.zeros((n, 3))
        self.netMoments = np.zeros((n, 3))


class _LevelModel:
    def __init__(self, nd, n, dt):
        self.nSpace_global = nd
        self.dt_last = dt
        self.coefficients = _Coefficients(n)

        class Mesh:
            elementBoundaryUserFlags = np.arange(n)
        self.mesh = Mesh()


class _StepController:
    t_model_last = 0.


class _Model:
    def __init__(self, nd, n, dt):
        self.levelModelList = [_LevelModel(nd, n, dt)]
        self.stepController = _StepController()


def _attach(body, dt, n=12):
    model = _Model(body.Shape.Domain.nd, n, dt)
    body.attachModel(model, None)
    body.i_start, body.i_end = 1, 1 + len(body.Shape.BC_list)
    body.calculate_init()
    return model


def _body_2d(mass=5., dim=(0.4, 0.2), coords=(1., 2.), cls=bd.RigidBody, **kw):
    d = PlanarStraightLineGraphDomain()
    shape = Rectangle(d, dim=dim, coords=coords)
    body = cls(shape, **kw)
    body.setMass(mass)
    return body


def _advance(model, dt):
    model.stepController.t_model_last += dt


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Profiling, 'logDir', str(tmp_path), raising=False)


# ---------------------------------------------------------------------------------------------------------------------
# the force sum and free motion
# ---------------------------------------------------------------------------------------------------------------------

def test_attachModel_counts_the_boundary_flags_and_forces_are_the_sums_over_the_body_range():
    body = _body_2d()
    model = _attach(body, 0.01)
    assert body.nd == 2 and body.nForces == 12
    c = model.levelModelList[-1].coefficients
    c.netForces_p[1:5] = [[1., 2., 0.]] * 4
    c.netForces_v[1:5] = [[0.5, -1., 0.]] * 4
    c.netMoments[1:5] = [[0., 0., 0.25]] * 4
    c.netForces_p[0] = [100., 100., 0.]          # not a boundary of this body: i_start = 1
    c.netForces_p[5] = [100., 100., 0.]          # beyond i_end
    npt.assert_allclose(body.getPressureForces(), c.netForces_p[1:5])
    npt.assert_allclose(body.getShearForces(), c.netForces_v[1:5])
    npt.assert_allclose(body.getTotalMoments(), [0., 0., 1.])
    npt.assert_allclose(body.getGravityForce(), [0., -5. * 9.81, 0.])
    npt.assert_allclose(body.getTotalForces(), [4 * 1.5, 4 * 1. - 5. * 9.81, 0.])


def test_gravity_force_in_3D_is_along_z():
    d = PiecewiseLinearComplexDomain()
    shape = Cuboid(d, dim=(1., 1., 1.), coords=(0., 0., 0.))
    body = bd.RigidBody(shape)
    body.setMass(3.)
    _attach(body, 0.01)
    npt.assert_allclose(body.getGravityForce(), [0., 0., -3. * 9.81])


@pytest.mark.parametrize('substeps', [1, 20])
def test_free_fall_follows_the_substepped_forward_euler_recurrence(substeps):
    dt, mass = 0.05, 5.
    body = _body_2d(mass=mass, substeps=substeps)
    model = _attach(body, dt)
    y0, x0 = body.Shape.barycenter[1], body.Shape.barycenter[0]
    _advance(model, dt)
    body.calculate()
    a = -9.81
    ds = dt / substeps
    # v_i = a * i * ds, h = sum v_i ds
    expected_h = a * ds ** 2 * substeps * (substeps + 1) / 2.
    npt.assert_allclose(body.h, [0., expected_h, 0.], atol=1e-14)
    npt.assert_allclose(body.velocity, [0., a * dt, 0.], atol=1e-13)
    # the shape moved by h, the body's position and barycenter follow
    npt.assert_allclose(body.Shape.barycenter, [x0, y0 + expected_h, 0.], atol=1e-14)
    npt.assert_allclose(body.position, body.Shape.barycenter)
    npt.assert_allclose(body.last_position, [x0, y0, 0.])
    npt.assert_allclose(body.rotation_matrix, np.eye(3))
    # a second step starts from the stored values
    _advance(model, dt)
    body.calculate()
    npt.assert_allclose(body.last_velocity, [0., a * dt, 0.], atol=1e-13)
    npt.assert_allclose(body.velocity, [0., 2 * a * dt, 0.], atol=1e-13)


def test_hydrostatic_equilibrium_leaves_the_body_at_rest():
    dt, mass = 0.05, 5.
    body = _body_2d(mass=mass)
    model = _attach(body, dt)
    # the pressure on the bottom boundary carries exactly the weight
    model.levelModelList[-1].coefficients.netForces_p[1] = [0., mass * 9.81, 0.]
    p0 = body.Shape.barycenter.copy()
    _advance(model, dt)
    body.calculate()
    npt.assert_allclose(body.h, 0., atol=1e-13)
    npt.assert_allclose(body.Shape.barycenter, p0, atol=1e-13)
    npt.assert_allclose(body.velocity, 0., atol=1e-13)


def test_constraints_remove_the_forces_of_the_fixed_directions():
    dt, mass = 0.05, 4.
    body = _body_2d(mass=mass)
    body.setConstraints(free_x=(1, 0, 0), free_r=(0, 0, 0))
    model = _attach(body, dt)
    c = model.levelModelList[-1].coefficients
    c.netForces_p[1] = [8., 0., 0.]
    c.netMoments[1] = [0., 0., 3.]
    _advance(model, dt)
    body.calculate()
    npt.assert_allclose(body.F, [8., 0., 0.])           # gravity and nothing else in y are removed
    npt.assert_allclose(body.M, 0.)
    assert body.h[1] == 0. and body.h[0] > 0.
    assert body.ang_disp[2] == 0.


def test_a_moment_rotates_the_shape_about_its_barycenter_by_the_euler_recurrence():
    dt, mass, substeps = 0.05, 5., 20
    dim = (0.4, 0.2)
    body = _body_2d(mass=mass, dim=dim, substeps=substeps)
    body.setConstraints(free_x=(0, 0, 0), free_r=(0, 0, 1))
    model = _attach(body, dt)
    Mz = 0.7
    model.levelModelList[-1].coefficients.netMoments[2] = [0., 0., Mz]
    bary0 = body.Shape.barycenter.copy()
    verts0 = body.Shape.vertices.copy()
    _advance(model, dt)
    body.calculate()
    inertia = (dim[0] ** 2 + dim[1] ** 2) / 12. * mass
    alpha = Mz / inertia
    ds = dt / substeps
    theta = alpha * ds ** 2 * substeps * (substeps + 1) / 2.
    npt.assert_allclose(body.ang_acc[2], alpha)
    npt.assert_allclose(body.inertia, inertia)
    npt.assert_allclose(body.ang_disp[2], theta, rtol=1e-12)
    npt.assert_allclose(body.ang_vel[2], alpha * dt, rtol=1e-12)
    # the shape turned about its barycenter by that angle: distances preserved, euler angle = theta
    npt.assert_allclose(body.Shape.barycenter, bary0, atol=1e-13)
    npt.assert_allclose(np.linalg.norm(body.Shape.vertices - bary0[:2], axis=1),
                        np.linalg.norm(verts0 - bary0[:2], axis=1), atol=1e-13)
    npt.assert_allclose(body.rotation_euler[2], theta, rtol=1e-12)
    # the rotation of this step is the stored rotation inverse times the new one: a rotation by theta
    npt.assert_allclose(abs(np.arctan2(body.rotation_matrix[0, 1], body.rotation_matrix[0, 0])), theta, rtol=1e-12)
    # a negative moment turns it the other way
    model.levelModelList[-1].coefficients.netMoments[2] = [0., 0., -Mz]
    _advance(model, dt)
    body.calculate()
    assert body.ang_vel[2] < alpha * dt


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="RigidBody.setInertiaTensor converts It with np.array() and then asserts "
                          "isinstance(It, float), which is never true, so a 2D body's inertia tensor cannot be set "
                          "(found by C49 R4; kernel territory: C0)")
def test_setInertiaTensor_of_a_2D_body_takes_a_float():
    body = _body_2d()
    body.setInertiaTensor(0.5)
    assert body.It == 0.5


def test_setInertiaTensor_in_3D_needs_a_3x3_array():
    d = PiecewiseLinearComplexDomain()
    body = bd.RigidBody(Cuboid(d, dim=(1., 1., 1.), coords=(0., 0., 0.)))
    body.setMass(2.)
    _attach(body, 0.01)
    It = np.diag([1., 2., 3.])
    body.setInertiaTensor(It)
    npt.assert_allclose(body.It, It)
    with pytest.raises(AssertionError):
        body.setInertiaTensor([1., 2., 3.])
    # moment of inertia about an axis: m * v^T It v for a unit vector in the shape's frame
    npt.assert_allclose(body.getInertia(vec=(0., 1., 0.)), 2. * 2.)
    npt.assert_allclose(body.getInertia(vec=(1., 1., 0.)), 2. * (1. + 2.) / 2.)


# ---------------------------------------------------------------------------------------------------------------------
# springs and dampers with the Runge-Kutta scheme
# ---------------------------------------------------------------------------------------------------------------------

def test_runge_kutta_spring_follows_the_undamped_oscillator_in_x_and_y():
    dt, mass = 0.02, 2.
    Kx, Ky, F0 = 50., 800., 3.
    body = _body_2d(mass=mass)
    body.setNumericalScheme('Runge_Kutta')
    body.setSprings(True, Kx=Kx, Ky=Ky, Krot=10., Cx=0., Cy=0., Crot=0.)
    model = _attach(body, dt)
    model.levelModelList[-1].coefficients.netForces_p[1] = [F0, 0., 0.]
    x0, y0 = body.Shape.barycenter[:2].copy()
    Fy = -mass * 9.81
    wx, wy = np.sqrt(Kx / mass), np.sqrt(Ky / mass)
    for n in (1, 2, 3):
        _advance(model, dt)
        body.calculate()
        t = n * dt
        x_exact = F0 / Kx * (1. - np.cos(wx * t))
        y_exact = Fy / Ky * (1. - np.cos(wy * t))
        npt.assert_allclose(body.Shape.barycenter[0] - x0, x_exact, rtol=1e-6, atol=1e-12)
        npt.assert_allclose(body.Shape.barycenter[1] - y0, y_exact, rtol=1e-6, atol=1e-12)
        npt.assert_allclose(body.velocity[0], F0 / Kx * wx * np.sin(wx * t), rtol=1e-6)
    # no scheme-dependent z motion in 2D
    assert body.h[2] == 0.


def test_runge_kutta_damped_spring_loses_energy():
    dt, mass = 0.02, 2.
    body = _body_2d(mass=mass)
    body.setNumericalScheme('Runge_Kutta')
    body.setSprings(True, Kx=50., Ky=800., Krot=10., Cx=4., Cy=0., Crot=0.)
    model = _attach(body, dt)
    model.levelModelList[-1].coefficients.netForces_p[1] = [3., 0., 0.]
    # underdamped: x(t) = F/K (1 - e^{-zeta w t} (cos wd t + zeta w / wd sin wd t))
    K, C, F0 = 50., 4., 3.
    w = np.sqrt(K / mass)
    zeta = C / (2. * mass * w)
    wd = w * np.sqrt(1. - zeta ** 2)
    x0 = body.Shape.barycenter[0]
    for n in range(1, 6):
        _advance(model, dt)
        body.calculate()
    t = 5 * dt
    exact = F0 / K * (1. - np.exp(-zeta * w * t) * (np.cos(wd * t) + zeta * w / wd * np.sin(wd * t)))
    npt.assert_allclose(body.Shape.barycenter[0] - x0, exact, rtol=1e-6)


def test_runge_kutta_rotational_spring_follows_the_undamped_oscillator():
    dt, mass, Krot, M0 = 0.02, 2., 20., 0.4
    dim = (0.4, 0.2)
    body = _body_2d(mass=mass, dim=dim)
    body.setNumericalScheme('Runge_Kutta')
    body.setSprings(True, Kx=0., Ky=0., Krot=Krot, Cx=0., Cy=0., Crot=0.)
    body.setConstraints(free_x=(0, 0, 0), free_r=(0, 0, 1))
    body.setPivot(body.Shape.barycenter.copy())
    model = _attach(body, dt)
    model.levelModelList[-1].coefficients.netMoments[1] = [0., 0., M0]
    inertia = (dim[0] ** 2 + dim[1] ** 2) / 12. * mass
    w = np.sqrt(Krot / inertia)
    for n in (1, 2, 3):
        _advance(model, dt)
        body.calculate()
        exact = M0 / Krot * (1. - np.cos(w * n * dt))
        npt.assert_allclose(body.rotation_euler[2], exact, rtol=1e-5, atol=1e-12)


# ---------------------------------------------------------------------------------------------------------------------
# imposed motion
# ---------------------------------------------------------------------------------------------------------------------

def test_imposed_sinusoidal_motion_brings_the_body_to_the_prescribed_position():
    dt = 0.05
    body = _body_2d(mass=5.)
    body.inputMotion(InputMotion=True, At=[0.2, 0.1, 0.], Tt=[2., 1., 0.], Ar=[0., 0., 0.3], Tr=[0., 0., 1.5])
    npt.assert_allclose(body.pivot, body.Shape.barycenter)       # default pivot
    model = _attach(body, dt)
    start = body.Shape.barycenter.copy()
    for step in range(1, 4):
        _advance(model, dt)
        body.calculate()
        t = model.stepController.t_model_last
        # RigidBody.imposeSinusoidalMotion uses the period through 2*3.14 (see the next test)
        target = start + np.array([0.2 * np.sin(2. * 3.14 / 2. * t), 0.1 * np.sin(2. * 3.14 / 1. * t), 0.])
        npt.assert_allclose(body.Shape.barycenter, target, atol=1e-12)
        npt.assert_allclose(body.rotation_euler[2], 0.3 * np.sin(2. * 3.14 / 1.5 * t), atol=1e-12)


@pytest.mark.xfail(strict=True, reason="RigidBody.imposeSinusoidalMotion uses 2 * 3.14 / T for the angular frequency "
                                       "where PaddleBody.imposeSinusoidalMotion uses 2 * pi / T: the same prescribed "
                                       "period gives a 0.05% different motion (found by C49 R4; C0)")
def test_imposed_sinusoidal_motion_uses_two_pi_over_the_period():
    body = _body_2d(mass=5.)
    body.inputMotion(InputMotion=True, At=[0.2, 0., 0.], Tt=[2., 0., 0.])
    model = _attach(body, 0.05)
    model.stepController.t_model_last = 0.5
    tra, rot = body.imposeSinusoidalMotion()
    npt.assert_allclose(tra[0], 0.2 * np.sin(2. * np.pi / 2. * 0.5), rtol=1e-12)


def test_imposed_motion_with_a_given_pivot_and_zero_periods_does_not_move():
    body = _body_2d(mass=5.)
    body.inputMotion(InputMotion=True, pivot=[3., 3., 0.], At=[0.2, 0., 0.], Tt=[0., 0., 0.])
    npt.assert_allclose(body.pivot, [3., 3., 0.])
    model = _attach(body, 0.05)
    model.stepController.t_model_last = 0.5
    tra, rot = body.imposeSinusoidalMotion()
    npt.assert_allclose(tra, 0.)             # Wt = 0: sin(0) = 0
    npt.assert_allclose(rot, 0.)


# ---------------------------------------------------------------------------------------------------------------------
# the CSV record
# ---------------------------------------------------------------------------------------------------------------------

def test_record_values_write_a_header_at_t0_and_a_row_per_step(tmp_path):
    dt = 0.05
    body = _body_2d(mass=5.)
    body.setRecordValues(filename='rec', all_values=True)
    assert body.record_filename == 'rec.csv'
    assert os.path.dirname(body.record_file) == str(tmp_path)
    model = _attach(body, dt)
    model.levelModelList[-1].coefficients.netForces_p[1] = [1., 0., 0.]
    rows = []
    for _ in range(3):
        _advance(model, dt)
        body.calculate()
        rows.append(dict(F=body.F.copy(), v=body.velocity.copy(), last_pos=body.last_position.copy()))
    with open(body.record_file) as f:
        table = list(csv.reader(f))
    header = table[0]
    assert header[0] == 't' and set(header[1:]) >= {'x', 'y', 'z', 'rx', 'Fx', 'Fy', 'Mz', 'ax', 'vx', 'ang_az',
                                                    'ang_vz'}
    assert len(table) == 1 + 3
    for k, row in enumerate(table[1:]):
        rec = dict(zip(header, map(float, row)))
        assert rec['t'] == pytest.approx(k * dt)
        assert rec['Fx'] == pytest.approx(rows[k]['F'][0])
        assert rec['Fy'] == pytest.approx(rows[k]['F'][1])
        assert rec['vx'] == pytest.approx(rows[k]['v'][0])
        assert rec['x'] == pytest.approx(rows[k]['last_pos'][0])
    # the default name comes from the shape
    body2 = _body_2d()
    body2.setRecordValues(pos=True)
    assert body2.record_filename == 'record_' + body2.Shape.name + '.csv'


def test_caisson_record_values_include_the_elastic_plastic_displacements(tmp_path):
    body = _body_2d(mass=5., cls=bd.CaissonBody, substeps=20)
    body.setRecordValues(filename='caisson', all_values=True)
    assert {'uxElastic', 'uxPlastic', 'uy'} <= set(body.record_dict)
    assert body.record_dict['uxPlastic'] == ['uxPl', None]


# ---------------------------------------------------------------------------------------------------------------------
# Caisson and Paddle bodies
# ---------------------------------------------------------------------------------------------------------------------

def test_caisson_body_steps_like_a_rigid_body_and_tracks_its_vertices():
    dt, mass = 0.05, 50.
    body = _body_2d(mass=mass, dim=(0.3, 0.385), coords=(1., 1.), cls=bd.CaissonBody, substeps=20)
    model = _attach(body, dt)
    npt.assert_allclose(body.cV_init, body.Shape.vertices[:4])
    v_before = body.Shape.vertices.copy()
    _advance(model, dt)
    body.calculate()
    a, n = -9.81, 20
    h = a * (dt / n) ** 2 * n * (n + 1) / 2.
    npt.assert_allclose(body.Shape.vertices, v_before + [0., h], atol=1e-13)
    npt.assert_allclose(body.cV, body.Shape.vertices[:4])
    # the previous vertices are the ones before the step after the next step stores them
    _advance(model, dt)
    body.calculate()
    npt.assert_allclose(body.cV_last, v_before + [0., h], atol=1e-13)
    assert body.sliding_last is False


def test_caisson_body_rotates_on_a_moment_about_the_barycenter_without_the_overturning_module():
    dt = 0.05
    body = _body_2d(mass=50., dim=(0.3, 0.385), coords=(1., 1.), cls=bd.CaissonBody, substeps=20)
    body.setConstraints(free_x=(0, 0, 0), free_r=(0, 0, 1))
    model = _attach(body, dt)
    model.levelModelList[-1].coefficients.netMoments[1] = [0., 0., 5.]
    _advance(model, dt)
    body.calculate()
    assert body.ang_disp[2] > 0.
    npt.assert_allclose(body.rotation_euler[2], body.ang_disp[2], rtol=1e-12)
    # the vertices stayed at the same distance from the barycenter
    d = np.linalg.norm(body.Shape.vertices[:4] - body.Shape.barycenter[:2], axis=1)
    npt.assert_allclose(d, np.hypot(0.15, 0.1925), atol=1e-12)


def test_paddle_body_follows_its_imposed_motion_with_the_pivot_as_centre_of_rotation():
    dt = 0.01
    d = PlanarStraightLineGraphDomain()
    shape = Rectangle(d, dim=(0.3, 1.), coords=(2., 0.1))
    body = bd.PaddleBody(shape, substeps=20)
    body.setMass(10.)
    At, Tt, Ar, Tr = [0.5, 0., 0.], [1., 0., 0.], [0., 0., 0.2], [0., 0., 1.]
    body.inputMotion(InputMotion=True, pivot=[2., 0., 0.], At=At, Tt=Tt, Ar=Ar, Tr=Tr, rampStart=0.05, rampEnd=0.05,
                     Tend=1.)
    # a given pivot becomes the barycenter and the reference position
    npt.assert_allclose(body.Shape.barycenter, [2., 0., 0.])
    npt.assert_allclose(body.init_barycenter, [2., 0., 0.])
    model = _attach(body, dt)
    for n in range(1, 4):
        _advance(model, dt)
        body.calculate()
        t = model.stepController.t_model_last
        ramp = min(1., t / 0.05, (1. - t) / 0.05)
        npt.assert_allclose(body.Shape.barycenter[0], 2. + ramp * 0.5 * np.sin(2. * np.pi * t), atol=1e-12)
        npt.assert_allclose(body.rotation_euler[2], ramp * 0.2 * np.sin(2. * np.pi * t), atol=1e-12)
    # the stored values of the paddle's last step are those of the step before
    assert body.last_pivot is not None
    npt.assert_allclose(body.last_position[:2], [body.position[0] - body.h[0], body.position[1] - body.h[1]],
                        atol=1e-12)
