"""ProtChSystem.state()/set_state(): state, advance, restore, advance again reproduces the first advance bit for bit
(Chrono's StateGather/StateScatter for the integrable state, plus the step counters and the bodies' arrays)."""
import numpy as np
import pytest
from proteus import Comm

pychrono = pytest.importorskip("pychrono")
fsi = pytest.importorskip("proteus.mbd.CouplingFSI")
Comm.init()


def build(spin=True):
    system = fsi.ProtChSystem()
    system.ChSystem.SetGravitationalAcceleration(pychrono.ChVector3d(0., -9.81, 0.))
    system.setTimeStep(1e-4)
    body = fsi.ProtChBody(system=system)
    body.ChBody.SetMass(3.)
    body.ChBody.SetInertiaXX(pychrono.ChVector3d(1., 2., 3.))
    body.ChBody.SetPos(pychrono.ChVector3d(0.1, 1., 0.))
    if spin:
        body.ChBody.SetAngVelLocal(pychrono.ChVector3d(0.3, -0.2, 0.5))
    system.calculate_init()
    return system, body


def advance(system, body, n, dt=0.01):
    for i in range(n):
        body.F_prot = np.array([0.5, 0.1 * i, -0.2])  # a fluid force, as the flow model would set it
        system.calculate(proteus_dt=dt)


def snapshot(system, body):
    b = body.ChBody
    pos, vel, rot = b.GetPos(), b.GetPosDt(), b.GetRot()
    return np.array([system.ChSystem.GetChTime(), pos.x, pos.y, pos.z, vel.x, vel.y, vel.z,
                     rot.e0, rot.e1, rot.e2, rot.e3, *body.acceleration, *body.position_last, *body.rotq_last])


@pytest.mark.parametrize("spin", [False, True])
def test_state_round_trip(spin):
    """Without rotation the restored run is the first run bit for bit. A rotating body is restored to the rounding of
    its unit quaternion (StateScatter rebuilds the rotation from it): ~3e-16 in the quaternion, 1e-14 in the angular
    velocity after four more steps."""
    system, body = build(spin)
    advance(system, body, 3)
    st = system.state()
    advance(system, body, 4)
    first = snapshot(system, body)
    system.set_state(st)
    assert not np.array_equal(first, snapshot(system, body))  # the restore went somewhere else
    advance(system, body, 4)
    again = snapshot(system, body)
    if not spin:
        assert np.array_equal(first, again), np.abs(first - again).max()
    else:
        assert np.array_equal(first[:7], again[:7])  # time, position, velocity: exact
        np.testing.assert_allclose(again, first, rtol=0, atol=1e-12)


def test_state_is_plain_data_and_a_copy():
    system, body = build()
    advance(system, body, 2)
    st = system.state()
    assert all(isinstance(v, (float, np.ndarray)) for v in st.values())
    kept = {k: np.copy(v) for k, v in st.items()}
    advance(system, body, 2)
    assert all(np.array_equal(kept[k], st[k]) for k in kept)  # later steps do not change a kept state


def test_set_state_refuses_another_system():
    system, body = build()
    st = system.state()
    other = fsi.ProtChSystem()
    other.calculate_init()
    with pytest.raises(Exception):
        other.set_state(st)
