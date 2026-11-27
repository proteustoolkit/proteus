"""TADR.Coefficients.state()/set_state(): the coupling lag of a transport model coupled to a flow model.

C14 note 3.5 rows a, c, d. A hot start rebuilds the coupling at attach time from a flow model that has not stepped, and
firstStep is False, so nothing re-seeds it. Measured one output interval after a hot start on the levee deck
(test/salinity/levee_small, density-coupled transport on a Richards flow model; test_state_oracle.py), each owner removed in turn:
  carriedStorage  Compatible (STAB 6): transport dofs 6.5e-4 and flow dofs 4.7e-7 off without it, bitwise with
  q_v             explicit EV with FCT (STAB 4), whose low-order flux lags against q_v_old
  M_dof, rho_nodal  edge-flux STAB 4: the carried mass and the nodal density the flow model's next preStep takes
  (and Richards' rho_nodal_old, ThetaScheme.state: test_richards_theta_state.py), STAB 4 and 5
q_rho, nodal_rho and q_porosity_old were measured not to matter and are not carried.

This is the round trip, with the carried-mass owners of the edge-flux modes.
"""
import types

import numpy as np

from proteus.kernels import TADR


def _coefficients(edge_flux=True, compatible=True):
    rng = np.random.RandomState(7)
    c = TADR.Coefficients.__new__(TADR.Coefficients)
    for name, shape in (("q_v", (6, 4, 2)), ("q_v_old", (6, 4, 2))):
        setattr(c, name, rng.rand(*shape))
    c.model = types.SimpleNamespace(M_dof=rng.rand(9), rho_nodal=rng.rand(9), _use_edge_flux=int(edge_flux),
                                    _M_carried=edge_flux)
    if compatible:
        c.model.carriedStorage = {1.0: rng.rand(9), 2.5: rng.rand(9)}
    return c


def _same(a, b):
    for k in TADR.Coefficients._state_arrays:
        np.testing.assert_array_equal(getattr(a, k), getattr(b, k))
    np.testing.assert_array_equal(a.model.M_dof, b.model.M_dof)
    np.testing.assert_array_equal(a.model.rho_nodal, b.model.rho_nodal)
    assert a.model._M_carried == b.model._M_carried
    assert sorted(a.model.carriedStorage) == sorted(b.model.carriedStorage)
    for t in a.model.carriedStorage:
        np.testing.assert_array_equal(a.model.carriedStorage[t], b.model.carriedStorage[t])


def test_the_coupling_lag_round_trips():
    a = _coefficients()
    state = a.state()
    assert {"q_v", "M_dof", "M_carried", "rho_nodal", "carried_n", "carried_t", "carried_M"} <= set(state)
    b = _coefficients()
    b.q_v[...] = 0.0
    b.model.M_dof[:] = 0.0
    b.model.rho_nodal[:] = 0.0
    b.model._M_carried = False
    b.model.carriedStorage = {}
    b.set_state(state)
    _same(a, b)


def test_the_state_is_a_copy_and_set_state_writes_in_place():
    a = _coefficients()
    state = a.state()
    q_v, rho_nodal = a.q_v, a.model.rho_nodal
    a.q_v[...] = -1.0
    a.model.carriedStorage[1.0][...] = -1.0
    assert state["q_v"].min() >= 0.0 and state["carried_M"].min() >= 0.0
    a.set_state(state)
    assert a.q_v is q_v and a.model.rho_nodal is rho_nodal  # the arrays the kernels were handed stay the same objects
    assert a.q_v.min() >= 0.0


def test_a_scheme_without_a_carried_mass_carries_none():
    a = _coefficients(edge_flux=False, compatible=False)
    state = a.state()
    assert not ({"M_dof", "M_carried", "rho_nodal", "carried_n"} & set(state))
    b = _coefficients(edge_flux=False, compatible=False)
    b.set_state(state)


def test_an_empty_carriedStorage_is_restored_empty():
    """A retried step restores the state of before the step: the carried storage of the Compatible scheme is empty then."""
    a = _coefficients()
    a.model.carriedStorage = {}
    state = a.state()
    assert state["carried_n"] == 0 and "carried_t" not in state
    a.model.carriedStorage = {1.0: np.ones(9)}
    a.set_state(state)
    assert a.model.carriedStorage == {}


def test_no_state_before_the_models_are_attached():
    c = TADR.Coefficients.__new__(TADR.Coefficients)
    assert c.state() is None
