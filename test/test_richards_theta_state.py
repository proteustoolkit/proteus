"""Richards.ThetaScheme carries the nodal density of the last accepted step (C14 note 3.5 row e).

updateTimeHistory moves rho_nodal to rho_nodal_old (the next step's old-time mass is theta(psi^n) rho^{n-1}); a hot start
rebuilt both as zeros with use_rho_nodal_old = 0 and Coefficients.preStep, with firstStep False, does not refresh them.
Measured by test_state_oracle.py on levee_small_ev4 and levee_small_ev5 (an edge-based TADR on a Richards flow model):
without it the flow and transport dofs one interval after a hot start differ by 1e-4 to 1e-3, with it they are bitwise.
"""
import types

import numpy as np

from proteus import TimeIntegration
from proteus.kernels import Richards


def _scheme(rho_old, used):
    ti = Richards.ThetaScheme.__new__(Richards.ThetaScheme)
    ti.m_last = {0: np.arange(6.0).reshape(2, 3)}
    ti.integrateInterpolationPoints = False
    ti.m_ip_last = {}
    ti.transport = types.SimpleNamespace(rho_nodal_old=rho_old, use_rho_nodal_old=used)
    return ti


def test_the_nodal_density_round_trips_with_the_mass():
    a = _scheme(np.linspace(1.0, 1.025, 7), 1)
    state = a.state()
    assert {"m_last_0", "rho_nodal_old", "use_rho_nodal_old"} <= set(state)
    b = _scheme(np.zeros(7), 0)
    b.m_last[0][...] = 0.0
    b.set_state(state)
    np.testing.assert_array_equal(b.transport.rho_nodal_old, a.transport.rho_nodal_old)
    assert b.transport.use_rho_nodal_old == 1
    np.testing.assert_array_equal(b.m_last[0], a.m_last[0])


def test_it_is_written_in_place_and_the_state_is_a_copy():
    a = _scheme(np.ones(4), 1)
    state = a.state()
    rho = a.transport.rho_nodal_old
    a.transport.rho_nodal_old[...] = 5.0
    assert state["rho_nodal_old"].max() == 1.0
    a.set_state(state)
    assert a.transport.rho_nodal_old is rho and rho.max() == 1.0


def test_an_archive_without_the_density_leaves_the_scheme_as_it_is():
    a = _scheme(np.full(4, 2.0), 1)
    a.set_state({"m_last_0": np.zeros((2, 3))})
    assert a.transport.use_rho_nodal_old == 1 and a.transport.rho_nodal_old.min() == 2.0
