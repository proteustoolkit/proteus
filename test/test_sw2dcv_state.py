"""SW2DCV RKEV carries the dry-state tolerance hEps through a hot start (C14 note 3.5 letter j; measured with the
state-difference oracle on circular_damBreak: hEps is eps * max(h) of the initial condition, a hot start builds it
from the archive's depth, which moved every dry-state decision of the limiter, 1e-4 after one output interval)."""
import types

import numpy as np

from proteus.kernels import SW2DCV


def _rkev(hEps):
    ti = object.__new__(SW2DCV.RKEV)
    ti.nc = 3
    ti.lstage = 0
    ti.u_dof_last = {ci: np.zeros(4) for ci in range(3)}
    ti.u_dof_lstage = {ci: np.zeros(4) for ci in range(3)}
    ti.transport = types.SimpleNamespace(hEps=hEps, dLow=np.zeros(4), h_dof_old=np.zeros(4))
    return ti


def test_heps_round_trips():
    a, b = _rkev(1.0e-5 * 2.0), _rkev(1.0e-5 * 1.5)
    b.set_state(a.state())
    assert b.transport.hEps == a.transport.hEps


def test_no_heps_before_the_data_structures_exist():
    a = _rkev(None)
    assert "transport_hEps" not in a.state()
    b = _rkev(3.0)
    b.set_state(a.state())
    assert b.transport.hEps == 3.0
