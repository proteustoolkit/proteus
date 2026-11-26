"""LaggedSGEState (SubgridError): the lagged subgrid errors carry tau_last and their own copies of the *_sge quadrature
arrays through state()/set_state(), and nothing when they do not lag. The end-to-end measurement of the
AdvectionDiffusionReaction_ASGS case is test_state_oracle.py (ladr_2d_sglag); this is the round trip, class by class.
"""
import numpy
import pytest

from proteus import SubgridError

CLASSES = [SubgridError.AdvectionDiffusionReaction_ASGS,
           SubgridError.HamiltonJacobi_ASGS,
           SubgridError.HamiltonJacobiDiffusionReaction_ASGS]


class _Coefficients(object):
    nc = 1
    sd = False
    diffusion = {0: {0: {0: None}}}


def _cq():
    shape = (5, 4)
    cq = {('u', 0): numpy.zeros(shape)}
    for name in ('df', 'dm', 'dmt', 'dH', 'dphi', 'da'):
        key = (name, 0, 0) if name not in ('dphi',) else (name, 0, 0)
        cq[key] = numpy.zeros(shape)
    cq[('da', 0, 0, 0)] = numpy.zeros(shape)
    cq[('dphi', 0, 0)] = numpy.zeros(shape)
    cq[('grad(phi)', 0)] = numpy.zeros(shape + (2,))
    return cq


def _make(cls, lag):
    sge = cls(_Coefficients(), 2, lag=lag)
    cq = _cq()
    sge.initializeElementQuadrature(None, 0.0, cq)
    return sge, cq


def _history(sge, cq, value):
    for key, a in cq.items():
        if not key[0].endswith("_sge"):
            a[...] = value
    for ci, tau in enumerate(sge.tau if isinstance(sge.tau, list) else [sge.tau[0]]):
        tau[...] = value + 0.5
    sge.updateSubgridErrorHistory()


@pytest.mark.parametrize("cls", CLASSES)
def test_unlagged_carries_nothing(cls):
    sge, cq = _make(cls, False)
    assert sge.state() is None
    assert sge.set_state(None) is None


@pytest.mark.parametrize("cls", CLASSES)
def test_lagged_round_trip(cls):
    sge, cq = _make(cls, True)
    _history(sge, cq, 3.0)
    state = sge.state()
    assert state
    snapshot = {k: numpy.array(v, copy=True) for k, v in sge._lagged_arrays().items()}
    fresh, cq2 = _make(cls, True)
    assert any(not numpy.array_equal(snapshot[k], v) for k, v in fresh._lagged_arrays().items())
    fresh.set_state(state)
    for k, v in fresh._lagged_arrays().items():
        assert numpy.array_equal(v, snapshot[k]), k
    # in place: the arrays other code holds (cq) are the ones restored
    key = [k for k in cq2 if k[0] == 'df_sge' or k[0] == 'dH_sge'][0]
    assert numpy.array_equal(cq2[key], snapshot["sge_%r" % (key,)])
    # the state is a copy, not a view of the live arrays
    for a in state.values():
        assert not any(numpy.shares_memory(a, b) for b in fresh._lagged_arrays().values())
