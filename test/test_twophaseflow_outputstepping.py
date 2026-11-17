"""OutputStepping of TwoPhaseFlowProblem: the output times of the run."""
import pytest

from proteus.TwoPhaseFlow.TwoPhaseFlowProblem import OutputStepping


def _stepping(**kw):
    o = OutputStepping()
    for k, v in kw.items():
        setattr(o, k, v)
    o.setOutputStepping()
    return o


def test_regular_run_ends_at_final_time():
    o = _stepping(final_time=1.0, dt_output=0.25, dt_init=0.001)
    assert o.nDTout == 4
    assert o.buildTnList() == [0., 0.001, 0.25, 0.5, 0.75, 1.0]


def test_final_time_below_half_dt_output_still_reaches_final_time():
    """final_time < dt_output/2 used to give nDTout = 0, tnList = [0, dt_init]:
    the run stopped after its first step, at t = dt_init, and reported
    normal completion (wavesloshing -C T=0.02 with dt_output = 0.05)."""
    o = _stepping(final_time=0.02, dt_output=0.05, dt_init=0.001)
    assert o.nDTout == 1
    tn = o.buildTnList()
    assert tn[-1] == pytest.approx(0.02)
    assert tn == sorted(tn) and len(tn) == 3


def test_final_time_below_dt_init_is_monotone():
    o = _stepping(final_time=0.0005, dt_output=0.05, dt_init=0.001)
    tn = o.buildTnList()
    assert tn[-1] == pytest.approx(0.0005)
    assert tn == sorted(tn) and len(set(tn)) == len(tn)


def test_nDTout_without_dt_output():
    """A deck that sets only nDTout used to raise NameError (nDTout was not
    self.nDTout) before the intended assertion."""
    o = _stepping(final_time=2.0, nDTout=4, dt_init=0.001)
    assert o.dt_output == pytest.approx(0.5)
    assert o.buildTnList() == [0., 0.001, 0.5, 1.0, 1.5, 2.0]


def test_neither_dt_output_nor_nDTout_asserts():
    o = OutputStepping()
    o.final_time = 1.0
    with pytest.raises(AssertionError, match="dt_output or nDTout"):
        o.setOutputStepping()


def test_unattached_auxiliary_variables_are_reported():
    """Gauges attached to a shape with attachPointGauges only land in
    domain.auxiliaryVariables; a deck that does not append them to a model
    still runs and writes no gauge file (air-water-vv 2d/benchmarks/wavesloshing).
    """
    from types import SimpleNamespace
    from proteus.TwoPhaseFlow.TwoPhaseFlowProblem import unattachedAuxiliaryVariables
    g1, g2, zone = object(), object(), object()
    domain_aux = {'twp': [g1, zone], 'ls': [g2], 'vof': [], 'SWFlow': [object()]}
    flow = SimpleNamespace(auxiliaryVariables=[g1])
    ncls = SimpleNamespace(auxiliaryVariables=[])
    assert unattachedAuxiliaryVariables(domain_aux, [flow, ncls]) == {'twp': [zone], 'ls': [g2]}
    ncls.auxiliaryVariables += domain_aux['ls']
    flow.auxiliaryVariables += domain_aux['twp']
    assert unattachedAuxiliaryVariables(domain_aux, [flow, ncls]) == {}
    # a domain that did not use SpatialTools has no such dict
    assert unattachedAuxiliaryVariables(None, [flow]) == {}
