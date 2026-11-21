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


def test_old_1x_api_fails_with_the_replacement_named():
    """1.x decks (TwoPhaseFlowProblem(ns_model=..., outputStepping=...),
    Parameters.Models, physical.densityA) failed with a bare TypeError or
    AttributeError (or nothing at all) and no hint of the 2.0 names."""
    from proteus.TwoPhaseFlow.TwoPhaseFlowProblem import TwoPhaseFlowProblem
    with pytest.raises(TypeError, match="useDefaultModels"):
        TwoPhaseFlowProblem(ns_model=None, ls_model=None, nd=2, cfl=0.9)
    with pytest.raises(TypeError, match="problem.outputStepping"):
        OutputStepping(final_time=1.0, dt_init=0.001)
    problem = TwoPhaseFlowProblem()
    with pytest.raises(AttributeError, match="modelDict"):
        problem.Parameters.Models
    # SystemPhysics and SystemNumerics are not frozen, and test/TwoPhaseFlow/fillingTank.py sets
    # densityA (ignored; its pinned results rest on that): warn, do not raise
    with pytest.warns(UserWarning, match="SystemPhysics.rho_0"):
        problem.SystemPhysics.densityA = 998.
    with pytest.warns(UserWarning, match="SystemNumerics.useSuperlu"):
        problem.SystemNumerics.useSuperLu = True
    with pytest.raises(AttributeError, match="SystemNumerics.cfl"):
        problem.cfl = 0.9  # the problem object is frozen
