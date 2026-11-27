"""Every shock capturing that runs the delayed-lag machine (starts with numDiff_last aliased to numDiff, switches to
lagging after nStepsToDelay history updates) carries it through state()/set_state() (ShockCapturing.DelayedLagState).

A hot start rebuilds such an object from the restart state: nSteps starts again, numDiff_last is the alias again, so
step n+1 is solved unlagged where the uninterrupted run lagged (C14 note, section 3.2 rows 1-6; measured on damBreakLevelSet
with turbulence=1: Dissipation and Kappa, dofs 7.5e-10 and 2.5e-10 off one interval after the restart, numDiff 4e-7).
The unit test is the machine and its round trip, class by class, before and after the switch; the end-to-end
measurement is test_state_oracle.py.
"""
import importlib
import inspect
import pkgutil

import numpy
import pytest

import proteus.kernels
from proteus.ShockCapturing import DelayedLagState


def _classes():
    found = []
    for m in pkgutil.iter_modules(proteus.kernels.__path__):
        if m.name.startswith("c"):  # the compiled kernels
            continue
        mod = importlib.import_module("proteus.kernels." + m.name)
        cls = getattr(mod, "ShockCapturing", None)
        # RANS2P's has its own variant of the machine (counts only when lag is requested, components 1-3); it is
        # measured exact end to end by test_hotstart_levelset.py
        if inspect.isclass(cls) and issubclass(cls, DelayedLagState) and m.name != "RANS2P":
            found.append((m.name, cls))
    return found


CLASSES = _classes()


class _Coefficients(object):
    nc = 4  # RANS2P keeps four components


def _fresh(cls):
    cq = {}
    for ci in range(_Coefficients.nc):
        cq[('numDiff', ci, ci)] = numpy.zeros((5, 4))
        cq[('u', ci)] = numpy.zeros((5, 4))
    sc = cls(_Coefficients(), 2)
    sc.initializeElementQuadrature(None, 0.0, cq)
    return sc, cq


def _step(sc, cq, value):
    for ci in range(_Coefficients.nc):
        cq[('numDiff', ci, ci)][:] = value + ci
    sc.updateShockCapturingHistory()


def _last(sc):
    last = sc.numDiff_last
    return [last[ci] for ci in (sorted(last) if isinstance(last, dict) else range(len(last)))]


def _numdiff(sc):
    return [sc.numDiff[ci] for ci in range(len(sc.numDiff))]


def test_the_classes_are_found():
    names = {name for name, _ in CLASSES}
    assert {"NCLS", "VOF", "RANS3PF", "NCLS3P", "VOF3P",
            "Dissipation", "Kappa", "TADR", "VOS3P", "RANS3PSed"} <= names
    # SW2D's is not here yet: kernels/SW2D.py is track B1's (cekees/proteus#358 fixes its imports; its ShockCapturing
    # calls a logEvent the module does not define); the one-line mixin follows when that merges


@pytest.mark.parametrize("name,cls", CLASSES, ids=[n for n, _ in CLASSES])
@pytest.mark.parametrize("steps", [0, 1, 4], ids=["unlagged", "delayed", "lagged"])
def test_the_delayed_lag_machine_round_trips(name, cls, steps):
    a, cqa = _fresh(cls)
    for k in range(steps):
        _step(a, cqa, 10.0 * (k + 1))
    state = a.state()
    assert state["nSteps"] == steps
    b, cqb = _fresh(cls)
    b.set_state(state)
    assert (b.lag, b.nSteps) == (a.lag, a.nSteps)
    # the alias is rebuilt as an alias (its values are the next residual's), the copy as a copy with the values
    for x, y, nx, ny in zip(_last(a), _last(b), _numdiff(a), _numdiff(b)):
        assert (x is nx) == (y is ny)
        if x is not nx:
            numpy.testing.assert_array_equal(x, y)
    # and the next step then goes as the uninterrupted one's does
    for sc, cq in ((a, cqa), (b, cqb)):
        _step(sc, cq, 77.0)
    assert (b.lag, b.nSteps) == (a.lag, a.nSteps)
    for x, y in zip(_last(a), _last(b)):
        numpy.testing.assert_array_equal(x, y)
