"""The state contract: every owner of history says how a hot start treats it.

A hot start rebuilds, from the restart state, whatever history an object keeps from one step to the
next, unless the object declares it with ``state()`` / ``set_state()`` (TI_base.state describes the
contract; NumericalSolution archives it per rank). A class that keeps history and does not declare
it restarts silently: the run goes on, with a different step sequence. That cost was measured for
only a few classes (the vortex2D hot start measurement, test/hotstart_measure).

``state_contract_inventory.txt`` lists every time integrator, shock capturing, subgrid error and
step controller class and its category. This test keeps the list complete (a new class fails until
someone puts it there, which is the moment to ask what it carries) and honest (a class that defines
``state`` is in the ``state`` category, and the other way round).
"""
import importlib
import inspect
import os
import pkgutil

import pytest

import proteus.mprans
from proteus import ShockCapturing, SplitOperator, StepControl, SubgridError, TimeIntegration

here = os.path.dirname(os.path.abspath(__file__))
inventory_file = os.path.join(here, "state_contract_inventory.txt")

FAMILIES = {"TI": TimeIntegration.TI_base, "SC": ShockCapturing.ShockCapturing_base,
            "SG": SubgridError.SGE_base, "STEP": StepControl.SC_base, "SO": SplitOperator.SO_base}
CATEGORIES = {"state", "rebuilt", "follows", "unmeasured"}


def _modules():
    yield from ("proteus.TimeIntegration", "proteus.ShockCapturing", "proteus.SubgridError",
                "proteus.StepControl", "proteus.SplitOperator")
    for m in pkgutil.iter_modules(proteus.mprans.__path__):
        if not m.name.startswith("c"):  # the compiled kernels are cNCLS, cVOF, ...
            yield "proteus.mprans." + m.name


def _classes():
    """(family, module, class name) -> the class, for the classes defined in the proteus modules."""
    found = {}
    for name in _modules():
        mod = importlib.import_module(name)
        for cls_name, cls in inspect.getmembers(mod, inspect.isclass):
            if cls.__module__ != name:
                continue
            for family, base in FAMILIES.items():
                if issubclass(cls, base) and cls is not base:
                    found[(family, name[len("proteus."):], cls_name)] = cls
    return found


def _inventory():
    rows = {}
    for line in open(inventory_file):
        line = line.strip()
        if line and not line.startswith("#"):
            family, module, cls, category = line.split()
            assert (family, module, cls) not in rows, "listed twice: %s" % line
            rows[(family, module, cls)] = category
    return rows


def test_every_class_is_in_the_inventory_and_every_entry_is_a_class():
    classes, rows = set(_classes()), _inventory()
    assert not classes - set(rows), \
        "not in test/state_contract_inventory.txt (what does it carry from one step to the next? " \
        "declare it with state()/set_state() or list it, unmeasured until measured): %s" % sorted(classes - set(rows))
    assert not set(rows) - classes, "listed but no longer a class: %s" % sorted(set(rows) - classes)
    assert {c for c in rows.values()} <= CATEGORIES


def test_the_state_category_is_the_classes_that_define_state():
    classes, rows = _classes(), _inventory()
    for key, cls in classes.items():
        base = FAMILIES[key[0]]
        defines = cls.state is not base.state
        assert defines == (rows[key] == "state"), \
            "%s: %s state() but the inventory says %s" % (key, "defines" if defines else "does not define", rows[key])
        if defines:
            assert cls.set_state is not base.set_state, "%s defines state() without set_state()" % (key,)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_the_base_class_carries_nothing(family):
    """The contract's default: a class that says nothing has nothing to restore, so the archive writes
    and restores nothing for it (and the inventory is where that is questioned)."""
    base = FAMILIES[family]
    obj = object.__new__(base)
    assert obj.state() is None
    assert obj.set_state({}) is None


def test_the_pressure_model_declares_its_state_through_the_coefficients_owner():
    """The lagged pressure and sharp pressure live on mprans.Pres.Coefficients, not on a shock
    capturing, subgrid error or time integration object; the archive reads them as the
    ``coefficients`` owner."""
    from proteus import NumericalSolution
    from proteus.mprans import Pres
    assert "coefficients" in NumericalSolution.NS_base.SOLVER_STATE_OWNERS
    assert "state" in vars(Pres.Coefficients) and "set_state" in vars(Pres.Coefficients)
    assert Pres.Coefficients().state() is None  # not attached to a model: nothing to carry
