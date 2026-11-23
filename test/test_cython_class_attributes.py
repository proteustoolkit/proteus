"""No class stores a compiled (Cython) function as a plain class attribute (C47).

Under Cython 3 (binding=True is the default) a ``def`` function in a .pyx module
is a ``cython_function_or_method`` with a ``__get__``: stored on a class and
called as ``self.f(...)`` it receives ``self`` as an extra first argument, like
a Python function, and raises ``TypeError: f() takes exactly N positional
arguments (N+1 given)``. Proteus stores such functions on classes with a
class-body ``from .cX import f``; each one is wrapped as ``staticmethod(f)``.

DEFERRED lists the ones not yet wrapped (none). The test fails when the set
found differs from DEFERRED in either direction.
"""
import importlib
import inspect

import pytest

BINDING = ("cython_function_or_method", "fused_cython_function")

# modules that store compiled functions on classes (a scan of every proteus
# module that imports, 2026-11-05)
MODULES = [
    "proteus.kernels.ADR",
    "proteus.AuxiliaryVariables",
    "proteus.PostProcessingTools",
    "proteus.SubsurfaceTransportCoefficients",
    "proteus.TimeIntegration",
    "proteus.TransportCoefficients",
    "proteus.TwophaseDarcyCoefficients",
    "proteus.kernels.m_comp_co2",
    "proteus.kernels.Richards",
    "proteus.kernels.Dissipation",
    "proteus.kernels.Kappa",
    "proteus.kernels.MCorr",
    "proteus.kernels.MCorr3P",
    "proteus.kernels.NCLS",
    "proteus.kernels.NCLS3P",
    "proteus.kernels.RANS2P",
    "proteus.kernels.RANS3PF",
    "proteus.kernels.RANS3PSed",
    "proteus.kernels.RDLS",
    "proteus.kernels.TADR",
    "proteus.kernels.VOF",
    "proteus.kernels.VOF3P",
    "proteus.kernels.VOS3P",
]

# none: every site is wrapped
DEFERRED = {}

def _binding_class_attributes(module):
    found = set()
    for class_name, cls in vars(module).items():
        if not inspect.isclass(cls) or cls.__module__ != module.__name__:
            continue
        for name, value in vars(cls).items():
            if type(value).__name__ in BINDING:
                found.add(f"{class_name}.{name}")
    return found


def test_compiled_functions_bind_self():
    """The premise: a compiled function stored on a class binds self, and
    staticmethod stops it."""
    from proteus import ctransportCoefficients
    f = ctransportCoefficients.groundwaterTransportCoefficientsEvaluate_hetMat
    if type(f).__name__ not in BINDING:
        pytest.skip("compiled functions do not bind here (Cython binding=False)")

    class Plain:
        g = f

    class Wrapped:
        g = staticmethod(f)

    assert Plain().g is not f and Plain().g.__self__.__class__ is Plain
    assert Wrapped().g is f


@pytest.mark.parametrize("module_name", MODULES)
def test_no_binding_compiled_class_attributes(module_name):
    module = importlib.import_module(module_name)
    assert _binding_class_attributes(module) == DEFERRED.get(module_name, set())
