"""The ctimeIntegration wrappers take the arguments TimeIntegration.py passes.

The 2019 Cython rewrite of the C module (#879) gave seven wrappers new signatures (the sizes as leading arguments,
the limiter flag first, psiTCtteDT writing dt into an argument instead of returning it) while TimeIntegration.py kept
calling them the old way, so every 1D DG limiter (DGlimiterP1Lagrange1d, its _Sw and P2/P3 variants, the vacuum
tolerance ones), the Durlofsky limiters and pseudo-transient continuation (PsiTCtte) raised TypeError on their first
call. test/shallow_water/dambreak_1d stopped there.
"""
import ast
import os
import re

import numpy

from proteus import ctimeIntegration

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "proteus")


def _signatures():
    """name -> (number of parameters, number with defaults) for every def in ctimeIntegration.pyx."""
    pyx = open(os.path.join(ROOT, "ctimeIntegration.pyx")).read()
    sigs = {}
    for m in re.finditer(r"^def (\w+)\((.*?)\):", pyx, re.S | re.M):
        params = [p for p in m.group(2).replace("\n", " ").split(",") if p.strip()]
        sigs[m.group(1)] = (len(params), sum("=" in p for p in params))
    return sigs


def test_every_call_from_time_integration_binds():
    sigs = _signatures()
    tree = ast.parse(open(os.path.join(ROOT, "TimeIntegration.py")).read())
    calls, wrong = 0, []
    for call in ast.walk(tree):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr in sigs:
            calls += 1
            nparams, ndefaults = sigs[call.func.attr]
            if not nparams - ndefaults <= len(call.args) <= nparams or call.keywords:
                wrong.append("TimeIntegration.py:%d %s: %d arguments, the wrapper takes %d to %d"
                             % (call.lineno, call.func.attr, len(call.args), nparams - ndefaults, nparams))
    assert calls >= 15
    assert wrong == [], "\n".join(wrong)


def _line(u):
    """Four P1 DG elements on [0, 4] with dof values u (two per element)."""
    nodes = numpy.zeros((5, 3))
    nodes[:, 0] = numpy.arange(5.0)
    elementNodes = numpy.array([[0, 1], [1, 2], [2, 3], [3, 4]], "i")
    neighbours = numpy.array([[1, -1], [2, 0], [3, 1], [-1, 2]], "i")  # across the face opposite local node 0, 1
    barycentres = numpy.zeros((4, 3))
    barycentres[:, 0] = numpy.arange(4) + 0.5
    l2g = numpy.arange(8, dtype="i").reshape(4, 2)
    return elementNodes, neighbours, nodes, barycentres, l2g, numpy.zeros(4, "i"), numpy.array(u, dtype=float)


def test_p1_1d_limiter_keeps_linear_data_and_flattens_extrema():
    for flag in (0, 1, 2):
        *mesh, tag, u = _line([0, 1, 1, 2, 2, 3, 3, 4])
        out = numpy.zeros(8)
        ctimeIntegration.applyDGlimitingP1Lagrange1d(*mesh, tag, u, out, flag)
        assert numpy.array_equal(out, u) and tag.tolist() == [1, 1, 1, 1]
        # element 1 (mean 3) is above both neighbours' means (0.5, 2.5): flattened; element 2 (mean 2.5, slope 1)
        # sees a falling left difference: flattened
        *mesh, tag, u = _line([0, 1, 1, 5, 2, 3, 3, 4])
        ctimeIntegration.applyDGlimitingP1Lagrange1d(*mesh, tag, u, out, flag)
        assert out.tolist() == [0, 1, 3, 3, 2.5, 2.5, 3, 4] and tag.tolist() == [1, 0, 0, 1]
    *mesh, tag, u = _line([0, 1, 1, 5, 2, 3, 3, 4])
    out = numpy.zeros(8)
    ctimeIntegration.applyDGlimitingP1Lagrange1d_withVacuumTol(0, 0.0, *mesh, tag, u, out)
    assert out.tolist() == [0, 1, 3, 3, 2.5, 2.5, 3, 4]


def test_psi_tc_returns_the_step_from_every_point():
    yn, ypn = numpy.zeros((2, 3)), numpy.ones((2, 3))
    ypnm1 = ypn.copy()
    ypnm1[1, 2] -= 0.4  # the largest change at the last of the 6 points
    dt = ctimeIntegration.psiTCtteDT(1.0e-2, 0.1, 0.2, yn, ypn, ypnm1)
    assert dt == numpy.sqrt(1.0e-2 / (0.2 + 1.0e-8)) * 0.9
