"""C49 R4: NCLS (proteus.kernels.NCLS) variants that no test selects, on the rotating-disk problem of this directory.

``test_ncls.py`` runs the 1D advection problem (periodic) with SUPG and entropy viscosity.  Here the 2D rotation of a
disk (problem 1, solid-body rotation at 2 pi per unit time about (0.5, 0.5)), which has an exact solution -- the initial
disk turned by the angle 2 pi t -- is run for the SSP time integrators of order 1, 2 and 3 (RKEV), the smoothness-based
indicator (STABILIZATION_TYPE 2), entropy viscosity with both entropy functions and the lumped mass matrix with its own
explicit solver.  Each run is compared with the exact solution, with the maximum principle the edge-based low-order
schemes keep, and with the other variants (they must differ: a stage that did not run would give the same answer).
"""
import os
from importlib import reload

import numpy as np
import pytest

import proteus
from proteus import Context, Profiling, default_s, default_so
from proteus.NonlinearSolvers import ExplicitLumpedMassMatrix

from . import ncls, ncls_n, ncls_p

T_FINAL = 0.04


def _run(opts, tmp_path, monkeypatch, name, lumped=False, **ct):
    monkeypatch.chdir(tmp_path)
    options = dict(problem=1, unstructured=False, refinement=1, T=T_FINAL, nDTout=1, COUPEZ=False,
                   DO_REDISTANCING=False, level_set_function=0, SATURATED_LEVEL_SET=False, ENTROPY_TYPE=1)
    options.update(ct)
    # the deck reads its options when it is imported: the string parun's -C writes
    Context.contextOptionsString = " ".join("%s=%r" % kv for kv in options.items())
    try:
        reload(ncls)
        reload(ncls_p)          # the numerics deck star-imports the physics deck: reload the physics first
        reload(ncls_n)
    finally:
        Context.contextOptionsString = None
    if lumped:
        ncls_p.coefficients.LUMPED_MASS_MATRIX = True
        ncls_n.levelNonlinearSolver = ExplicitLumpedMassMatrix
    so = default_so
    so.name = name
    so.tnList = ncls_n.tnList
    ns = proteus.NumericalSolution.NS_base(so, [ncls_p], [ncls_n], [default_s], opts)
    ns.calculateSolution(name)
    m = ns.modelList[0].levelModelList[-1]
    x = np.array(m.mesh.nodeArray[:, :2])
    u = np.array(m.u[0].dof)
    return x, u, m


def _exact(x, t):
    c = np.array([0.5, 0.5])
    a = -2. * np.pi * t              # the field is a counter-clockwise rotation: look back by the angle
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    y = (x - c) @ R.T + c
    init = ncls_p.init_cond(None, scaling=1.0)
    return np.array([init.uOfXT(p, 0.) for p in y])


def _initial(x):
    init = ncls_p.init_cond(None, scaling=1.0)
    return np.array([init.uOfXT(p, 0.) for p in x])


VARIANTS = {
    "ssp1": dict(STABILIZATION_TYPE=2, SSPOrder=1),
    "ssp2": dict(STABILIZATION_TYPE=2, SSPOrder=2),
    "ssp3": dict(STABILIZATION_TYPE=2, SSPOrder=3),
    "ev1": dict(STABILIZATION_TYPE=1, SSPOrder=2),
    "ev2_saturated": dict(STABILIZATION_TYPE=1, SSPOrder=3, ENTROPY_TYPE=2, level_set_function=1,
                          SATURATED_LEVEL_SET=True),
}
# relative L2 error against the exact rotated field, measured 2026-11-26 (macOS arm64, 21 x 21 nodes, t = 0.04)
PIN_L2 = {"ssp1": 0.025647, "ssp2": 0.025745, "ssp3": 0.025698, "ev1": 0.018944, "ev2_saturated": 0.089165}
_cache = {}


@pytest.fixture
def run(opts, tmp_path_factory, monkeypatch):
    def go(name):
        if name not in _cache:
            d = tmp_path_factory.mktemp(name)
            kw = dict(VARIANTS[name])
            _cache[name] = _run(opts, d, monkeypatch, "ncls_" + name, **kw)[:2]
        return _cache[name]
    return go


def _l2(x, u):
    e = _exact(x, T_FINAL)
    return np.linalg.norm(u - e) / np.linalg.norm(e)


@pytest.mark.parametrize("name", sorted(PIN_L2))
def test_the_rotating_disk_matches_the_exact_solution_to_the_recorded_error(run, name):
    x, u = run(name)
    assert np.all(np.isfinite(u))
    assert _l2(x, u) == pytest.approx(PIN_L2[name], rel=2e-3)


def test_the_smoothness_indicator_scheme_keeps_the_maximum_principle(run):
    for name in ("ssp1", "ssp2", "ssp3"):
        x, u = run(name)
        u0 = _initial(x)
        assert u.max() <= u0.max() + 1e-12, name
        assert u.min() >= u0.min() - 1e-12, name


def test_every_variant_changes_the_answer(run):
    # the three SSP orders solve the same spatial problem and differ only through their stages: a stage that did not
    # run would give the same field
    names = list(VARIANTS)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert np.abs(run(a)[1] - run(b)[1]).max() > 1e-5, (a, b)


@pytest.mark.xfail(strict=True, raises=AttributeError,
                   reason="NCLS asserts that LUMPED_MASS_MATRIX needs levelNonlinearSolver=ExplicitLumpedMassMatrix and "
                          "STABILIZATION_TYPE 2, but that solver calls model.FCTStep(), which NCLS.LevelModel does not "
                          "have: the lumped-mass option cannot run (found by C49 R4; kernel territory: C0)")
def test_lumped_mass_matrix_with_its_explicit_solver_runs(opts, tmp_path, monkeypatch):
    x, u, m = _run(opts, tmp_path, monkeypatch, "ncls_lumped", STABILIZATION_TYPE=2, SSPOrder=2, lumped=True)
    assert np.all(np.isfinite(u))
