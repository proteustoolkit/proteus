"""The surface weight D of order nP = 3 (``gf_s`` of RANS2P/RANS2P2D) against the integrands of P1 and P2 velocity spaces.

Audit item (a) of C0 round 11, measured in round 14 on the triangle of ``test_equivalent_polynomials_moments.py`` (three cuts):
int_Gamma N_a N_b dGamma by ``sum_q w_q D_q N_a N_b`` against an independent 8-point Gauss rule on the cut segment.

  integrand            degree   exact (1e-11) from quadrature degree
  P1 x P1               2        5   (= nP + 2)
  P1 x P2               3        6   (= nP + 3)
  P2 x P2               4        never: 8e-2 .. 2e-1 relative error at every degree 6..9

The moment fit of order nP reproduces integrands up to degree nP; P2 x P2 is degree 4. A P2 velocity with a solid particle
under ``useExact`` therefore has an O(10%) error in the degree-4 surface integrals however fine the quadrature (the Python
binding offers nP <= 3; the kernel's nP is the template argument of ``gf_s``). The test pins the measurement so that a fix
(nP = 4 for gf_s) flips it visibly, and checks the one-time warning RANS2P gives for exactly this combination.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from proteus import equivalent_polynomials as eqp
from proteus.Quadrature import GaussTriangle

import importlib.util
import os

# the triangle, the cuts and the exact segment rule of the round-10 moments test
_spec = importlib.util.spec_from_file_location(
    "test_equivalent_polynomials_moments",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_equivalent_polynomials_moments.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def _p1(l):
    return np.asarray(l)


def _p2(l):
    l0, l1, l2 = l
    return np.array([l0 * (2 * l0 - 1), l1 * (2 * l1 - 1), l2 * (2 * l2 - 1), 4 * l1 * l2, 4 * l0 * l2, 4 * l0 * l1])


def _exact(phi, fa, fb):
    p0, p1 = m._cut_points(phi)
    L = np.linalg.norm(p1 - p0)
    x, w = np.polynomial.legendre.leggauss(8)
    M = 0
    for s, ws in zip(x, w):
        l = m._bary(p0 + 0.5 * (s + 1) * (p1 - p0))
        M = M + 0.5 * ws * L * np.outer(fa(l), fb(l))
    return M


def _quadrature(phi, order, fa, fb, nP=3):
    quad = GaussTriangle(order=order)
    gf = eqp.Simplex(nSpace=2, nP=nP, nQ=len(quad.points))
    gf.calculate(np.array(phi), np.array([[m.V[i, 0], m.V[i, 1], 0.0] for i in range(3)]), np.array(quad.points))
    M = 0
    for k in range(len(quad.points)):
        gf.set_quad(k)
        l = m._bary(m.V[0] + m.JAC @ np.array(quad.points[k][:2]))
        M = M + quad.weights[k] * m.DET * gf.D * np.outer(fa(l), fb(l))
    return M


def _error(phi, order, fa, fb):
    return m._rel(_quadrature(phi, order, fa, fb), _exact(phi, fa, fb))


@pytest.mark.parametrize("cut", sorted(m.CUTS))
@pytest.mark.parametrize("fa,fb,first", [(_p1, _p1, 5), (_p1, _p2, 6)])
def test_the_p1_and_mixed_blocks_are_exact_from_degree_nP_plus_the_integrand_degree(cut, fa, fb, first):
    phi = m.CUTS[cut]
    for order in range(first, 10):
        assert _error(phi, order, fa, fb) < 1.0e-11, (cut, order)
    assert _error(phi, first - 1, fa, fb) > 1.0e-3          # and not one degree below


@pytest.mark.parametrize("cut", sorted(m.CUTS))
def test_the_p2_velocity_block_is_not_exact_at_any_quadrature_degree(cut):
    phi = m.CUTS[cut]
    errors = [_error(phi, order, _p2, _p2) for order in range(6, 10)]
    assert min(errors) > 5.0e-2 and max(errors) < 0.3, errors   # measured 7.8e-2 .. 2.0e-1


def _model(useExact, nParticles, ndof):
    return SimpleNamespace(coefficients=SimpleNamespace(useExact=useExact, nParticles=nParticles),
                           nDOF_trial_element=[3, ndof, ndof], nSpace_global=2)


@pytest.mark.parametrize("useExact,nParticles,ndof,warned", [(True, 1, 6, True), (True, 1, 3, False),
                                                              (False, 1, 6, False), (True, 0, 6, False)])
def test_rans2p_warns_once_for_exact_particles_with_a_quadratic_velocity(monkeypatch, useExact, nParticles, ndof, warned):
    from proteus.mprans import RANS2P
    messages = []
    monkeypatch.setattr(RANS2P, 'logEvent', lambda msg, *a, **k: messages.append(msg))
    model = _model(useExact, nParticles, ndof)
    RANS2P.LevelModel._warnParticleSurfaceDegree(model)
    RANS2P.LevelModel._warnParticleSurfaceDegree(model)
    assert len(messages) == (1 if warned else 0)
