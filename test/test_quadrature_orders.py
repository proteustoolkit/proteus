"""
Quadrature orders: every available rule integrates the monomials up to its degree, and asking for an
order that does not exist is an error.

Until 2026-11 `Q_base.setOrder` logged a warning and silently used the highest tabulated rule (order 8 on
the triangle and tetrahedron), so a degree-12 request integrated with degree 8. Orders 9..20 on the
triangle and tetrahedron are now the Stroud conical product rules (collapsed Gauss-Jacobi, derived in
`Quadrature.py`, positive weights, interior points), and the 1D Gauss rule goes to 20 points.
"""
import itertools
from math import factorial

import numpy as np
import pytest

from proteus import Quadrature as Q

TOL = 1.0e-13
MAX = Q.MAX_GENERATED_SIMPLEX_ORDER


def exact_simplex(a):
    """int over the unit simplex of x^a = prod(a_i!)/(sum(a)+d)!"""
    num = 1
    for ai in a:
        num *= factorial(ai)
    return num / factorial(sum(a) + len(a))


def monomial_error(points, weights, nd, degree):
    pts = np.array([list(p)[:nd] for p in points])
    w = np.array(weights)
    worst = 0.0
    for a in itertools.product(range(degree + 1), repeat=nd):
        if sum(a) > degree:
            continue
        val = np.sum(w * np.prod(pts ** np.array(a), axis=1))
        worst = max(worst, abs(val - exact_simplex(a)))
    return worst


@pytest.mark.parametrize("nd,cls", [(2, Q.GaussTriangle), (3, Q.GaussTetrahedron)])
def test_every_simplex_rule_is_exact_to_its_order(nd, cls):
    q = cls(1)
    assert q.maxAvailableOrder() == MAX >= 12
    for k in range(1, MAX + 1):
        q.setOrder(k)
        assert q.order == k
        err = monomial_error(q.points, q.weights, nd, k)
        assert err < TOL, "%s order %d: monomial error %g" % (cls.__name__, k, err)
        assert sum(q.weights) == pytest.approx(1.0 / factorial(nd), abs=1.0e-14)


@pytest.mark.parametrize("nd,cls,negative", [(2, Q.GaussTriangle, {3}), (3, Q.GaussTetrahedron, {3, 7, 8})])
def test_weights_are_positive_except_the_known_tabulated_rules(nd, cls, negative):
    """Pin which tabulated rules have negative weights (changing that is a decision, not an accident);
    all generated rules (order >= 9) have positive weights and interior points."""
    q = cls(1)
    neg = set()
    for k in range(1, MAX + 1):
        q.setOrder(k)
        if min(q.weights) < 0.0:
            neg.add(k)
        if k >= 9:
            assert min(q.weights) > 0.0
            pts = np.array([list(p)[:nd] for p in q.points])
            assert (pts > 0.0).all() and (pts.sum(axis=1) < 1.0).all()
    assert neg == negative


def test_gauss_edge_rules_to_20_points():
    q = Q.GaussEdge(1)
    assert q.maxAvailableOrder() == 20
    for n in range(1, 21):
        q.setOrder(n)
        assert len(q.points) == n
        assert min(q.weights) > 0.0
        pts = np.array([p[0] for p in q.points])
        w = np.array(q.weights)
        for d in range(2 * n):  # n points: exact to degree 2n-1
            assert abs(np.sum(w * pts ** d) - 1.0 / (d + 1)) < TOL


def test_generated_rule_sizes():
    """Pins the cost: the order-12 triangle rule has 49 points and the tetrahedron 343."""
    t = Q.GaussTriangle(12)
    assert len(t.points) == 49
    assert len(Q.GaussTetrahedron(12).points) == 343
    assert len(Q.GaussTriangle(9).points) == 25 and len(Q.GaussTetrahedron(9).points) == 125


@pytest.mark.parametrize("cls", [Q.GaussEdge, Q.GaussTriangle, Q.GaussTetrahedron,
                                 Q.LobattoEdge, Q.LobattoEdgeAlt, Q.LobattoTriangle, Q.LobattoTetrahedron])
def test_a_missing_order_is_an_error(cls):
    q = cls(1)
    kmax = q.maxAvailableOrder()
    q.setOrder(kmax)  # the largest one exists
    for bad in (kmax + 1, 0, -1):
        with pytest.raises(ValueError, match="not available"):
            q.setOrder(bad)
        with pytest.raises(ValueError, match="not available"):
            cls(bad)
    assert q.order == kmax  # a refused request changes nothing


@pytest.mark.parametrize("nd", [1, 2, 3])
def test_wrappers_refuse_missing_orders(nd):
    with pytest.raises(ValueError):
        Q.SimplexGaussQuadrature(nd, MAX + 1)
    s = Q.SimplexGaussQuadrature(nd, 4)
    with pytest.raises(ValueError):
        s.setOrder(MAX + 1)
    assert s.quadrature.order == 4
    with pytest.raises(ValueError):
        Q.CubeGaussQuadrature(nd, 21)


@pytest.mark.parametrize("nd", [2, 3])
def test_orders_above_8_are_no_longer_order_8(nd):
    """The old behavior returned the order-8 rule for 9..: the point counts differ now."""
    r8 = Q.SimplexGaussQuadrature(nd, 8)
    r12 = Q.SimplexGaussQuadrature(nd, 12)
    assert len(r12.points) > len(r8.points)
    assert monomial_error(r8.points, r8.weights, nd, 12) > 1.0e-8
    assert monomial_error(r12.points, r12.weights, nd, 12) < TOL


def test_cube_rules_above_9_points_per_direction():
    for nd in (2, 3):
        q = Q.CubeGaussQuadrature(nd, 12)
        assert len(q.points) == 12 ** nd
        assert sum(q.weights) == pytest.approx(2.0 ** nd, abs=1.0e-13)


def test_gauss_jacobi_nodes_are_interior_with_positive_weights():
    for alpha in (0, 1, 2):
        for n in range(1, 11):
            t, w = Q._gauss_jacobi(n, alpha)
            assert (t > 0.0).all() and (t < 1.0).all() and (w > 0.0).all()
            # int_0^1 (1-t)^alpha t^d dt = d! alpha!/(d+alpha+1)!, exact for d <= 2n-1
            for d in range(2 * n):
                assert abs(np.sum(w * t ** d) - factorial(d) * factorial(alpha) / factorial(d + alpha + 1)) < TOL
