"""C44 phase 1b, part A: curved cuts of a tetrahedron by a quadratic (P2) level set (proteus/cutquadric.h).

A1  U0  node order: ten proteus P2 nodal values -> monomial coefficients, against FemTools' own basis
    U1  classification: Bernstein sign test, exact range test, labels (simple / edge cut twice / sub-nodal)
A2  U2  volume moments against closed forms (octant of a ball, ball inside the tet)
    U3  linear u embedded as P2: curved moments equal cutclip's, fast path on and forced off
    U4  nearly planar u across the fast-path threshold
    U5  order sweep: spectral convergence
    U8  partition I(u) + I(-u) = full tet
    U10 July golden values (cutquad.cpp tet_quad_u rebuilt locally on 2026-11-13)
A3  2D face moments and face cut detection
A4  surface rule, scalar and normal-weighted surface moments, divergence identity (U6, U7)
A5  QuadricFit (U9)

The July quadric code is exact on curved cuts. Its Python root finder returns a wrong root for a round-off leading
coefficient on exactly planar 2-2 cuts (1e-5..1e-7, any order); cutquad.cpp does not, because it drops leading
coefficients below 1e-13. cutquadric.h drops below 1e-10 (deflate_tol) and sends planar data to cutclip.
"""
import itertools
import math

import numpy as np
import pytest

from proteus import cutclip, cutquadric

ref_tet = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
# proteus P2 node order on the reference tet (RefUtils.p2refNodes[2]); mid-edge nodes (0,1) (1,2) (2,3) (0,2) (1,3) (0,3)
NODES = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                  [.5, 0, 0], [.5, .5, 0], [0, .5, .5], [0, .5, 0], [.5, 0, .5], [0, 0, .5]])
EDGES = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3)]
IDX2 = cutclip.monomials(2)          # [1, x, y, z, xx, xy, xz, yy, yz, zz]


def mono_eval(coef, pts, idx=IDX2):
    pts = np.atleast_2d(pts)
    return sum(c * pts[:, 0] ** p * pts[:, 1] ** q * pts[:, 2] ** r for c, (p, q, r) in zip(coef, idx))


def nodal(f):
    return np.array([f(*p) for p in NODES])


def linear_p2(phi):
    """P2 nodal values (proteus order) of the P1 function with vertex values phi."""
    lam = lambda x, y, z: (1 - x - y - z, x, y, z)
    return nodal(lambda x, y, z: float(np.dot(phi, lam(x, y, z))))


# ---------------------------------------------------------------------------------------------------- U0
def test_u0_node_order_matches_femtools_basis():
    from proteus import FemTools, RefUtils
    assert np.allclose(RefUtils.p2refNodes[2], NODES)            # the order assumed in this file is proteus's
    space = FemTools.QuadraticOnSimplexWithNodalBasis(3)
    rng = np.random.default_rng(0)
    for _ in range(5):
        u = rng.standard_normal(10)
        coef = np.array(cutquadric.p2_to_monomial(u))
        for _ in range(5):
            x = rng.uniform(0, 1, 3)
            from_basis = sum(u[i] * space.basis[i](x) for i in range(10))
            assert mono_eval(coef, x)[0] == pytest.approx(from_basis, abs=1e-13)


def test_u0_quadratic_sampled_at_nodes_is_reproduced():
    rng = np.random.default_rng(1)
    for _ in range(10):
        c = rng.standard_normal(10)
        u = mono_eval(c, NODES)
        assert np.max(np.abs(np.array(cutquadric.p2_to_monomial(u)) - c)) <= 1e-13
        pts = rng.uniform(0, 1, (20, 3))
        assert np.max(np.abs(mono_eval(np.array(cutquadric.p2_to_monomial(u)), pts) - mono_eval(c, pts))) <= 1e-13


# ---------------------------------------------------------------------------------------------------- U1
def sphere_u(c, r):
    c = np.asarray(c, float)
    return nodal(lambda x, y, z: r * r - np.sum((np.array([x, y, z]) - c) ** 2))


INCENTRE = np.full(3, 1.0 / (3.0 + math.sqrt(3.0)))


def test_u1_uncut_and_simple():
    c = cutquadric.classify(nodal(lambda x, y, z: 1.0 + 0.2 * x - 0.1 * y * z))
    assert (c["label"], c["cut"]) == ("uncut_positive", False)
    c = cutquadric.classify(nodal(lambda x, y, z: -1.0 + 0.2 * x * x))
    assert (c["label"], c["cut"]) == ("uncut_negative", False)
    c = cutquadric.classify(linear_p2([0.31, -0.62, 0.45, -0.27]))
    assert (c["label"], c["cut"], c["double_edges"]) == ("simple", True, 0)
    c = cutquadric.classify(sphere_u([0.0, 0.0, 0.0], 0.4))               # octant of a ball at a vertex
    assert (c["label"], c["cut"]) == ("simple", True)


def test_u1_edge_cut_twice():
    delta = 0.01                      # u = (x - 1/2)^2 - delta: vertices all positive, three edges dip negative
    u = nodal(lambda x, y, z: (x - .5) ** 2 - delta)
    assert np.all(u[:4] > 0) and np.any(u[4:] < 0)
    c = cutquadric.classify(u)
    assert (c["label"], c["cut"]) == ("edge_cut_twice", True)
    assert c["double_edges"] == 3


def test_u1_sub_nodal_cap_and_closed_component():
    # a cap: a ball centred outside the tet just reaches through the face x = 0, no node of the P2 element inside it
    u = sphere_u([-0.1, 0.3, 0.3], 0.2)
    assert np.all(u < 0)
    c = cutquadric.classify(u)
    assert (c["label"], c["cut"]) == ("sub_nodal", True)
    assert c["bernstein_max"] > 0 and c["range_max"] > 0 > c["range_min"]
    # a closed component: ball of radius 0.15 at the incentre; the note's Bernstein range is -0.69 .. +0.31
    u = sphere_u(INCENTRE, 0.15)
    assert np.all(u < 0)
    c = cutquadric.classify(u)
    assert (c["label"], c["cut"]) == ("sub_nodal", True)
    assert c["bernstein_min"] == pytest.approx(-0.69, abs=0.01) and c["bernstein_max"] == pytest.approx(0.31, abs=0.01)
    # the same ball, the other sign: the whole tet except the ball is positive, still cut
    c = cutquadric.classify(-u)
    assert (c["label"], c["cut"]) == ("sub_nodal", True)


def test_u1_bernstein_is_only_sufficient():
    # u = (x - 1/2)^2 + 0.001 is positive on the whole tet, but the mid-edge Bernstein coefficient of edge (0,1) is
    # 0.001 - 1/4 < 0: the sign test cannot decide, the exact range test can
    u = nodal(lambda x, y, z: (x - .5) ** 2 + 0.001)
    c = cutquadric.classify(u)
    assert c["bernstein_min"] < 0 < c["range_min"]
    assert (c["label"], c["cut"]) == ("uncut_positive", False)


@pytest.mark.parametrize("seed", range(8))
def test_u1_range_against_sampling(seed):
    rng = np.random.default_rng(10 + seed)
    pts = []
    n = 24
    for i in range(n + 1):
        for j in range(n + 1 - i):
            for k in range(n + 1 - i - j):
                pts.append((i / n, j / n, k / n))
    pts = np.array(pts)
    for _ in range(40):
        u = rng.standard_normal(10) * rng.choice([0.3, 1.0, 3.0]) + rng.choice([-1.0, 0.0, 1.0])
        cls = cutquadric.classify(u)
        vals = mono_eval(np.array(cutquadric.p2_to_monomial(u)), pts)
        smin, smax = vals.min(), vals.max()
        scale = max(1.0, np.max(np.abs(u)))
        # the exact range contains every sampled value, and is within the sampling resolution of it
        assert cls["range_min"] <= smin + 1e-12 * scale and cls["range_max"] >= smax - 1e-12 * scale
        assert smin - cls["range_min"] <= 0.1 * scale and cls["range_max"] - smax <= 0.1 * scale
        if smin < 0 < smax:
            assert cls["cut"]
        if not cls["cut"]:
            assert cls["label"] in ("uncut_positive", "uncut_negative")
            assert (cls["range_min"] >= 0) or (cls["range_max"] <= 0)
        # Bernstein bounds enclose the range
        assert cls["bernstein_min"] <= cls["range_min"] + 1e-12 * scale
        assert cls["bernstein_max"] >= cls["range_max"] - 1e-12 * scale
        if cls["bernstein_min"] > 0:
            assert cls["label"] == "uncut_positive"


def test_u1_labels_cover_all_cases_with_random_fields():
    rng = np.random.default_rng(99)
    seen = set()
    for _ in range(4000):
        u = rng.standard_normal(10) * rng.choice([0.1, 1.0]) + rng.choice([-0.5, 0.0, 0.5])
        seen.add(cutquadric.classify(u)["label"])
    assert seen == {"uncut_positive", "uncut_negative", "simple", "edge_cut_twice", "sub_nodal"}
