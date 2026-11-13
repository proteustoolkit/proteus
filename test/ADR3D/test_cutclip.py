"""C44 phase 1: the clipping oracle (proteus/cutclip.h) and the checks the 3D cut kernels are measured against.

U1  clipped triangle: plane +/- partition of every monomial of degree <= 4
U2  clipped tet: the same in 3D, all three topologies (1-3, 2-2, 3-1); parity with an independent scipy oracle and
    with the July P1 reference (golden values below)
U3  fit against clip: the equivalent-polynomial fit of order nP reproduces the clipped moment of every monomial of
    degree <= nP and misses at nP + 1
U5  degenerate cuts: plane through a vertex, an edge, a face; partition identity
U6  the cut face of two neighbours, given the same nodal level-set values, clips to the same moments
"""
import itertools

import numpy as np
import pytest

from proteus import cutclip

ref_tet = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
DEG = 4
IDX = cutclip.monomials(DEG)


def random_tet(rng, scale=1.0):
    while True:
        t = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]) + 0.35 * rng.standard_normal((4, 3))
        if abs(np.linalg.det(t[1:] - t[0])) > 0.2:
            return scale * t


def plane_from_nodal(tet, phi):
    """a.x + b = phi at the four vertices (P1 level set on this element)."""
    return np.linalg.solve(np.hstack([tet, np.ones((4, 1))]), np.asarray(phi, float))


def full_moments(tet, deg=DEG):
    return np.array(cutclip.tet_moments(tet, np.zeros((0, 4)), deg))


# ---------------------------------------------------------------- independent oracle (scipy hull + other quadrature)
def scipy_moments(tet, planes, deg=DEG):
    from scipy.spatial import ConvexHull, Delaunay, HalfspaceIntersection
    from proteus.Quadrature import GaussTetrahedron
    # inequalities A x + b <= 0 for scipy; tet facets then the planes (kept side a.x + b > 0 -> -a.x - b <= 0)
    hs = []
    c = tet.mean(axis=0)
    for i in range(4):
        f = [j for j in range(4) if j != i]
        n = np.cross(tet[f[1]] - tet[f[0]], tet[f[2]] - tet[f[0]])
        b = -n @ tet[f[0]]
        if n @ c + b > 0:
            n, b = -n, -b
        hs.append([*n, b])
    for p in planes:
        hs.append([-p[0], -p[1], -p[2], -p[3]])
    hs = np.array(hs)
    # interior point by LP
    from scipy.optimize import linprog
    norm = np.linalg.norm(hs[:, :3], axis=1)
    res = linprog([0, 0, 0, -1], A_ub=np.hstack([hs[:, :3], norm[:, None]]), b_ub=-hs[:, 3],
                  bounds=[(None, None)] * 3 + [(0, None)])
    if res.status != 0 or res.x[3] < 1e-9:
        return np.zeros(len(cutclip.monomials(deg)))
    pts = HalfspaceIntersection(hs, res.x[:3]).intersections
    tri = Delaunay(pts)
    quad = GaussTetrahedron(order=8)
    X, W = np.array(quad.points), np.array(quad.weights)
    m = np.zeros(len(cutclip.monomials(deg)))
    for s in tri.simplices:
        v = pts[s]
        J = (v[1:] - v[0]).T
        vol = abs(np.linalg.det(J))
        xp = v[0] + X @ J.T
        for n, (p, q, r) in enumerate(cutclip.monomials(deg)):
            m[n] += np.sum(W * vol * xp[:, 0] ** p * xp[:, 1] ** q * xp[:, 2] ** r)
    return m


@pytest.mark.parametrize("seed", range(6))
def test_u2_partition_identity_all_topologies(seed):
    rng = np.random.default_rng(seed)
    tet = random_tet(rng)
    full = full_moments(tet)
    seen = set()
    for _ in range(40):
        phi = rng.uniform(-1, 1, 4)
        pl = plane_from_nodal(tet, phi)
        seen.add(int((phi > 0).sum()))
        plus = np.array(cutclip.tet_moments(tet, pl[None], DEG))
        minus = np.array(cutclip.tet_moments(tet, -pl[None], DEG))
        assert np.max(np.abs(plus + minus - full)) <= 1e-13 * max(1.0, np.max(np.abs(full)))
    assert {1, 2, 3} <= seen          # all three topologies were exercised


@pytest.mark.parametrize("seed", range(4))
def test_u2_against_independent_scipy_oracle(seed):
    rng = np.random.default_rng(100 + seed)
    tet = random_tet(rng)
    for _ in range(6):
        pls = np.array([plane_from_nodal(tet, rng.uniform(-1, 1, 4)) for _ in range(1 + seed % 2 + 1)])
        got = np.array(cutclip.tet_moments(tet, pls, DEG))
        want = scipy_moments(tet, pls)
        assert np.max(np.abs(got - want)) <= 1e-12 * max(1.0, np.max(np.abs(want)))


@pytest.mark.parametrize("seed", range(4))
def test_u1_clipped_face_partition(seed):
    rng = np.random.default_rng(200 + seed)
    tet = random_tet(rng)
    for face in itertools.combinations(range(4), 3):
        tri = tet[list(face)]
        full = np.array(cutclip.tri_moments(tri, np.zeros((0, 4)), DEG))
        for _ in range(10):
            pl = plane_from_nodal(tet, rng.uniform(-1, 1, 4))
            plus = np.array(cutclip.tri_moments(tri, pl[None], DEG))
            minus = np.array(cutclip.tri_moments(tri, -pl[None], DEG))
            assert np.max(np.abs(plus + minus - full)) <= 1e-13 * max(1.0, np.max(np.abs(full)))


def test_u1_face_area_closed_form():
    # triangle (0,0,0),(1,0,0),(0,1,0) cut by x - 0.25 > 0: kept area = 0.5 (1 - 0.25)^2
    tri = np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0]])
    area = cutclip.tri_moments(tri, np.array([[1., 0, 0, -0.25]]), 0)[0]
    assert area == pytest.approx(0.5 * 0.75 ** 2, abs=1e-15)


def test_u4_cut_surface_area_and_moments():
    # plane x + y + z = 0.5 on the unit tet: equilateral triangle, side 0.5 sqrt(2): area sqrt(3)/4 * 0.5 = 0.2165
    pl = np.array([1., 1, 1, -0.5])
    area = cutclip.surface_moments(ref_tet, pl, np.zeros((0, 4)), 0)[0]
    assert area == pytest.approx(np.sqrt(3) / 4 * 0.5, abs=1e-15)
    # 2-2 split: y + z = 0.5 has vertex values -0.5, -0.5, 0.5, 0.5, a rectangle
    pl = np.array([0., 1, 1, -0.5])
    area = cutclip.surface_moments(ref_tet, pl, np.zeros((0, 4)), 0)[0]
    # the section is the rectangle x in [0,0.5], with y+z=0.5 of length 0.5 sqrt 2: area 0.5 * 0.5 sqrt 2
    assert area == pytest.approx(0.5 * 0.5 * np.sqrt(2), abs=1e-15)


@pytest.mark.parametrize("kind", ["vertex", "edge", "face"])
def test_u5_degenerate_planes_partition(kind):
    rng = np.random.default_rng(7)
    for _ in range(10):
        tet = random_tet(rng)
        full = full_moments(tet)
        if kind == "vertex":
            phi = np.array([0., rng.uniform(.1, 1), -rng.uniform(.1, 1), rng.uniform(-1, 1)])
        elif kind == "edge":
            phi = np.array([0., 0., rng.uniform(.1, 1), -rng.uniform(.1, 1)])
        else:
            phi = np.array([0., 0., 0., rng.uniform(.1, 1)])
        pl = plane_from_nodal(tet, phi)
        plus = np.array(cutclip.tet_moments(tet, pl[None], DEG))
        minus = np.array(cutclip.tet_moments(tet, -pl[None], DEG))
        assert np.max(np.abs(plus + minus - full)) <= 1e-13 * max(1.0, np.max(np.abs(full)))
    # and a near-degenerate theta = 1e-12: vertex value 1e-12 against the others
    tet = random_tet(rng)
    pl = plane_from_nodal(tet, [1e-12, 1.0, 0.7, -0.4])
    plus = np.array(cutclip.tet_moments(tet, pl[None], DEG))
    minus = np.array(cutclip.tet_moments(tet, -pl[None], DEG))
    assert np.max(np.abs(plus + minus - full_moments(tet))) <= 1e-13


def test_u5_plane_containing_a_face_keeps_everything_or_nothing():
    tet = ref_tet
    pl = plane_from_nodal(tet, [0., 0., 0., 1.0])      # zero on face 0,1,2, positive at vertex 3: all kept
    assert cutclip.tet_moments(tet, pl[None], 0)[0] == pytest.approx(1 / 6, abs=1e-16)
    assert cutclip.tet_moments(tet, -pl[None], 0)[0] == 0.0


@pytest.mark.parametrize("seed", range(4))
def test_u6_shared_face_clips_identically_from_both_neighbours(seed):
    rng = np.random.default_rng(300 + seed)
    face = rng.standard_normal((3, 3))
    apex1, apex2 = face.mean(0) + [0, 0, 1.0], face.mean(0) - [0, 0, 1.0]
    # nodal level-set values are shared on the face; each neighbour adds its own apex value
    phi_face = rng.uniform(-1, 1, 3)
    t1 = np.vstack([face, apex1])
    t2 = np.vstack([face[[2, 0, 1]], apex2])           # a different local vertex order
    phi1 = np.r_[phi_face, 0.3]
    phi2 = np.r_[phi_face[[2, 0, 1]], -0.8]
    m1 = np.array(cutclip.tri_moments(t1[:3], plane_from_nodal(t1, phi1)[None], DEG))
    m2 = np.array(cutclip.tri_moments(t2[:3], plane_from_nodal(t2, phi2)[None], DEG))
    assert np.max(np.abs(m1 - m2)) <= 1e-12 * max(1.0, np.max(np.abs(m1)))


# ---------------------------------------------------------------- July P1 reference, golden values (C7)
# Integral of a P1 function w over {u > 0} (and {u > 0} and {v > 0}) in the unit tet, nodal values at (0,0,0), (1,0,0),
# (0,1,0), (0,0,1). Values from the July Python reference p1_two_level_sets_tet.py and, independently, from
# cutquad.cpp rebuilt on this machine on 2026-11-13 (agree to 3e-17); cutclip agrees to 4e-16.
JULY_U = [[1, -1, -1, -1], [1, 1, -1, -1], [-0.3, 0.9, 0.5, -0.2], [0.4, -0.7, 0.8, 0.1], [1, 1, 1, -1],
          [-1, 0.2, 0.2, 0.2]]
JULY_V = [[-1, 1, 1, 1], [-1, 1, 1, -1], [0.6, -0.4, 0.3, 0.5], [-0.5, 0.5, -0.1, 0.9], [-1, -1, 1, 1],
          [0.3, -0.8, 0.4, 0.6]]
JULY_W = [[1, 1, 1, 1], [1, 2, 3, 4], [0.5, -1, 2, 0.25], [2, -1, 0.5, 1.5], [1, 0, 0, 0], [0, 1, -1, 1]]
JULY_ONE = [0.02083333333333333, 0.17708333333333337, 0.059416103250442734, 0.1163752941345271,
            0.03906249999999999, 0.021572788065843618]
JULY_TWO = [0.0, 0.09375000000000001, 0.06419984390719925, 0.07562097697754958, 0.010416666666666666,
            0.008410563996260372]


def p1_integral(fs, w):
    planes = np.array([[f[1] - f[0], f[2] - f[0], f[3] - f[0], f[0]] for f in fs], float)
    m = dict(zip(map(tuple, cutclip.monomials(1)), cutclip.tet_moments(ref_tet, planes, 1)))
    return (w[0] * (m[(0, 0, 0)] - m[(1, 0, 0)] - m[(0, 1, 0)] - m[(0, 0, 1)]) + w[1] * m[(1, 0, 0)]
            + w[2] * m[(0, 1, 0)] + w[3] * m[(0, 0, 1)])


@pytest.mark.parametrize("i", range(6))
def test_july_p1_parity_golden_values(i):
    assert p1_integral([JULY_U[i]], JULY_W[i]) == pytest.approx(JULY_ONE[i], abs=1e-15)
    assert p1_integral([JULY_U[i], JULY_V[i]], JULY_W[i]) == pytest.approx(JULY_TWO[i], abs=1e-15)


# ---------------------------------------------------------------- U3: fit against clip
@pytest.mark.parametrize("nP", [1, 2, 3])
@pytest.mark.parametrize("phi", [[-0.3, 0.8, 0.5, -0.4], [-0.3, 0.8, 0.5, 0.4], [0.9, -0.2, -0.5, -0.7]])
def test_u3_fit_reproduces_clipped_moments_to_degree_nP_and_misses_above(nP, phi):
    from proteus import equivalent_polynomials as eqp
    from proteus.Quadrature import GaussTetrahedron
    rng = np.random.default_rng(3)
    tet = ref_tet + 0.2 * rng.standard_normal((4, 3))
    phi = np.array(phi)
    pl = plane_from_nodal(tet, phi)
    quad = GaussTetrahedron(order=8)
    X, W = np.array(quad.points), np.array(quad.weights)
    J = (tet[1:] - tet[0]).T
    xp = tet[0] + X @ J.T
    vol = abs(np.linalg.det(J))
    gf = eqp.Simplex(nSpace=3, nP=nP, nQ=len(X))
    gf.calculate(phi, tet, X)
    H = np.empty(len(X))
    for q in range(len(X)):
        gf.set_quad(q)
        H[q] = gf.H
    clip = np.array(cutclip.tet_moments(tet, pl[None], nP + 1))
    err = {}
    for n, (p, q, r) in enumerate(cutclip.monomials(nP + 1)):
        fit = np.sum(W * vol * H * xp[:, 0] ** p * xp[:, 1] ** q * xp[:, 2] ** r)
        d = p + q + r
        err[d] = max(err.get(d, 0.0), abs(fit - clip[n]))
    for d in range(nP + 1):
        assert err[d] <= 1e-12, (nP, d, err[d])
    assert err[nP + 1] > 1e-7      # the documented miss at degree nP + 1
