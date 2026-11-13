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


# ================================================================================================== A2: volume moments
def gam_sphere(beta, octant):
    """integral over the unit sphere (or its first octant) of n^beta."""
    if not octant and any(b % 2 for b in beta):
        return 0.0
    g = np.prod([math.gamma((b + 1) / 2) for b in beta]) / math.gamma((sum(beta) + 3) / 2)
    return 0.25 * g if octant else 2.0 * g


def ball_volume_moments(c, r, deg, octant=False):
    """integral of x^alpha over the ball of radius r about c (or, with octant, the first octant of the ball about 0)."""
    out = []
    for (p, q, s) in cutclip.monomials(deg):
        alpha = (p, q, s)
        tot = 0.0
        for beta in itertools.product(*[range(a + 1) for a in alpha]):
            coef = np.prod([math.comb(a, b) * c[i] ** (a - b) for i, (a, b) in enumerate(zip(alpha, beta))])
            nb = sum(beta)
            tot += coef * r ** (nb + 3) / (nb + 3) * gam_sphere(beta, octant)
        out.append(tot)
    return np.array(out)


def full_moments(deg):
    return np.array(cutclip.tet_moments(ref_tet, np.zeros((0, 4)), deg))


def plane_of(phi):
    phi = np.asarray(phi, float)
    return np.array([phi[1] - phi[0], phi[2] - phi[0], phi[3] - phi[0], phi[0]])


# July tet_quad_u (cutquad.cpp, rebuilt locally 2026-11-13, order 64) at u given in proteus order: integral over {u > 0}
# of the monomials GOLD_MON. Generic random quadrics (seeded 2026, rounded to 3 digits) and the octant of a ball r = 0.4.
GOLD_U = [
    [-0.642, 0.28, -0.065, -0.259, -0.29, 0.581, 0.81, -0.645, 0.306, -0.403],
    [0.934, 0.84, 0.272, 0.505, 0.03, 0.652, -0.103, -0.322, -0.444, -0.547],
    [0.052, -0.138, 0.326, -0.974, -0.105, -0.27, -0.609, 0.19, -0.129, -0.4],
    [-0.581, 0.749, 0.595, 0.213, -0.31, 0.894, 0.127, -0.134, 0.801, -0.361],
    [0.392, -0.372, -0.477, 0.402, -0.544, -0.014, 0.16, -0.622, 0.462, 0.097],
    [0.243, -0.256, -0.16, -0.01, -0.06, 0.351, 0.154, -0.167, -0.996, 0.588],
    [0.16, -0.84, -0.84, -0.84, -0.09, -0.34, -0.34, -0.09, -0.34, -0.09],
]
GOLD_MON = [(0, 0, 0), (1, 0, 0), (0, 1, 1), (2, 0, 1), (1, 1, 1), (0, 0, 3)]
GOLD_VAL = [
    [0.10345528059321807, 0.031120880939368534, 0.006861100683863874, 0.002423086069099657, 0.0012329369901757858, 0.0059467815754211],
    [0.026403058815367492, 0.011019272299206538, 0.0006062050056157083, 0.00042224723421322764, 0.00018552675871393395, 0.0005836897624046226],
    [0.010189474546237174, 0.0006321037939281349, 0.0002271913423554042, 1.5847639956858525e-06, 1.1095049413698142e-05, 2.0520514072544913e-06],
    [0.11060823886342377, 0.03359597305300239, 0.006486385450124173, 0.002529587847314849, 0.0012366168768382982, 0.006464374677847813],
    [0.059603849256863656, 0.014105297290406969, 0.004190831108072278, 0.001621706931256944, 0.0007660336780055362, 0.0073483973128458694],
    [0.09744678438366536, 0.016325247339488844, 0.005615906128875937, 0.0004425553106001235, 0.0006383128172086954, 0.004110949827531276],
    [0.03351032163829113, 0.005026548245743673, 0.0006826666666666665, 0.00013404128655316463, 8.53333333333333e-05, 0.0002680825731063292],
]


@pytest.mark.parametrize("case", range(len(GOLD_U)))
def test_u10_july_golden_values(case):
    idx = {m: i for i, m in enumerate(map(tuple, cutclip.monomials(3)))}
    # parity at the same order. The two codes agree to 1e-15 at equal order; the limit itself is reached more slowly for
    # some random quadrics (case 5: 5e-9 at order 16, 5e-12 at 32, ~1e-13 at 64), which is the July method's endpoint
    # behaviour, not the port.
    got = np.array(cutquadric.volume_moments(np.array(GOLD_U[case]), 3, 64))
    for m, v in zip(GOLD_MON, GOLD_VAL[case]):
        assert got[idx[m]] == pytest.approx(v, abs=1e-13)


def test_u2_octant_of_a_ball_closed_form():
    r = 0.4
    want = ball_volume_moments(np.zeros(3), r, 4, octant=True)
    assert want[0] == pytest.approx(np.pi * r ** 3 / 6, rel=1e-15)
    got = np.array(cutquadric.volume_moments(sphere_u([0, 0, 0], r), 4, 24))
    assert np.max(np.abs(got - want)) <= 1e-13


def test_u2_ball_inside_the_tet_closed_form():
    r = 0.15
    want = ball_volume_moments(INCENTRE, r, 4)
    got = np.array(cutquadric.volume_moments(sphere_u(INCENTRE, r), 4, 24))
    assert np.max(np.abs(got - want)) <= 1e-13
    # the complement: the tet minus the ball
    got_c = np.array(cutquadric.volume_moments(-sphere_u(INCENTRE, r), 4, 24))
    assert np.max(np.abs(got_c - (full_moments(4) - want))) <= 1e-13


def test_u2_degree_above_cutclip_limit_raises():
    with pytest.raises(Exception):
        cutquadric.volume_moments(sphere_u([0, 0, 0], 0.4), 8, 8)


@pytest.mark.parametrize("planar_tol", [1e-12, 0.0])
def test_u3_linear_level_set_embedded_as_p2_equals_cutclip(planar_tol):
    """planar_tol = 1e-12: the fast path (cutclip itself); 0: forced slicing, which sees round-off quadratic terms."""
    rng = np.random.default_rng(3)
    topologies = set()
    worst = 0.0
    for _ in range(60):
        phi = rng.uniform(-1, 1, 4)
        topologies.add(int((phi > 0).sum()))
        want = np.array(cutclip.tet_moments(ref_tet, plane_of(phi)[None], 4))
        for order in (12, 24):
            got = np.array(cutquadric.volume_moments(linear_p2(phi), 4, order, planar_tol=planar_tol))
            worst = max(worst, np.max(np.abs(got - want)))
    assert {1, 2, 3} <= topologies
    assert worst <= 1e-13


def test_u3_planar_two_two_split_regression():
    """The July Python quadric code returns 1e-5..1e-7 here, at any order (round-off leading coefficient in its
    root finder). Forced slicing with leading-coefficient deflation must not."""
    phi = np.array([0.31, -0.62, 0.45, -0.27])
    want = np.array(cutclip.tet_moments(ref_tet, plane_of(phi)[None], 3))
    for order in (8, 16, 24):
        got = np.array(cutquadric.volume_moments(linear_p2(phi), 3, order, planar_tol=0.0))
        assert np.max(np.abs(got - want)) <= 1e-13


@pytest.mark.parametrize("kappa", [1e-1, 1e-3, 1e-6, 1e-9, 1e-11, 1e-12, 1e-13, 1e-15, 0.0])
def test_u4_nearly_planar_across_the_fast_path_threshold(kappa):
    phi = np.array([0.31, -0.62, 0.45, -0.27])
    bump = lambda x, y, z: x * x + 0.7 * y * z - 0.4 * z * z + 0.3 * x * y
    u = linear_p2(phi) + kappa * nodal(bump)
    planar = np.array(cutclip.tet_moments(ref_tet, plane_of(phi)[None], 3))
    ref = np.array(cutquadric.volume_moments(u, 3, 40, planar_tol=0.0))        # slicing, high order
    got = np.array(cutquadric.volume_moments(u, 3, 16))                          # default: fast path below 1e-12
    assert np.max(np.abs(got - ref)) <= 1e-12                                    # no loss of accuracy at the threshold
    # and it is a smooth perturbation of the planar answer: no jump anywhere
    assert np.max(np.abs(got - planar)) <= 2.0 * kappa + 1e-13


def generic_sphere_cut():
    R = 2.5
    cc = np.array([0.2, 0.35, 0.25]) - R * np.array([0.48, 0.6, 0.64])
    return sphere_u(cc, R)


def test_u5_order_sweep_converges_spectrally():
    u = generic_sphere_cut()
    ref = np.array(cutquadric.volume_moments(u, 4, 48))
    err = {o: np.max(np.abs(np.array(cutquadric.volume_moments(u, 4, o)) - ref)) for o in (4, 6, 8, 12, 16, 24)}
    assert err[16] <= 1e-12
    assert err[24] <= 1e-13
    assert err[8] < err[6] < err[4]                       # stagnation would mean a missed event
    assert err[12] < err[8]
    # the same sweep on closed forms
    r = 0.4
    want = ball_volume_moments(np.zeros(3), r, 4, octant=True)
    e = [np.max(np.abs(np.array(cutquadric.volume_moments(sphere_u([0, 0, 0], r), 4, o)) - want)) for o in (6, 8, 12, 16)]
    assert e[3] <= 1e-13 and e[2] < e[1] < e[0]


@pytest.mark.parametrize("seed", range(6))
def test_u8_partition_i_of_u_plus_i_of_minus_u_is_the_full_tet(seed):
    rng = np.random.default_rng(300 + seed)
    full = full_moments(4)
    labels = set()
    for _ in range(25):
        u = rng.standard_normal(10) * rng.choice([0.2, 1.0]) + rng.choice([-0.3, 0.0, 0.3])
        labels.add(cutquadric.classify(u)["label"])
        plus = np.array(cutquadric.volume_moments(u, 4, 24))
        minus = np.array(cutquadric.volume_moments(-u, 4, 24))
        assert np.max(np.abs(plus + minus - full)) <= 1e-13
    assert len(labels) >= 3


# ================================================================================================== A3: faces
# face k is opposite vertex k; its vertices are the other three in increasing order, its P2 values are
# (v0, v1, v2, (v0,v1), (v1,v2), (v0,v2)) taken from the element's ten.
FACE_VERTS = [[m for m in range(4) if m != k] for k in range(4)]


def face_points(k):
    v = FACE_VERTS[k]
    a, b, c = ref_tet[v[0]], ref_tet[v[1]], ref_tet[v[2]]
    st = np.array([[0, 0], [1, 0], [0, 1], [.5, 0], [.5, .5], [0, .5]])
    return a + st[:, :1] * (b - a) + st[:, 1:] * (c - a)


TRI_NODES = np.array([[0, 0], [1, 0], [0, 1], [.5, 0], [.5, .5], [0, .5]])
IDX2D = cutquadric.monomials2(3)


def nodal2(f):
    return np.array([f(s, t) for s, t in TRI_NODES])


@pytest.mark.parametrize("k", range(4))
def test_a3_face_nodal_values(k):
    rng = np.random.default_rng(20 + k)
    c = rng.standard_normal(10)
    u = mono_eval(c, NODES)
    want = mono_eval(c, face_points(k))
    assert np.max(np.abs(np.array(cutquadric.face_nodal(u, k)) - want)) <= 1e-13


def test_a3_monomials2_order():
    assert cutquadric.monomials2(2) == [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]]


def test_a3_face_classification():
    f = cutquadric.classify_face
    assert f(nodal2(lambda s, t: 1 + s * t))["label"] == "uncut_positive"
    assert f(nodal2(lambda s, t: -1 - s))["label"] == "uncut_negative"
    c = f(nodal2(lambda s, t: s - 0.3))                                  # a line
    assert c["label"] == "cut" and c["cut"]
    c = f(nodal2(lambda s, t: (s - .5) ** 2 + (t - .5) ** 2 - 0.1))     # vertices (0,0): .4 > 0 ... mids inside: cut twice
    assert c["cut"]
    # all six nodal values negative but a disc pokes into the face (no node inside it)
    u = nodal2(lambda s, t: 0.04 - (s + 0.1) ** 2 - (t - 0.3) ** 2)
    assert np.all(u < 0)
    c = f(u)
    assert c["cut"] and c["range_max"] > 0
    # the closed component: disc at the centroid, r = 0.15
    u = nodal2(lambda s, t: 0.0225 - (s - 1 / 3) ** 2 - (t - 1 / 3) ** 2)
    assert np.all(u < 0) and f(u)["cut"]


def test_a3_shared_face_gives_the_same_answer_from_both_elements():
    """The same face seen with its three vertices in another order (the neighbour's local numbering)."""
    rng = np.random.default_rng(7)
    ed = [(0, 1), (1, 2), (0, 2)]
    for _ in range(300):
        u = rng.standard_normal(6) * rng.choice([0.2, 1.0]) + rng.choice([-0.4, 0.0, 0.4])
        base = cutquadric.classify_face(u)
        for perm in itertools.permutations(range(3)):
            # new local vertex i is old vertex perm[i]; the mid-edge value of (i,j) is the old mid of (perm[i], perm[j])
            def mid(i, j):
                a, b = sorted((perm[i], perm[j]))
                return u[3 + ed.index((a, b))]
            up = np.array([u[perm[0]], u[perm[1]], u[perm[2]], mid(0, 1), mid(1, 2), mid(0, 2)])
            c = cutquadric.classify_face(up)
            assert c["label"] == base["label"]
            assert c["range_min"] == pytest.approx(base["range_min"], abs=1e-13)
            assert c["range_max"] == pytest.approx(base["range_max"], abs=1e-13)


def test_a3_face_range_against_sampling():
    rng = np.random.default_rng(8)
    n = 60
    pts = np.array([(i / n, j / n) for i in range(n + 1) for j in range(n + 1 - i)])
    for _ in range(80):
        u = rng.standard_normal(6) * rng.choice([0.3, 1.0]) + rng.choice([-1.0, 0.0, 1.0])
        c = cutquadric.classify_face(u)
        coef = np.array(cutquadric.p2_tri_to_monomial(u))
        vals = sum(cc * pts[:, 0] ** p * pts[:, 1] ** q for cc, (p, q) in zip(coef, IDX2D_FULL))
        sc = max(1.0, np.max(np.abs(u)))
        assert c["range_min"] <= vals.min() + 1e-12 * sc and c["range_max"] >= vals.max() - 1e-12 * sc
        assert vals.min() - c["range_min"] <= 0.02 * sc and c["range_max"] - vals.max() <= 0.02 * sc
        assert c["bernstein_min"] <= c["range_min"] + 1e-12 * sc and c["bernstein_max"] >= c["range_max"] - 1e-12 * sc
        if vals.min() < 0 < vals.max():
            assert c["cut"]


IDX2D_FULL = cutquadric.monomials2(2)


def disc_sphere_moments(c, r, deg, quarter=False):
    """integral of s^p t^q over the disc of radius r about c (quarter: the first quadrant of the disc about 0)."""
    out = []
    for (p, q) in cutquadric.monomials2(deg):
        tot = 0.0
        for beta in itertools.product(range(p + 1), range(q + 1)):
            coef = math.comb(p, beta[0]) * c[0] ** (p - beta[0]) * math.comb(q, beta[1]) * c[1] ** (q - beta[1])
            nb = sum(beta)
            if quarter:
                g = 0.5 * np.prod([math.gamma((b + 1) / 2) for b in beta]) / math.gamma((nb + 2) / 2)
            else:
                g = 0.0 if any(b % 2 for b in beta) else 2 * np.prod([math.gamma((b + 1) / 2) for b in beta]) / math.gamma((nb + 2) / 2)
            tot += coef * r ** (nb + 2) / (nb + 2) * g
        out.append(tot)
    return np.array(out)


def full_tri_moments(deg):
    return np.array([math.factorial(p) * math.factorial(q) / math.factorial(p + q + 2) for p, q in cutquadric.monomials2(deg)])


def test_a3_tri_moments_closed_forms():
    r = 0.4
    want = disc_sphere_moments((0, 0), r, 5, quarter=True)
    assert want[0] == pytest.approx(np.pi * r * r / 4)
    u = nodal2(lambda s, t: r * r - s * s - t * t)
    errs = [np.max(np.abs(np.array(cutquadric.tri_moments(u, 5, o)) - want)) for o in (6, 8, 12, 16, 24)]
    assert errs[-1] <= 1e-13 and errs[2] < errs[1] < errs[0]
    c, r = (1 / 3, 1 / 3), 0.15
    u = nodal2(lambda s, t: r * r - (s - c[0]) ** 2 - (t - c[1]) ** 2)
    assert np.all(u < 0)
    want = disc_sphere_moments(c, r, 5)
    assert np.max(np.abs(np.array(cutquadric.tri_moments(u, 5, 24)) - want)) <= 1e-13
    assert np.max(np.abs(np.array(cutquadric.tri_moments(-u, 5, 24)) - (full_tri_moments(5) - want))) <= 1e-13


@pytest.mark.parametrize("planar_tol", [1e-12, 0.0])
def test_a3_tri_linear_embedded_equals_cutclip(planar_tol):
    rng = np.random.default_rng(31)
    tri3 = np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0]])
    worst = 0.0
    for _ in range(60):
        phi = rng.uniform(-1, 1, 3)
        pl = np.array([phi[1] - phi[0], phi[2] - phi[0], 0.0, phi[0]])
        u = nodal2(lambda s, t: phi[0] * (1 - s - t) + phi[1] * s + phi[2] * t)
        full3 = np.array(cutclip.tri_moments(tri3, pl[None], 4))
        want = np.array([full3[i] for i, (p, q, r) in enumerate(cutclip.monomials(4)) if r == 0])
        got = np.array(cutquadric.tri_moments(u, 4, 24, planar_tol=planar_tol))
        worst = max(worst, np.max(np.abs(got - want)))
    assert worst <= 1e-13


def test_a3_tri_partition_and_order_sweep():
    rng = np.random.default_rng(32)
    full = full_tri_moments(4)
    for _ in range(80):
        u = rng.standard_normal(6) * rng.choice([0.2, 1.0]) + rng.choice([-0.3, 0.0, 0.3])
        plus = np.array(cutquadric.tri_moments(u, 4, 32))
        minus = np.array(cutquadric.tri_moments(-u, 4, 32))
        assert np.max(np.abs(plus + minus - full)) <= 1e-12


def test_a3_tri_nearly_planar_across_threshold():
    base = nodal2(lambda s, t: 0.2 - s + 0.3 * t)
    bump = nodal2(lambda s, t: s * s - 0.5 * s * t + 0.3 * t * t)
    for kappa in (1e-1, 1e-6, 1e-11, 1e-12, 1e-13, 0.0):
        u = base + kappa * bump
        ref = np.array(cutquadric.tri_moments(u, 4, 40, planar_tol=0.0))
        got = np.array(cutquadric.tri_moments(u, 4, 16))
        assert np.max(np.abs(got - ref)) <= 1e-12


@pytest.mark.parametrize("k", range(4))
def test_a3_face_moments_linear_u_against_cutclip(k):
    rng = np.random.default_rng(40 + k)
    v = FACE_VERTS[k]
    tri3 = ref_tet[v]
    jac = np.linalg.norm(np.cross(tri3[1] - tri3[0], tri3[2] - tri3[0]))     # dA = jac ds dt
    for _ in range(20):
        phi = rng.uniform(-1, 1, 4)
        u = linear_p2(phi)
        want = np.array(cutclip.tri_moments(tri3, plane_of(phi)[None], 4)) / jac
        got = np.array(cutquadric.face_moments(u, k, 4, 24))
        assert np.max(np.abs(got - want)) <= 1e-13


def test_a3_face_moments_curved_closed_form():
    # ball about (-0.1, 0.3, 0.3), r = 0.2 pokes through face 1 (x = 0, vertices 0,2,3; (s,t) = (y,z)): a disc about
    # (0.3, 0.3) of radius sqrt(0.04 - 0.01); no nodal value of the element is positive
    u = sphere_u([-0.1, 0.3, 0.3], 0.2)
    assert np.all(u < 0)
    rho = math.sqrt(0.04 - 0.01)
    disc = disc_sphere_moments((0.3, 0.3), rho, 4)
    d2 = {tuple(m): disc[i] for i, m in enumerate(cutquadric.monomials2(4))}
    got = np.array(cutquadric.face_moments(u, 1, 4, 24))
    for i, (p, q, r) in enumerate(cutclip.monomials(4)):
        want = d2[(q, r)] if p == 0 else 0.0
        assert got[i] == pytest.approx(want, abs=1e-13)
    # the other three faces are not touched by the ball
    for k in (0, 2, 3):
        assert np.max(np.abs(np.array(cutquadric.face_moments(u, k, 4, 24)))) == 0.0
    assert cutquadric.classify_face(cutquadric.face_nodal(u, 1))["cut"]
    for k in (0, 2, 3):
        assert not cutquadric.classify_face(cutquadric.face_nodal(u, k))["cut"]


# ================================================================================================== A4: surface rule
def sphere_surface_moments(c, r, deg, octant=False, normal_comp=None):
    """integral over the sphere about c (octant: the first octant of the sphere about 0) of x^alpha, or of x^alpha n_i with
    n the inward unit normal (the direction of grad(r^2 - |x - c|^2)), for every monomial of degree <= deg."""
    out = []
    for alpha in cutclip.monomials(deg):
        tot = 0.0
        for beta in itertools.product(*[range(a + 1) for a in alpha]):
            coef = np.prod([math.comb(a, b) * c[i] ** (a - b) for i, (a, b) in enumerate(zip(alpha, beta))])
            beta = list(beta)
            if normal_comp is None:
                tot += coef * r ** (sum(beta) + 2) * gam_sphere(beta, octant)
            else:
                beta[normal_comp] += 1                    # n_i = -y_i / r
                tot += -coef * r ** (sum(beta) + 1) * gam_sphere(beta, octant)
        out.append(tot)
    return np.array(out)


def surf(u, degree, order=16, **kw):
    S, Sn = cutquadric.surface_moments(u, degree, order, **kw)
    return np.array(S), np.array(Sn)


def test_u6_octant_of_a_sphere_closed_forms():
    r = 0.4
    u = sphere_u([0, 0, 0], r)
    P, W, N = cutquadric.surface_rule(u, 16)
    assert np.all(W > 0) and len(W) > 0
    assert np.allclose(np.linalg.norm(N, axis=1), 1.0, atol=1e-14)
    assert np.all(np.linalg.norm(P, axis=1) == pytest.approx(r, abs=1e-12))
    wantS = sphere_surface_moments(np.zeros(3), r, 4, octant=True)
    assert wantS[0] == pytest.approx(np.pi * r * r / 2, rel=1e-14)
    wantN = np.array([sphere_surface_moments(np.zeros(3), r, 4, octant=True, normal_comp=i) for i in range(3)])
    res = {}
    for order in (8, 16, 32):
        S, Sn = surf(u, 4, order)
        res[order] = max(np.max(np.abs(S - wantS)), np.max(np.abs(Sn - wantN)))
    assert res[8] <= 1e-8 and res[16] <= 1e-12 and res[32] <= 1e-12


def test_u6_closed_sphere_inside_the_tet_closed_forms():
    r = 0.15
    u = sphere_u(INCENTRE, r)
    wantS = sphere_surface_moments(INCENTRE, r, 4)
    wantN = np.array([sphere_surface_moments(INCENTRE, r, 4, normal_comp=i) for i in range(3)])
    assert wantS[0] == pytest.approx(4 * np.pi * r * r)
    for order in (16, 24):
        S, Sn = surf(u, 4, order)
        assert np.max(np.abs(S - wantS)) <= 1e-12 and np.max(np.abs(Sn - wantN)) <= 1e-12


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_u6_every_inner_axis(axis):
    for u, want in ((sphere_u([0, 0, 0], 0.4), sphere_surface_moments(np.zeros(3), 0.4, 3, octant=True)),
                    (sphere_u(INCENTRE, 0.15), sphere_surface_moments(INCENTRE, 0.15, 3))):
        S, _ = surf(u, 3, 24, axis=axis)
        assert np.max(np.abs(S - want)) <= 1e-12


def test_u6_axis_choice_cylinder_along_x():
    # u does not depend on x: no x-roots at all, so the inner axis must be chosen by the gradient
    u = nodal(lambda x, y, z: 0.04 - (y - 0.3) ** 2 - (z - 0.3) ** 2)
    assert cutquadric.choose_axis(u) in (1, 2)
    assert len(cutquadric.surface_rule(u, 16, axis=0)[1]) == 0
    S, Sn = surf(u, 3, 24)                                            # automatic
    for axis in (1, 2):
        S1, Sn1 = surf(u, 3, 24, axis=axis)
        assert np.max(np.abs(S - S1)) <= 1e-12 and np.max(np.abs(Sn - Sn1)) <= 1e-12
    assert S[0] > 0.05


def test_u6_planar_surface_equals_cutclip_with_and_without_fast_path():
    rng = np.random.default_rng(55)
    for planar_tol in (1e-12, 0.0):
        worst = 0.0
        for _ in range(30):
            phi = rng.uniform(-1, 1, 4)
            if not (phi.min() < 0 < phi.max()):
                continue
            pl = plane_of(phi)
            want = np.array(cutclip.surface_moments(ref_tet, pl, np.zeros((0, 4)), 4))
            nrm = pl[:3] / np.linalg.norm(pl[:3])
            S, Sn = surf(linear_p2(phi), 4, 24, planar_tol=planar_tol)
            worst = max(worst, np.max(np.abs(S - want)), np.max(np.abs(Sn - nrm[:, None] * want[None, :])))
        assert worst <= 1e-13


def test_u6_physical_scaling():
    rng = np.random.default_rng(56)
    J = 3.0 * np.eye(3) + 0.3 * rng.standard_normal((3, 3))
    Jinv = np.linalg.inv(J)
    x0 = np.array([0.2, -0.1, 0.4])
    c_ref = np.full(3, 0.25)
    c_phys = x0 + J @ c_ref
    r = 0.15
    u = nodal(lambda x, y, z: r * r - np.sum((x0 + J @ np.array([x, y, z]) - c_phys) ** 2))
    S, Sn = surf(u, 1, 24, J=J)
    area, vol = 4 * np.pi * r * r, 4 * np.pi * r ** 3 / 3
    assert S[0] == pytest.approx(area, rel=1e-12)
    assert np.max(np.abs(Sn[:, 0])) <= 1e-12                               # integral of n dS = 0
    # reference coordinate moments with physical weights: int xi dS = Jinv (c - x0) area, int xi_k n_j dS = -V Jinv[k, j]
    assert np.max(np.abs(S[1:4] - Jinv @ (c_phys - x0) * area)) <= 1e-12
    for j in range(3):
        assert np.max(np.abs(Sn[j, 1:4] + vol * Jinv[:, j])) <= 1e-12
    # without J the weights are reference weights and differ
    S0, _ = surf(u, 1, 24)
    assert abs(S0[0] - area) > 1e-3


# the divergence theorem ties volume, face and surface moments together: for q a polynomial, K+ = K cap {u > 0}, whose
# outward normal on Gamma is -n (n = grad u / |grad u|):   int_Gamma q n_i = sum_faces int_{F+} q nu_i - int_{K+} d_i q
FACE_NU = np.array([[1., 1, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])


def divergence_residual(u, degree, order):
    idx = [tuple(m) for m in cutclip.monomials(degree)]
    pos = {m: i for i, m in enumerate(idx)}
    vol = np.array(cutquadric.volume_moments(u, max(degree - 1, 0), order))
    pos_v = {tuple(m): i for i, m in enumerate(cutclip.monomials(max(degree - 1, 0)))}
    faces = sum(FACE_NU[k][:, None] * np.array(cutquadric.face_moments(u, k, degree, order))[None, :] for k in range(4))
    _, Sn = surf(u, degree, order)
    res = np.zeros((3, len(idx)))
    for i, alpha in enumerate(idx):
        for comp in range(3):
            dv = 0.0
            if alpha[comp] > 0:
                beta = list(alpha)
                beta[comp] -= 1
                dv = alpha[comp] * vol[pos_v[tuple(beta)]]
            res[comp, i] = Sn[comp, i] - (faces[comp, i] - dv)
    return np.max(np.abs(res))


def test_u7_divergence_identity_named_cases():
    cases = {
        "octant": sphere_u([0, 0, 0], 0.4),
        "closed": sphere_u(INCENTRE, 0.15),
        "cap": sphere_u([-0.1, 0.3, 0.3], 0.2),
        "generic": generic_sphere_cut(),
        "edge twice": nodal(lambda x, y, z: (x - .5) ** 2 + 0.3 * y - 0.01),
    }
    for name, u in cases.items():
        assert divergence_residual(u, 3, 32) <= 1e-12, name


def test_u7_divergence_identity_random_quadrics_all_labels():
    rng = np.random.default_rng(77)
    seen = {}
    worst = 0.0
    tries = 0
    while tries < 300 and len(seen) < 3:
        tries += 1
        u = rng.standard_normal(10) * rng.choice([0.3, 1.0]) + rng.choice([-0.3, 0.0, 0.3])
        lab = cutquadric.classify(u)["label"]
        if lab.startswith("uncut"):
            continue
        seen.setdefault(lab, 0)
        if seen[lab] >= 6:
            continue
        seen[lab] += 1
        worst = max(worst, divergence_residual(u, 3, 48))
    assert "simple" in seen and "edge_cut_twice" in seen
    assert worst <= 1e-12


# Two random quadrics that exposed the slow cases of the slicing scheme as ported from July: with the cubic smoothstep on
# the outer coordinate and clustering only at the segment ends the divergence residual fell like n^-3 (2e-10 at order 48).
# The causes, each pinned by the converged residual below: (1) a silhouette pair that has become complex but lies close to
# the real axis near a turning point of the silhouette curve (the y-integrand 1/sqrt(disc) is nearly singular inside the run),
# (2) a run truncated just short of a silhouette by a kink (singularity just outside the end), (3) the logarithm F ~ ln(dz)
# at the turning point itself. All three are removed by changing variables so that sqrt(disc) is analytic (integrateRootRun)
# and by double-exponential nodes at the logarithmic ends.
HARD_QUADRICS = [
    [0.7277699642047284, -0.27083755688644556, 2.954460689730097, -1.3085449528642095, 0.9617156616641691,
     0.15657405602100335, -0.054506388471426914, 1.3663588121198411, -1.5179220006075487, -0.6846762100886532],
    [-1.1508209949401316, -0.707676270095551, -0.6026114888058611, 0.40130571326444403, -2.4298908888000788,
     0.9893576133847783, -1.4491045549445354, 0.7378197272566023, -1.559569515328055, -0.5053469074759236],
]


@pytest.mark.parametrize("case", [0, 1])
def test_u5_slow_converging_random_quadrics_regression(case):
    u = np.array(HARD_QUADRICS[case])
    assert cutquadric.classify(u)["cut"]
    assert divergence_residual(u, 3, 24) <= 1e-12
    assert divergence_residual(u, 3, 32) <= 1e-12


def test_u7_all_axes_agree_on_random_quadrics():
    rng = np.random.default_rng(88)
    checked = 0
    for _ in range(60):
        u = rng.standard_normal(10) * rng.choice([0.3, 1.0]) + rng.choice([-0.3, 0.0, 0.3])
        if not cutquadric.classify(u)["cut"]:
            continue
        ref = surf(u, 2, 32, axis=-1)
        for axis in range(3):
            S, Sn = surf(u, 2, 32, axis=axis)
            if len(cutquadric.surface_rule(u, 8, axis=axis)[1]) == 0:
                continue
            # rare near-coincident critical values (three events within 1e-7 of each other) leave an axis at ~1e-8 at
            # order 32; the automatic axis is a heuristic, the divergence tests above use it
            assert np.max(np.abs(S - ref[0])) <= 1e-6 and np.max(np.abs(Sn - ref[1])) <= 1e-6
        checked += 1
    assert checked >= 20


# ================================================================================================== A5: QuadricFit
def collapsed_gauss_tet(n=7):
    """Positive Gauss rule on the reference tet, exact to degree 2n - 1 (collapsed coordinates)."""
    from numpy.polynomial.legendre import leggauss
    x, w = leggauss(n)
    x, w = 0.5 * (x + 1), 0.5 * w
    P, W = [], []
    for a, wa in zip(x, w):
        for b, wb in zip(x, w):
            for c, wc in zip(x, w):
                l1, l2, l3 = a, b * (1 - a), c * (1 - a) * (1 - b)
                P.append((l1, l2, l3))
                W.append(wa * wb * wc * (1 - a) ** 2 * (1 - b))
    return np.array(P), np.array(W)


QX, QW = collapsed_gauss_tet(7)


def mono_matrix(deg, pts=QX):
    return np.array([pts[:, 0] ** p * pts[:, 1] ** q * pts[:, 2] ** r for p, q, r in cutclip.monomials(deg)])


def fit_moments(fit, deg):
    """sum_q w_q f(x_q) x_q^alpha for the four fits."""
    M = mono_matrix(deg)
    H, D = fit.H(QX), fit.D(QX)
    out = {"H": M @ (QW * H), "ImH": M @ (QW * fit.ImH(QX)), "D": M @ (QW * D)}
    Dn = fit.Dn(QX)
    out["Dn"] = np.array([M @ (QW * Dn[:, i]) for i in range(3)])
    return out


def test_a5_gram_matrix_is_exact():
    # the monomial Gram matrix of the reference tet: int x^a y^b z^c = a! b! c! / (a+b+c+3)!
    M = mono_matrix(4)
    G = (M * QW) @ M.T
    mono = cutclip.monomials(4)
    G_exact = np.array([[math.factorial(a[0] + b[0]) * math.factorial(a[1] + b[1]) * math.factorial(a[2] + b[2])
                         / math.factorial(sum(a) + sum(b) + 3) for b in mono] for a in mono])
    assert np.max(np.abs(G - G_exact)) <= 1e-15


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_u9_fits_reproduce_the_moments_for_every_polynomial_of_degree_k(k):
    rng = np.random.default_rng(90 + k)
    cases = [sphere_u([0, 0, 0], 0.4), sphere_u(INCENTRE, 0.15), sphere_u([-0.1, 0.3, 0.3], 0.2), generic_sphere_cut(),
             linear_p2([0.31, -0.62, 0.45, -0.27]), linear_p2([0.5, 0.2, -0.3, -0.9])]
    while len(cases) < 14:
        u = rng.standard_normal(10) * rng.choice([0.3, 1.0]) + rng.choice([-0.3, 0.0, 0.3])
        if cutquadric.classify(u)["cut"]:
            cases.append(u)
    full = full_moments(k)
    for u in cases:
        fit = cutquadric.QuadricFit(u, k, 24)
        vol = np.array(cutquadric.volume_moments(u, k, 24))
        S, Sn = surf(u, k, 24)
        got = fit_moments(fit, k)
        scale = max(1.0, np.max(np.abs(vol)))
        assert np.max(np.abs(got["H"] - vol)) <= 1e-11 * scale
        assert np.max(np.abs(got["ImH"] - (full - vol))) <= 1e-11 * scale
        assert np.max(np.abs(got["D"] - S)) <= 1e-11 * max(1.0, np.max(np.abs(S)))
        assert np.max(np.abs(got["Dn"] - Sn)) <= 1e-11 * max(1.0, np.max(np.abs(Sn)))


def test_u9_fits_against_closed_forms_and_the_documented_miss():
    r, k = 0.4, 3
    fit = cutquadric.QuadricFit(sphere_u([0, 0, 0], r), k, 24)
    got = fit_moments(fit, k + 1)
    wantV = ball_volume_moments(np.zeros(3), r, k + 1, octant=True)
    wantS = sphere_surface_moments(np.zeros(3), r, k + 1, octant=True)
    wantN = np.array([sphere_surface_moments(np.zeros(3), r, k + 1, octant=True, normal_comp=i) for i in range(3)])
    deg = np.array([sum(m) for m in cutclip.monomials(k + 1)])
    ok = deg <= k
    assert np.max(np.abs(got["H"] - wantV)[ok]) <= 1e-12
    assert np.max(np.abs(got["D"] - wantS)[ok]) <= 1e-12
    assert np.max(np.abs(got["Dn"] - wantN)[:, ok]) <= 1e-12
    # a degree-(k+1) integrand is outside the fit: the miss is real
    assert np.max(np.abs(got["H"] - wantV)[~ok]) > 1e-7
    assert np.max(np.abs(got["D"] - wantS)[~ok]) > 1e-7


def test_u9_planar_data_reproduces_the_equivalent_polynomial_fits():
    from proteus import equivalent_polynomials as eqp
    k = 3
    X3, _ = collapsed_gauss_tet(3)                    # eqp.Simplex holds at most 50 points
    gf = eqp.Simplex(nSpace=3, nP=k, nQ=len(X3))
    for phi in ([-0.3, 0.8, 0.5, -0.4], [-0.3, 0.8, 0.5, 0.4], [0.9, -0.2, -0.5, -0.7], [0.31, -0.62, 0.45, -0.27]):
        gf.calculate(np.array(phi), ref_tet, X3)
        fit = cutquadric.QuadricFit(linear_p2(phi), k, 16)
        H, D, ImH = fit.H(X3), fit.D(X3), fit.ImH(X3)
        for q in range(len(X3)):
            gf.set_quad(q)
            assert H[q] == pytest.approx(gf.H, abs=1e-9)
            assert ImH[q] == pytest.approx(gf.ImH, abs=1e-9)
            assert D[q] == pytest.approx(gf.D, abs=1e-9)


def test_u9_physical_scaling_of_the_dirac_fits():
    rng = np.random.default_rng(91)
    J = 3.0 * np.eye(3) + 0.3 * rng.standard_normal((3, 3))
    x0 = np.array([0.2, -0.1, 0.4])
    r = 0.15
    c_phys = x0 + J @ np.full(3, 0.25)
    u = nodal(lambda x, y, z: r * r - np.sum((x0 + J @ np.array([x, y, z]) - c_phys) ** 2))
    k = 3
    fit = cutquadric.QuadricFit(u, k, 24, J=J)
    detJ = abs(np.linalg.det(J))
    M = mono_matrix(0)
    # the kernel multiplies each weight by |J|: |J| sum_q w_q D(x_q) = physical area, |J| sum w Dn dS = physical int n dS = 0
    assert detJ * (QW @ fit.D(QX)) == pytest.approx(4 * np.pi * r * r, rel=1e-11)
    assert np.max(np.abs(detJ * (QW @ fit.Dn(QX)))) <= 1e-11
    vol = 4 * np.pi * r ** 3 / 3
    assert detJ * (QW @ fit.H(QX)) == pytest.approx(vol, rel=1e-11)
    # int n_j x_k dS = -V delta_jk physical; in reference coordinates xi: int xi_k n_j dS = -V Jinv[k, j]
    Jinv = np.linalg.inv(J)
    Xi = QX
    for j in range(3):
        for kk in range(3):
            val = detJ * np.sum(QW * fit.Dn(QX)[:, j] * Xi[:, kk])
            assert val == pytest.approx(-vol * Jinv[kk, j], abs=1e-11)
