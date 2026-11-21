"""C44 phase 3, part A: a tetrahedron cut by two P1 level sets (proteus/cutpair.h).

phi_s > 0 is the fluid, phi_s < 0 the solid; phi_f < 0 is the A side, phi_f > 0 the B side.
A = {phi_s > 0, phi_f < 0}, B = {phi_s > 0, phi_f > 0}, S = {phi_s < 0}.

The oracle below does not use cutclip: it enumerates the vertices of each convex region from all triples of planes
(as the July reference does), tetrahedralizes with scipy, and integrates with its own collapsed Gauss rule.

U1  partition: A + B + S = K and the split surfaces add up to the unsplit ones, every monomial of degree <= 4
U2  regions, surface pieces and the triple line against the oracle; volumes against a brute-force grid
U3  parity with the July P1 reference (golden values)
U4  fits: the degree-k fit of each region reproduces its moments to degree k, and misses at k + 1
U5  split Dirac fits, physical measure and normals on an affine element; divergence theorem on A, B and S
U6  triple line: end points, the faces they lie on, the face triple points, length, the line fit
U7  faces: the face regions against clipping the face in 3D, and against the 2D kernel's composite fit
U8  reductions and degenerate cuts: equal, opposite and parallel level sets, cuts through a vertex, an edge, a face
U9  vertex permutation: volumes, physical areas and the physical length do not depend on the local numbering
"""
import itertools

import numpy as np
import pytest

from proteus import cutclip, cutpair

REF = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
# lambda_i >= 0; face i is {lambda_i = 0}
TET_PLANES = np.array([[-1., -1., -1., 1.], [1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]])
DEG = 4
IDX = np.array(cutpair.monomials(DEG))
NO_PLANES = np.zeros((0, 4))
NO_POINTS = np.zeros((0, 3))


def plane(phi):
    phi = np.asarray(phi, float)
    return np.array([phi[1] - phi[0], phi[2] - phi[0], phi[3] - phi[0], phi[0]])


def mono(x, idx=IDX):
    x = np.atleast_2d(x)
    return np.prod(x[:, None, :] ** idx[None, :, :], axis=2)


def gauss01(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def tet_rule(n=6):
    """Collapsed Gauss rule on the unit tet (weights sum to 1/6), exact to degree 2n - 3."""
    x, w = gauss01(n)
    u, v, s = np.meshgrid(x, x, x, indexing="ij")
    wu, wv, ws = np.meshgrid(w, w, w, indexing="ij")
    l1, l2, l3 = u, v * (1 - u), s * (1 - u) * (1 - v)
    bary = np.stack([1 - l1 - l2 - l3, l1, l2, l3], axis=-1).reshape(-1, 4)
    return bary, (wu * wv * ws * (1 - u) ** 2 * (1 - v)).ravel()


def tri_rule(n=6):
    x, w = gauss01(n)
    u, v = np.meshgrid(x, x, indexing="ij")
    wu, wv = np.meshgrid(w, w, indexing="ij")
    l1, l2 = u, v * (1 - u)
    return np.stack([1 - l1 - l2, l1, l2], axis=-1).reshape(-1, 3), (wu * wv * (1 - u)).ravel()


TB, TW = tet_rule()
FB, FW = tri_rule()
XQ = TB @ REF  # points of the degree-9 rule on the reference tet


# ------------------------------------------------------------------------------------------------------- the oracle
def polytope_vertices(planes):
    """Vertices of {x : a.x + b >= 0 for every plane}, from all triples of planes."""
    P = np.asarray(planes, float)
    P = P / np.abs(P).max(axis=1, keepdims=True)
    verts = []
    for tr in itertools.combinations(range(len(P)), 3):
        A = P[list(tr), :3]
        if abs(np.linalg.det(A)) < 1e-10:
            continue
        x = np.linalg.solve(A, -P[list(tr), 3])
        if (P[:, :3] @ x + P[:, 3] >= -1e-11).all() and not any(np.abs(x - q).max() < 1e-9 for q in verts):
            verts.append(x)
    return np.array(verts).reshape(-1, 3), P


def volume_moments(planes):
    from scipy.spatial import Delaunay, QhullError
    V, _ = polytope_vertices(np.vstack([TET_PLANES, planes]))
    m = np.zeros(len(IDX))
    if len(V) < 4:
        return m
    try:
        tri = Delaunay(V)
    except QhullError:
        return m
    for s in tri.simplices:
        v = V[s]
        m += abs(np.linalg.det(v[1:] - v[0])) * (TW @ mono(TB @ v))
    return m


def polygon_on(planes, k):
    """Triangles (fan) of the face of the polytope lying on plane k."""
    V, P = polytope_vertices(planes)
    on = [x for x in V if abs(P[k, :3] @ x + P[k, 3]) < 1e-9]
    if len(on) < 3:
        return []
    on = np.array(on)
    c = on.mean(axis=0)
    n = P[k, :3] / np.linalg.norm(P[k, :3])
    t1 = np.cross(n, [1., 0., 0.]) if abs(n[0]) < 0.9 else np.cross(n, [0., 1., 0.])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    on = on[np.argsort(np.arctan2((on - c) @ t2, (on - c) @ t1))]
    return [np.array([on[0], on[i], on[i + 1]]) for i in range(1, len(on) - 1)]


def surface_moments(tris, X=None):
    """Monomials of the reference point, measure = area of the triangles after the affine map X (reference if None)."""
    m = np.zeros(len(IDX))
    for t in tris:
        p = t if X is None else to_physical(t, X)
        m += np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0])) * (FW @ mono(FB @ t))
    return m


def to_physical(xi, X):
    return X[0] + xi @ (X[1:] - X[0])


def pieces(phi_s, phi_f):
    hs, hf = plane(phi_s), plane(phi_f)
    base = list(TET_PLANES)
    return {"A": volume_moments([hs, -hf]), "B": volume_moments([hs, hf]), "S": volume_moments([-hs]),
            "GsA": polygon_on(base + [hs, -hf], 4), "GsB": polygon_on(base + [hs, hf], 4),
            "GfActive": polygon_on(base + [hf, hs], 4), "GfMasked": polygon_on(base + [hf, -hs], 4)}


def line_ends(phi_s, phi_f):
    V, P = polytope_vertices(np.vstack([TET_PLANES, plane(phi_s), -plane(phi_s), plane(phi_f), -plane(phi_f)]))
    return V


def random_pair(rng, both_cut=True):
    while True:
        ps, pf = rng.uniform(-1, 1, 4), rng.uniform(-1, 1, 4)
        if not both_cut or (cutpair.side(ps) == 0 and cutpair.side(pf) == 0):
            return ps, pf


def random_tet(rng):
    while True:
        X = REF + 0.35 * rng.standard_normal((4, 3))
        if abs(np.linalg.det(X[1:] - X[0])) > 0.2:
            return 1.7 * X


def jac(X):
    return (X[1:] - X[0]).T  # J[I, i] = d x_I / d xi_i


FULL = np.array(cutclip.tet_moments(REF, NO_PLANES, DEG))
# fixed pairs: a triple line through the element, both cuts without a common line, the interface inside the solid
CASES = {"triple": ([0.7, -0.2, 0.5, -0.6], [-0.4, 0.6, 0.3, -0.5]),
         "disjoint": ([-1., 0.2, 0.2, 0.2], [0.9, 0.9, 0.9, -0.1]),
         "buried": ([0.8, -0.2, -0.2, -0.2], [0.9, -0.05, 0.9, 0.9])}


# ------------------------------------------------------------------------------------------------------ U1 partition
@pytest.mark.parametrize("seed", range(4))
def test_u1_partition(seed):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(50):
        ps, pf = random_pair(rng, both_cut=False)
        m = {k: np.array(v) for k, v in cutpair.moments(ps, pf, DEG).items()}
        worst = max(worst, np.abs(m["A"] + m["B"] + m["S"] - FULL).max())
        gs = np.array(cutclip.surface_moments(REF, plane(ps), NO_PLANES, DEG))
        gf = np.array(cutclip.surface_moments(REF, plane(pf), NO_PLANES, DEG))
        worst = max(worst, np.abs(m["GsA"] + m["GsB"] - gs).max(), np.abs(m["GfActive"] + m["GfMasked"] - gf).max())
    assert worst < 1e-14


# --------------------------------------------------------------------------------------------------------- U2 oracle
@pytest.mark.parametrize("seed", range(4))
def test_u2_regions_and_surfaces_against_vertex_enumeration(seed):
    rng = np.random.default_rng(100 + seed)
    for _ in range(25):
        ps, pf = random_pair(rng)
        m = cutpair.moments(ps, pf, DEG)
        ref = pieces(ps, pf)
        for k in ("A", "B", "S"):
            assert np.abs(np.array(m[k]) - ref[k]).max() < 1e-13, k
        for k in ("GsA", "GsB", "GfActive", "GfMasked"):
            assert np.abs(np.array(m[k]) - surface_moments(ref[k])).max() < 1e-13, k


@pytest.mark.parametrize("name", list(CASES))
def test_u2_volumes_against_brute_force(name):
    ps, pf = (np.array(v) for v in CASES[name])
    # one random point per cell of an n^3 grid: cell centres are biased for planes aligned with the grid diagonal
    n = 160
    c = np.arange(n) / n
    rng = np.random.default_rng(160)
    x, y, z = (g + rng.uniform(0, 1 / n, g.shape) for g in np.meshgrid(c, c, c, indexing="ij"))
    inside = x + y + z < 1.0
    lam = np.stack([1 - x - y - z, x, y, z], axis=-1)[inside]
    s, f = lam @ ps, lam @ pf
    brute = {"A": np.sum((s > 0) & (f < 0)), "B": np.sum((s > 0) & (f > 0)), "S": np.sum(s < 0)}
    fit = cutpair.fit(ps, pf, 1, None, NO_POINTS)
    for k, key in (("A", "vol_a"), ("B", "vol_b"), ("S", "vol_s")):
        assert fit[key] == pytest.approx(brute[k] / n ** 3, abs=2e-4), k


def test_u2_case_labels():
    f = {k: cutpair.fit(*CASES[k], 1, None, NO_POINTS) for k in CASES}
    assert f["triple"]["line"]["exists"] and f["triple"]["demoted"] == 0
    assert not f["disjoint"]["line"]["exists"] and f["disjoint"]["demoted"] == 0
    assert min(f["disjoint"]["vol_a"], f["disjoint"]["vol_b"], f["disjoint"]["vol_s"]) > 1e-4
    # the interface crosses the element inside the solid only: the fluid is all B
    assert f["buried"]["side_f"] == 0 and f["buried"]["vol_a"] == 0.0 and f["buried"]["demoted"] == 1
    g = cutpair.fit(CASES["buried"][0], -np.array(CASES["buried"][1]), 1, None, NO_POINTS)
    assert g["vol_b"] == 0.0 and g["demoted"] == -1


# ------------------------------------------------------------------------------------------- U3 July golden values
# Integral of the P1 function w over A, B and S in the unit tet with phi_s = u, phi_f = v, from the July reference
# p1_two_level_sets_tet.py (integrate_over_uv_positive(u, -v, w), (u, v, w), integrate_over_positive(-u, w)), run on
# 2026-11-14. The first six (u, v, w) are the cases of test_cutclip.py.
JULY_U = [[1, -1, -1, -1], [1, 1, -1, -1], [-0.3, 0.9, 0.5, -0.2], [0.4, -0.7, 0.8, 0.1], [1, 1, 1, -1],
          [-1, 0.2, 0.2, 0.2], [0.7, -0.2, 0.5, -0.6], [0.25, 0.5, -0.75, 1.0]]
JULY_V = [[-1, 1, 1, 1], [-1, 1, 1, -1], [0.6, -0.4, 0.3, 0.5], [-0.5, 0.5, -0.1, 0.9], [-1, -1, 1, 1],
          [0.3, -0.8, 0.4, 0.6], [-0.4, 0.6, 0.3, -0.5], [-0.5, 1.0, 0.25, -0.25]]
JULY_W = [[1, 1, 1, 1], [1, 2, 3, 4], [0.5, -1, 2, 0.25], [2, -1, 0.5, 1.5], [1, 0, 0, 0], [0, 1, -1, 1],
          [1, 1, 1, 1], [0.3, 0.7, -1.1, 2.0]]
JULY_A = [0.02083333333333333, 0.08333333333333331, -0.00478374065675653, 0.04075431715697747, 0.02864583333333334,
          0.01316222406958325, 0.051227736844299786, 0.03814341226712625]
JULY_B = [0.0, 0.09375000000000001, 0.06419984390719925, 0.07562097697754958, 0.010416666666666666,
          0.008410563996260372, 0.057792872176309254, 0.048588305419948574]
JULY_S = [0.14583333333333331, 0.23958333333333337, 0.013500563416223924, 0.008624705865472908,
          0.002604166666666666, 0.020093878600823047, 0.05764605764605765, -0.007565051020408165]


@pytest.mark.parametrize("i", range(8))
def test_u3_july_parity_golden_values(i):
    m = cutpair.moments(JULY_U[i], JULY_V[i], 1)
    w = np.array(JULY_W[i], float)
    lin = np.array([w[0], w[1] - w[0], w[2] - w[0], w[3] - w[0]])  # 1, x, y, z in monomial order
    assert cutpair.monomials(1) == [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for key, gold in (("A", JULY_A), ("B", JULY_B), ("S", JULY_S)):
        assert lin @ np.array(m[key]) == pytest.approx(gold[i], abs=1e-15), key


# ----------------------------------------------------------------------------------------------------------- U4 fits
@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_u4_region_fits_reproduce_moments(k):
    rng = np.random.default_rng(7)
    idx_k, idx_k1 = np.array(cutpair.monomials(k)), np.array(cutpair.monomials(k + 1))
    worst, miss = 0.0, 0.0
    for _ in range(20):
        ps, pf = random_pair(rng)
        f = cutpair.fit(ps, pf, k, None, XQ)
        m = cutpair.moments(ps, pf, k + 1)
        for key in ("A", "B"):
            got = (TW * f[key]) @ mono(XQ, idx_k1)
            ref = np.array(m[key])
            worst = max(worst, np.abs(got - ref)[:len(idx_k)].max())
            miss = max(miss, np.abs(got - ref)[len(idx_k):].max())
    assert worst < 1e-12
    assert miss > 1e-7  # degree k + 1 is not reproduced: the check above is not vacuous


# ------------------------------------------------------------------------------- U5 split Diracs, physical measure
@pytest.mark.parametrize("seed", range(3))
def test_u5_dirac_fits_physical_measure_and_normals(seed):
    rng = np.random.default_rng(200 + seed)
    k = 3
    idx_k = np.array(cutpair.monomials(k))
    for _ in range(12):
        ps, pf = random_pair(rng)
        X = random_tet(rng)
        detJ = abs(np.linalg.det(jac(X)))
        f = cutpair.fit(ps, pf, k, jac(X), XQ)
        ref = pieces(ps, pf)
        for key in ("GsA", "GsB", "GfActive", "GfMasked"):
            got = detJ * (TW * f["D" + key[1:]]) @ mono(XQ, idx_k)
            want = surface_moments(ref[key], X)[:len(idx_k)]
            assert np.abs(got - want).max() < 1e-11, key
        # the physical gradient of each level set gives the normal
        for phi, key in ((ps, "normal_s"), (pf, "normal_f")):
            g = np.linalg.solve(np.hstack([X, np.ones((4, 1))]), phi)[:3]
            assert np.abs(np.array(f[key]) - g / np.linalg.norm(g)).max() < 1e-12
        # reference measure: kappa = 1, the fit of the split pieces adds up to the fit of the whole surface
        f0 = cutpair.fit(ps, pf, k, None, NO_POINTS)
        assert f0["kappa_s"] == pytest.approx(1.0, abs=1e-15) and f0["kappa_f"] == pytest.approx(1.0, abs=1e-15)


@pytest.mark.parametrize("seed", range(3))
def test_u5_divergence_theorem_on_each_region(seed):
    """int over the boundary of R of q n_i = int_R d_i q, with the boundary assembled from the clipped tet faces and
    the split cut surfaces: ties the volumes, the surface pieces and the normal directions together."""
    rng = np.random.default_rng(300 + seed)
    out = np.array([[1., 1., 1.], [-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])  # area-weighted: |n| = 2 area / (ds dt)
    out_unit = out / np.linalg.norm(out, axis=1, keepdims=True)
    worst = 0.0
    for _ in range(15):
        ps, pf = random_pair(rng)
        hs, hf = plane(ps), plane(pf)
        ns, nf = hs[:3] / np.linalg.norm(hs[:3]), hf[:3] / np.linalg.norm(hf[:3])
        m = {k: np.array(v) for k, v in cutpair.moments(ps, pf, DEG).items()}
        regions = {"A": ([hs, -hf], [("GsA", -ns), ("GfActive", nf)]),
                   "B": ([hs, hf], [("GsB", -ns), ("GfActive", -nf)]),
                   "S": ([-hs], [("GsA", ns), ("GsB", ns)])}
        for name, (planes, surfs) in regions.items():
            flux = np.zeros((len(IDX), 3))
            for kf in range(4):
                tri = REF[[v for v in range(4) if v != kf]]
                flux += np.outer(cutclip.tri_moments(tri, np.array(planes), DEG), out_unit[kf])
            for key, n in surfs:
                flux += np.outer(m[key], n)
            for i in range(3):
                for a, (p, q, r) in enumerate(IDX):
                    e = [p, q, r]
                    if e[i] == 0:
                        want = 0.0
                    else:
                        c = e[i]
                        e[i] -= 1
                        want = c * m[name][[tuple(row) for row in IDX].index(tuple(e))]
                    if p + q + r <= DEG:
                        worst = max(worst, abs(flux[a, i] - want))
    assert worst < 1e-14


# ---------------------------------------------------------------------------------------------------- U6 triple line
def face_point(kf, st):
    v = REF[[m for m in range(4) if m != kf]]
    return v[0] + st[0] * (v[1] - v[0]) + st[1] * (v[2] - v[0])


@pytest.mark.parametrize("seed", range(4))
def test_u6_triple_line(seed):
    rng = np.random.default_rng(400 + seed)
    k = 3
    idx_k = np.array(cutpair.monomials(k))
    seen = 0
    for _ in range(40):
        ps, pf = random_pair(rng)
        L = cutpair.triple_line(ps, pf)
        ends = line_ends(ps, pf)
        assert L["exists"] == (len(ends) == 2)
        faces_with_point = [kf for kf in range(4)
                            if cutpair.face_pair(cutpair.face_nodal(ps, kf), cutpair.face_nodal(pf, kf), 1,
                                                 np.zeros((0, 2)))["has_triple_point"]]
        if not L["exists"]:
            assert faces_with_point == []
            continue
        seen += 1
        x0, x1 = np.array(L["x0"]), np.array(L["x1"])
        lam = lambda x: np.array([1 - x.sum(), *x])
        for x, kf in ((x0, L["face0"]), (x1, L["face1"])):
            assert abs(lam(x) @ ps) < 1e-13 and abs(lam(x) @ pf) < 1e-13   # on both cut planes
            assert abs(lam(x)[kf]) < 1e-13 and lam(x).min() > -1e-13          # on face kf of K
            assert min(np.abs(x - e).max() for e in ends) < 1e-11             # the oracle's vertex
            fp = cutpair.face_pair(cutpair.face_nodal(ps, kf), cutpair.face_nodal(pf, kf), 1, np.zeros((0, 2)))
            assert fp["has_triple_point"]
            assert np.abs(face_point(kf, fp["triple_point"]) - x).max() < 1e-12  # that face's own triple point
        assert sorted(faces_with_point) == sorted([L["face0"], L["face1"]])
        assert L["length"] == pytest.approx(np.linalg.norm(ends[1] - ends[0]), abs=1e-12)
        assert np.abs(np.cross(L["tangent"], np.cross(plane(ps)[:3], plane(pf)[:3]))).max() < 1e-13
        # line moments (8-point Gauss here) and the line fit with the physical length on an affine element
        g, w = gauss01(8)
        pts = x0 + g[:, None] * (x1 - x0)
        X = random_tet(rng)
        f = cutpair.fit(ps, pf, k, jac(X), XQ)
        phys = np.linalg.norm(to_physical(x1, X) - to_physical(x0, X))
        want = phys * (w @ mono(pts, idx_k))
        got = abs(np.linalg.det(jac(X))) * (TW * f["L"]) @ mono(XQ, idx_k)
        assert np.abs(got - want).max() < 1e-11
        assert np.abs(np.array(cutpair.moments(ps, pf, k)["L"]) - L["length"] * (w @ mono(pts, idx_k))).max() < 1e-14
        t = (to_physical(x1, X) - to_physical(x0, X)) / phys
        assert np.abs(np.array(f["tangent_l"]) - t).max() < 1e-12
    assert seen >= 5


# ------------------------------------------------------------------------------------------------------------ U7 faces
@pytest.mark.parametrize("seed", range(3))
def test_u7_face_regions_against_3d_clipping(seed):
    rng = np.random.default_rng(500 + seed)
    idx2 = cutpair.monomials2(3)
    for _ in range(15):
        ps, pf = random_pair(rng)
        hs, hf = plane(ps), plane(pf)
        for kf in range(4):
            v = REF[[m for m in range(4) if m != kf]]
            scale = np.linalg.norm(np.cross(v[1] - v[0], v[2] - v[0]))  # dS = scale ds dt
            fp = cutpair.face_pair(cutpair.face_nodal(ps, kf), cutpair.face_nodal(pf, kf), 3, np.zeros((0, 2)))
            assert fp["area_a"] + fp["area_b"] + fp["area_s"] == pytest.approx(0.5, abs=1e-15)
            for key, planes in (("area_a", [hs, -hf]), ("area_b", [hs, hf]), ("area_s", [-hs])):
                assert fp[key] * scale == pytest.approx(cutclip.tri_moments(v, np.array(planes), 0)[0], abs=1e-14)
            # the face fit reproduces the face moments (ds dt) to degree 3
            st = FB[:, 1:]
            for key in ("A", "B"):
                c = cutpair.face_pair(cutpair.face_nodal(ps, kf), cutpair.face_nodal(pf, kf), 3, st)[key]
                got = np.array([(FW * c) @ (st[:, 0] ** p * st[:, 1] ** q) for p, q in idx2])
                assert np.abs(got - np.array(fp["m" + key])).max() < 1e-13


@pytest.mark.parametrize("nP", [2, 3, 4])
def test_u7_face_fit_equals_the_2d_composite_fit(nP):
    """The face fit is the fit the 2D ADR kernel uses on a triply cut triangle (composite::weights<nP>).

    Measured over 1100 faces per degree: 9e-14, 2e-12, 2e-10 for nP = 2, 3, 4. The difference is the 2D code's
    tabulated inverse Gram matrix (its fit reproduces the moments to 3e-12 at nP = 4, the Cholesky solve to 5e-15)."""
    rng = np.random.default_rng(600 + nP)
    st = FB[:, 1:]
    worst, n = 0.0, 0
    for _ in range(40):
        ps, pf = rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3)
        if cutpair.side(ps) != 0 or cutpair.side(pf) != 0:
            continue
        two = cutpair.composite2d(ps, pf, nP, st)
        assert two["valid"]
        fp = cutpair.face_pair(ps, pf, nP, st)
        worst = max(worst, np.abs(fp["A"] - two["A"]).max(), np.abs(fp["B"] - two["B"]).max())
        n += 1
    assert n >= 10
    assert worst < 1e-9


# ------------------------------------------------------------------------------------- U8 reductions and degeneracy
def test_u8_reductions():
    rng = np.random.default_rng(8)
    for _ in range(20):
        ps, pf = random_pair(rng)
        one = np.array(cutclip.tet_moments(REF, plane(ps)[None, :], DEG))
        same = {k: np.array(v) for k, v in cutpair.moments(ps, ps, DEG).items()}
        assert np.abs(same["B"] - one).max() < 1e-15 and np.abs(same["A"]).max() < 1e-15
        opp = {k: np.array(v) for k, v in cutpair.moments(ps, -ps, DEG).items()}
        assert np.abs(opp["A"] - one).max() < 1e-15 and np.abs(opp["B"]).max() < 1e-15
        # swapping the roles of the two level sets keeps {both positive}
        assert np.abs(np.array(cutpair.moments(pf, ps, DEG)["B"]) - np.array(cutpair.moments(ps, pf, DEG)["B"])).max() < 1e-15
        # parallel planes: no triple line
        assert not cutpair.triple_line(ps, ps + 0.1)["exists"]
        assert not cutpair.triple_line(ps, 3.0 * ps)["exists"]


DEGENERATE = {
    # both planes contain the line from v0 along (1, 2, 3), which leaves through face 0 at (1, 2, 3) / 6
    "line through a vertex": ([0., 0.5, 0.5, -0.5], [0., 0.4, -0.8, 0.4]),
    "line along an edge": ([0., 0., 0.5, -0.5], [0., 0., -0.3, -0.6]),
    "solid plane on a face": ([0., 0., 0., 1.], [0.5, -0.5, 0.2, 0.3]),
    "both through a vertex and an edge": ([0., 0., 0.4, -0.7], [0., 0.6, -0.2, 0.5]),
    "interface on a face, solid cut": ([0.3, -0.6, 0.2, 0.5], [0., 0., 0., -1.]),
    "coincident": ([0.3, -0.6, 0.2, 0.5], [0.3, -0.6, 0.2, 0.5]),
    "uncut by both": ([1., 2., 3., 4.], [-1., -2., -3., -4.]),
}


@pytest.mark.parametrize("name", list(DEGENERATE))
def test_u8_degenerate_cuts(name):
    ps, pf = (np.array(v) for v in DEGENERATE[name])
    m = {k: np.array(v) for k, v in cutpair.moments(ps, pf, DEG).items()}
    assert np.abs(m["A"] + m["B"] + m["S"] - FULL).max() < 1e-15
    ref = pieces(ps, pf)
    for k in ("A", "B", "S"):
        assert np.abs(m[k] - ref[k]).max() < 1e-13, k
    f = cutpair.fit(ps, pf, 3, jac(random_tet(np.random.default_rng(1))), XQ)
    for key in ("A", "B", "DsA", "DsB", "DfActive", "DfMasked", "L"):
        assert np.isfinite(f[key]).all(), key
    L = cutpair.triple_line(ps, pf)
    if name == "line through a vertex":
        assert L["exists"] and L["length"] == pytest.approx(np.sqrt(14.) / 6., abs=1e-15)
        assert min(np.abs(np.array(L["x0"])).max(), np.abs(np.array(L["x1"])).max()) < 1e-15
    if name == "line along an edge":
        assert L["exists"] and L["length"] == pytest.approx(1.0, abs=1e-15)   # the edge (v0, v1)
    if name in ("coincident", "uncut by both"):
        assert not L["exists"]


# ------------------------------------------------------------------------------------------------- U9 permutations
@pytest.mark.parametrize("seed", range(2))
def test_u9_vertex_permutation(seed):
    rng = np.random.default_rng(900 + seed)
    for _ in range(6):
        ps, pf = random_pair(rng)
        X = random_tet(rng)

        def measures(perm):
            p = list(perm)
            Xp = X[p]
            f = cutpair.fit(ps[p], pf[p], 2, jac(Xp), XQ)
            d = abs(np.linalg.det(jac(Xp)))
            out = [f["vol_a"] * d, f["vol_b"] * d, f["vol_s"] * d]
            out += [d * (TW @ f[key]) for key in ("DsA", "DsB", "DfActive", "DfMasked", "L")]  # areas, length
            return np.array(out)

        base = measures(range(4))
        for perm in itertools.permutations(range(4)):
            assert np.abs(measures(perm) - base).max() < 1e-12
