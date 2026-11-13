#!/usr/bin/env python
"""The two-sided P1 IFEM basis on tetrahedra (C44 phase 2, piece P1).

The object under test is the real ``equivalent_polynomials::Simplex`` with the IFEM basis (the one ADR.h holds
per element), reached through ``proteus.cutclip.ifem_basis_3d`` / ``ifem_basis_2d``. The oracles are written here
and share no code with the kernel:

* ``pointwise_basis``: the defining conditions solved as a linear system, as IFEM_3D.ipynb does (nodal values,
  continuity at d cut points, one flux condition), for any dimension;
* ``closed_form``: the formula of the C44 plan, section 2.1, for any dimension, so the 2D construction that ADR
  already uses (the 6x6 solve of ifemBasisCoefficients_wrapper.h) is checked against the same formula;
* the 32 coefficients IFEM_3D.ipynb prints for x0 = y0 = z0 = 0.45, mu_a = 2, mu_b = 1.

What is checked: nodal values, partition of unity, continuity on the cut, zero flux jump, agreement of the three
constructions to rounding for random tetrahedra, node orderings and orientations, all three splits, contrasts
1e-3 .. 1e3; the unit flux-jump lift; the singular-basis guard (obtuse tetrahedron at the critical contrast).
"""
import itertools
import math

import numpy as np
import pytest

cutclip = pytest.importorskip("proteus.cutclip")

# IFEM_3D.ipynb, cell 5 output (15 significant digits): vertex 0 on side a (mu_a = 2), vertices 1-3 on side b
# (mu_b = 1), interface x/x0 + y/y0 + z/z0 = 1 with x0 = y0 = z0 = 0.45, nodal values e_i. Rows: basis function i;
# columns: coefficients of 1, x, y, z.
NB_A = np.array([
    [1.00000000000000, -0.645161290322581, -0.645161290322581, -0.645161290322581],
    [0.0, 0.881720430107527, -0.118279569892473, -0.118279569892473],
    [0.0, -0.118279569892473, 0.881720430107527, -0.118279569892473],
    [0.0, -0.118279569892473, -0.118279569892473, 0.881720430107527]])
NB_B = np.array([
    [1.29032258064516, -1.29032258064516, -1.29032258064516, -1.29032258064516],
    [-0.0967741935483871, 1.09677419354839, 0.0967741935483871, 0.0967741935483871],
    [-0.0967741935483871, 0.0967741935483871, 1.09677419354839, 0.0967741935483871],
    [-0.0967741935483871, 0.0967741935483871, 0.0967741935483871, 1.09677419354839]])

REF_VERTICES = {2: np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float),
                3: np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)}


def bary_gradients(X):
    """Physical gradients of the barycentric coordinates of the simplex with vertices X (d+1, d): (d+1, d)."""
    d = X.shape[1]
    A = np.hstack([np.ones((d + 1, 1)), X])
    return np.linalg.inv(A)[1:, :].T


def closed_form(X, phi, ma, mb):
    """Nodal values (Vm, Vp) of the two pieces of every basis function (row i = function i), plan section 2.1."""
    G = bary_gradients(X)
    gphi = phi @ G
    psi = np.where(phi > 0, phi, 0.0)
    gpsi = psi @ G
    gg = gphi @ gphi
    D = mb * (gg - gpsi @ gphi) + ma * (gpsi @ gphi)
    c = (ma - mb) * (G @ gphi) / D
    n = len(phi)
    return np.eye(n) - np.outer(c, psi), np.eye(n) + np.outer(c, phi - psi), D / gg


def lift_closed_form(X, phi, ma, mb):
    G = bary_gradients(X)
    gphi = phi @ G
    psi = np.where(phi > 0, phi, 0.0)
    gpsi = psi @ G
    gg = gphi @ gphi
    D = mb * (gg - gpsi @ gphi) + ma * (gpsi @ gphi)
    cJ = math.sqrt(gg) / D
    return -cJ * psi, cJ * (phi - psi)


def pointwise_basis(X, phi, ma, mb, jf=0.0, lift=False):
    """Solve the defining conditions: unknowns are the two linear pieces (d+1 coefficients each); conditions are the
    nodal values (each node on its own side), continuity at d points of the cut, and the flux condition
    mb dv^b/dn - ma dv^a/dn = jf. Returns the nodal values (Vm, Vp) of the pieces, row i = function i (a single
    row for the lift, whose nodal values are zero)."""
    d = X.shape[1]
    n = d + 1
    G = bary_gradients(X)
    nrm = phi @ G
    nrm = nrm / np.linalg.norm(nrm)
    cuts = []
    for i, j in itertools.combinations(range(n), 2):
        if phi[i] * phi[j] < 0:
            cuts.append(X[i] + phi[i] / (phi[i] - phi[j]) * (X[j] - X[i]))
    M = np.zeros((2 * n, 2 * n))
    # a piece: v = a0 + a.x with unknowns [a0, a]; b piece likewise
    r = 0
    for j in range(n):
        off = n if phi[j] > 0 else 0
        M[r, off] = 1.0
        M[r, off + 1:off + n] = X[j]
        r += 1
    for q in cuts[:d]:
        M[r, 0] = -1.0
        M[r, 1:n] = -q
        M[r, n] = 1.0
        M[r, n + 1:] = q
        r += 1
    M[r, 1:n] = -ma * nrm
    M[r, n + 1:] = mb * nrm
    rhs = np.zeros((2 * n, 1 if lift else n))
    if lift:
        rhs[r, 0] = jf
    else:
        rhs[:n, :] = np.eye(n)
    C = np.linalg.solve(M, rhs)
    P = np.hstack([np.ones((n, 1)), X])
    return (P @ C[:n]).T, (P @ C[n:]).T


def random_simplex(rng, d, nonobtuse=True):
    while True:
        X = rng.normal(size=(d + 1, d))
        G = bary_gradients(X)
        gram = G @ G.T
        off = gram[~np.eye(d + 1, dtype=bool)]
        if abs(np.linalg.det(X[1:] - X[0])) < 0.05:
            continue
        if not nonobtuse or off.max() <= 0:
            return X


def random_cut(rng, d):
    while True:
        phi = rng.uniform(-1, 1, d + 1)
        if (phi > 0).any() and (phi < 0).any() and np.abs(phi).min() > 0.02:
            return phi


def as3(X):
    out = np.zeros((X.shape[0], 3))
    out[:, :X.shape[1]] = X
    return out


def evaluate(d, X, phi, pts, ma, mb):
    nodes = as3(X)
    P = np.zeros((len(pts), 3))
    P[:, :d] = pts
    fn = cutclip.ifem_basis_3d if d == 3 else cutclip.ifem_basis_2d
    return fn(nodes, phi, P, ma, mb)


def bary_of(pts):
    return np.hstack([1.0 - pts.sum(axis=1, keepdims=True), pts])


def piece_values(V, lam):
    """value of function i's piece at barycentric points: (m, n)"""
    return lam @ V.T


def piece_gradients(V, X):
    """(n, d): gradient of function i's piece"""
    return V @ bary_gradients(X)


@pytest.mark.parametrize("d", [2, 3])
def test_kernel_basis_equals_pointwise_solve_and_closed_form(d):
    """U1: random affine simplices, random node orders (either orientation), all splits, contrasts 1e-3..1e3."""
    rng = np.random.default_rng(11 + d)
    splits = set()
    worst = 0.0
    for it in range(300 if d == 3 else 200):
        X = random_simplex(rng, d)
        phi = random_cut(rng, d)
        ma, mb = 10.0 ** rng.uniform(-3, 3, 2)
        splits.add(int((phi > 0).sum()))
        pts = rng.dirichlet(np.ones(d + 1), size=7)[:, 1:]
        r = evaluate(d, X, phi, pts, ma, mb)
        assert r["icase"] == 0
        Vm, Vp, s = closed_form(X, phi, ma, mb)
        Vm2, Vp2 = pointwise_basis(X, phi, ma, mb)
        lam = bary_of(pts)
        scale = max(1.0, np.abs(Vm).max(), np.abs(Vp).max())
        # closed form against the direct solve (the plan's formula)
        worst = max(worst, np.abs(Vm - Vm2).max() / scale, np.abs(Vp - Vp2).max() / scale)
        # the kernel against both
        np.testing.assert_allclose(r["va"], piece_values(Vm, lam), rtol=1e-9, atol=1e-9 * scale)
        np.testing.assert_allclose(r["vb"], piece_values(Vp, lam), rtol=1e-9, atol=1e-9 * scale)
        np.testing.assert_allclose(r["va_grad"][:, :, :d], np.broadcast_to(piece_gradients(Vm, X), (7, d + 1, d)),
                                   rtol=1e-8, atol=1e-8 * scale)
        np.testing.assert_allclose(r["vb_grad"][:, :, :d], np.broadcast_to(piece_gradients(Vp, X), (7, d + 1, d)),
                                   rtol=1e-8, atol=1e-8 * scale)
        if d == 3:
            assert np.all(r["va_grad"][:, :, 2] == r["va_grad"][:, :, 2])  # no nan
    assert worst < 1e-9, worst
    if d == 3:
        assert splits == {1, 2, 3}, splits


def test_3d_nodal_partition_of_unity_continuity_and_flux():
    rng = np.random.default_rng(5)
    for it in range(200):
        X = random_simplex(rng, 3)
        phi = random_cut(rng, 3)
        ma, mb = 10.0 ** rng.uniform(-3, 3, 2)
        # nodal: the piece on a node's own side takes the value delta at that node
        nodes_ref = REF_VERTICES[3][:, :3]
        r = evaluate(3, X, phi, nodes_ref, ma, mb)
        for k in range(4):
            own = r["vb"][k] if phi[k] > 0 else r["va"][k]
            np.testing.assert_allclose(own, np.eye(4)[k], atol=1e-9)
        # points on the cut (convex combinations of the cut points), reference coordinates = barycentric 1..3
        cuts = []
        for i, j in itertools.combinations(range(4), 2):
            if phi[i] * phi[j] < 0:
                t = phi[i] / (phi[i] - phi[j])
                lam = np.zeros(4)
                lam[i], lam[j] = 1 - t, t
                cuts.append(lam)
        cuts = np.array(cuts)
        w = rng.dirichlet(np.ones(len(cuts)), size=6)
        lam_pts = w @ cuts
        r = evaluate(3, X, phi, lam_pts[:, 1:], ma, mb)
        scale = max(1.0, np.abs(r["va"]).max(), np.abs(r["vb"]).max())
        np.testing.assert_allclose(r["va"], r["vb"], atol=1e-9 * scale)           # [v] = 0 on the cut
        np.testing.assert_allclose(r["va"].sum(axis=1), 1.0, atol=1e-9)           # partition of unity
        np.testing.assert_allclose(r["vb"].sum(axis=1), 1.0, atol=1e-9)
        n = np.array(r["normal"])
        gphi = phi @ bary_gradients(X)
        n = gphi / np.linalg.norm(gphi)
        jump = mb * (r["vb_grad"] @ n) - ma * (r["va_grad"] @ n)                   # (m, 4): zero flux jump
        np.testing.assert_allclose(jump, 0.0, atol=1e-8 * scale * max(ma, mb))
        # equal coefficients: the ordinary linear basis on both sides
        r1 = evaluate(3, X, phi, lam_pts[:, 1:], 3.0, 3.0)
        np.testing.assert_allclose(r1["va"], lam_pts, atol=1e-12)
        np.testing.assert_allclose(r1["vb"], lam_pts, atol=1e-12)


def test_matches_ifem_3d_notebook():
    """U2: the 32 coefficients IFEM_3D.ipynb prints (x0 = y0 = z0 = 0.45, mu_a = 2, mu_b = 1)."""
    x0 = 0.45
    X = REF_VERTICES[3]
    phi = np.array([-1.0, 1 / x0 - 1, 1 / x0 - 1, 1 / x0 - 1])       # x/x0 + y/y0 + z/z0 - 1 at the vertices
    r = evaluate(3, X, phi, X, 2.0, 1.0)
    # reference == physical here: coefficients of 1, x, y, z from the values at the vertices
    for name, ref in (("va", NB_A), ("vb", NB_B)):
        vals = r[name]                                      # (vertex q, function i)
        coef = np.column_stack([vals[0], vals[1] - vals[0], vals[2] - vals[0], vals[3] - vals[0]])
        np.testing.assert_allclose(coef, ref, atol=1e-13)


def test_permuted_and_reflected_notebook_geometry():
    """The same physical cut with the vertices listed in another order and mirrored (negative orientation)."""
    x0 = 0.45
    X = REF_VERTICES[3]
    phi = np.array([-1.0, 1 / x0 - 1, 1 / x0 - 1, 1 / x0 - 1])
    base = evaluate(3, X, phi, np.array([[0.2, 0.1, 0.3]]), 2.0, 1.0)
    for perm in itertools.permutations(range(4)):
        perm = list(perm)
        Xp, phip = X[perm], phi[perm]
        # the point with barycentric coordinates (1-0.6, 0.2, 0.1, 0.3) in the original order
        lam = np.array([0.4, 0.2, 0.1, 0.3])
        lamp = lam[perm]
        r = evaluate(3, Xp, phip, lamp[1:][None, :], 2.0, 1.0)
        inv = np.argsort(perm)
        np.testing.assert_allclose(r["va"][0][inv], base["va"][0], atol=1e-13)
        np.testing.assert_allclose(r["vb"][0][inv], base["vb"][0], atol=1e-13)


def test_unit_flux_jump_lift():
    """U4: zero at the nodes of its own side, continuous across the cut, mu_b dl^b/dn - mu_a dl^a/dn = 1;
    agrees with the pointwise solve."""
    rng = np.random.default_rng(21)
    for it in range(200):
        X = random_simplex(rng, 3)
        phi = random_cut(rng, 3)
        ma, mb = 10.0 ** rng.uniform(-2, 2, 2)
        la, lb = lift_closed_form(X, phi, ma, mb)
        Va, Vb = pointwise_basis(X, phi, ma, mb, jf=1.0, lift=True)
        scale = max(1.0, np.abs(la).max(), np.abs(lb).max())
        np.testing.assert_allclose(la, Va[0], atol=1e-8 * scale)
        np.testing.assert_allclose(lb, Vb[0], atol=1e-8 * scale)
        pts = rng.dirichlet(np.ones(4), size=5)[:, 1:]
        r = evaluate(3, X, phi, pts, ma, mb)
        lam = bary_of(pts)
        G = bary_gradients(X)
        np.testing.assert_allclose(r["psi"][:, 0, 0], lam @ la, atol=1e-9 * scale)
        np.testing.assert_allclose(r["psi"][:, 1, 0], lam @ lb, atol=1e-9 * scale)
        np.testing.assert_allclose(r["psi"][:, 0, 1:], np.broadcast_to(la @ G, (5, 3)), atol=1e-8 * scale)
        np.testing.assert_allclose(r["psi"][:, 1, 1:], np.broadcast_to(lb @ G, (5, 3)), atol=1e-8 * scale)
        gphi = phi @ G
        n = gphi / np.linalg.norm(gphi)
        assert abs(mb * (lb @ G) @ n - ma * (la @ G) @ n - 1.0) < 1e-8
        # nodal values: l^b vanishes at positive nodes, l^a at negative ones
        assert np.abs(lb[phi > 0]).max() < 1e-9 * scale and np.abs(la[phi < 0]).max() < 1e-9 * scale


def test_uncut_and_touching_elements_use_the_standard_basis():
    X = REF_VERTICES[3]
    for phi, expect in ((np.array([1.0, 2.0, 3.0, 4.0]), 1), (np.array([-1.0, -2.0, -3.0, -4.0]), -1)):
        r = evaluate(3, X, phi, np.array([[0.2, 0.2, 0.2]]), 2.0, 1.0)
        assert r["icase"] == expect
    # the interface lies on a face of the element (3 zero values): element wholly on one side
    r = evaluate(3, X, np.array([1.0, 0.0, 0.0, 0.0]), np.array([[0.2, 0.2, 0.2]]), 2.0, 1.0)
    assert r["icase"] == 0 and (r["edge"] != 0 or r["corner"] != 0)
    lam = bary_of(np.array([[0.2, 0.2, 0.2]]))
    np.testing.assert_allclose(r["va"], lam, atol=1e-14)
    np.testing.assert_allclose(r["vb"], lam, atol=1e-14)
    # the interface passes through a vertex and cuts the opposite face: a genuine cut, closed form still holds
    phi = np.array([0.0, 1.0, -1.0, -0.4])
    r = evaluate(3, X, phi, np.array([[0.2, 0.3, 0.1]]), 2.0, 1.0)
    Vm, Vp, _ = closed_form(X, phi, 2.0, 1.0)
    np.testing.assert_allclose(r["va"][0], bary_of(np.array([[0.2, 0.3, 0.1]]))[0] @ Vm.T, atol=1e-12)
    np.testing.assert_allclose(r["vb"][0], bary_of(np.array([[0.2, 0.3, 0.1]]))[0] @ Vp.T, atol=1e-12)
    # the interface contains an edge and cuts the opposite edge
    phi = np.array([0.0, 0.0, 1.0, -0.5])
    r = evaluate(3, X, phi, np.array([[0.2, 0.3, 0.1]]), 2.0, 1.0)
    Vm, Vp, _ = closed_form(X, phi, 2.0, 1.0)
    np.testing.assert_allclose(r["va"][0], bary_of(np.array([[0.2, 0.3, 0.1]]))[0] @ Vm.T, atol=1e-12)
    np.testing.assert_allclose(r["vb"][0], bary_of(np.array([[0.2, 0.3, 0.1]]))[0] @ Vp.T, atol=1e-12)


def test_singular_basis_on_an_obtuse_tetrahedron_stops_with_a_message():
    """U3: D = 0 at the contrast ma/mb = 1 - 1/s on a tetrahedron with an obtuse dihedral angle; the kernel
    raises. The same tetrahedra at other contrasts, and non-obtuse tetrahedra at every contrast, do not."""
    rng = np.random.default_rng(3)
    found = None
    for it in range(20000):
        X = random_simplex(rng, 3, nonobtuse=False)
        phi = random_cut(rng, 3)
        G = bary_gradients(X)
        gphi = phi @ G
        psi = np.where(phi > 0, phi, 0.0)
        s = (psi @ G) @ gphi / (gphi @ gphi)
        if s < -0.05 or s > 1.05:
            ratio = 1.0 - 1.0 / s          # ma / mb at which D = 0
            if ratio > 0:
                found = (X, phi, ratio)
                break
    assert found is not None
    X, phi, ratio = found
    pts = np.array([[0.2, 0.2, 0.2]])
    with pytest.raises(RuntimeError, match="does not exist.*obtuse"):
        evaluate(3, X, phi, pts, ratio, 1.0)
    evaluate(3, X, phi, pts, 1.5 * ratio, 1.0)           # nearby contrast: fine
    evaluate(3, X, phi, pts, 1.0, 1.0)
    # every non-obtuse tetrahedron at every contrast up to 1e6
    for it in range(100):
        Xn = random_simplex(rng, 3)
        evaluate(3, Xn, random_cut(rng, 3), pts, 1e-6, 1.0)
        evaluate(3, Xn, random_cut(rng, 3), pts, 1.0, 1e-6)


def test_basis_values_stay_bounded_on_a_small_cut():
    """The basis does not blow up when the cut leaves a sliver (unit right tetrahedron, contrasts 1e+-6)."""
    X = REF_VERTICES[3]
    worst = 0.0
    rng = np.random.default_rng(2)
    for contrast in (1e-6, 1e-3, 1e3, 1e6):
        for eps in (1e-1, 1e-4, 1e-8):
            for phi in (np.array([eps, -1, -1, -1.0]), np.array([1 - eps, -eps, -eps, -eps]),
                        np.array([eps, eps, -1, -1.0]), np.array([1, 1, -eps, -eps])):
                lam = rng.dirichlet(np.ones(4), size=200)
                r = evaluate(3, X, phi, lam[:, 1:][:64], contrast, 1.0)
                ph = lam[:64] @ phi
                vals = np.where(ph[:, None] > 0, r["vb"], r["va"])
                worst = max(worst, np.abs(vals).max())
    assert worst <= 1.0 + 1e-8, worst
