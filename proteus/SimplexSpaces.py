"""
Local polynomial spaces of any degree on the reference simplex: node sets, entity-grouped dof layout
and tabulation (values, gradients, Hessians, traces).

This module is numpy only and knows nothing about meshes; ``FemTools`` wraps it
(``PolynomialOnSimplex``, ``SimplexEntityDOFMap``, ``C0_AffinePolynomialOnSimplex``).

Reference cell: the unit simplex of ``FemTools.ReferenceSimplex`` (vertices 0, e_1, .., e_d), barycentric
coordinates ``lambda_0 = 1 - sum(xi)``, ``lambda_i = xi_{i-1}``; face ``f`` is opposite vertex ``f``.

A local dof is identified by a multi-index ``alpha`` in N^{d+1}, ``|alpha| = p``:

* ``family='lagrange'``: the dof is the value at the node ``X(alpha)``, which is ``alpha/p`` for
  ``nodes='equispaced'`` and the warp-and-blend point for ``nodes='gll'`` (Gauss-Lobatto-Legendre points on
  the edges). The two coincide for p <= 2.
* ``family='bernstein'``: the dof is the coefficient of ``B_alpha = p!/alpha! lambda^alpha``; ``alpha/p`` is
  its control point, not an interpolation point.

Local order (grouped by entity): vertices 0..d; edges (0,1),(1,2),(0,2) in 2D and
(0,1),(1,2),(2,3),(0,2),(1,3),(0,3) in 3D; faces 0..3 in 3D; the cell interior. Inside an entity with local
vertices a_1 < a_2 < .. the dofs are in decreasing lexicographic order of (alpha_{a_1}, alpha_{a_2}, ..), so
on an edge (a, b) they count from a. For p = 1 and 2 this is the numbering of
``LinearOnSimplexWithNodalBasis`` and ``QuadraticOnSimplexWithNodalBasis``.

Node sets (``nodes='gll'``)
---------------------------
Warburton's warp-and-blend construction (T. Warburton, "An explicit construction of interpolation nodes on
the simplex", J. Eng. Math. 56 (2006) 247-262; also Hesthaven and Warburton, "Nodal Discontinuous Galerkin
Methods", Springer 2008, sections 6.1 and 10.1), written here directly in barycentric coordinates, which makes
the symmetry explicit: every edge {a, b} moves a point along the edge direction by

    lambda_b += s,  lambda_a -= s,   s = 2 lambda_a lambda_b w(lambda_b - lambda_a) (1 + (alpha lambda_c)^2),

where ``w(r)`` is the degree-p interpolant of (GLL - equispaced) through the equispaced points, divided by
(1 - r^2). On the tetrahedron each face contributes its triangle shift times the face blend
``lambda_b lambda_c lambda_d / ((lambda_b + lambda_a/2)(lambda_c + lambda_a/2)(lambda_d + lambda_a/2))
(1 + (alpha lambda_a)^2)``. The blend parameters ``alpha`` are the published optimized values for p = 1..7.

cekees's derivation (``cekees/AnalyticalADR`` ``FEM_Basis_Construction_Sympy.ipynb``, ``nodes2d``/``nodes3d``,
which generates the test oracle) is the same construction on the triangle and on the boundary of the
tetrahedron; inside the tetrahedron it deflates the face warp with the face's own coordinates instead of
(1 - r^2), which moves the interior nodes for p >= 5 (there are none below p = 4 and one, at the centroid, at
p = 4). The measured Lebesgue constants are lower for the published construction (see
``test/test_simplex_spaces.py``), so that one is used.

Tabulation
----------
``values = Phi(points) C`` with ``Phi`` the Bernstein basis of degree p (closed form, exact derivatives
through ``dB_alpha^p/dlambda_i = p B_{alpha-e_i}^{p-1}``) and ``C`` the inverse of ``V[i, alpha] =
B_alpha(x_i)`` (the identity for ``bernstein``). The test oracle uses a monomial basis and a 50-digit solve,
so the two paths are independent.
"""
import itertools
from math import factorial

import numpy as np

# Optimized blend parameters for p = 1..7 (Hesthaven and Warburton 2008, the alpopt tables of sections 6.1
# and 10.1). Higher degrees are deliberately not offered until their values are checked against the source.
ALPHA_TRIANGLE = (0.0, 0.0, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999)
ALPHA_TETRAHEDRON = (0.0, 0.0, 0.0, 0.1002, 1.1332, 1.5608, 1.3413)
MAX_WARP_BLEND_ORDER = len(ALPHA_TRIANGLE)

EDGES = {1: ((0, 1),),
         2: ((0, 1), (1, 2), (0, 2)),
         3: ((0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3))}


def faces(d):
    """Local vertices of the faces of the d-simplex; face f is opposite vertex f."""
    return tuple(tuple(v for v in range(d + 1) if v != f) for f in range(d + 1))


def entities(d):
    """entity_vertices[dim][n]: local vertices of entity n of dimension dim (dim = 0..d)."""
    ev = [tuple((v,) for v in range(d + 1))]
    if d >= 1:
        ev.append(EDGES[d])
    if d == 2:
        ev.append(((0, 1, 2),))
    if d == 3:
        ev.append(faces(3))
        ev.append(((0, 1, 2, 3),))
    return ev


def interior_multi_indices(m, p):
    """Positive m-tuples summing to p, in decreasing lexicographic order (the dofs interior to an entity
    with m vertices)."""
    out = [b for b in itertools.product(range(p - m + 1, 0, -1), repeat=m) if sum(b) == p]
    return sorted(out, reverse=True)


def multi_indices(d, p):
    """The local dofs in entity-grouped order.

    Returns (alpha, entity_dofs): alpha is (dim, d+1) int; entity_dofs[dim][n] lists the local dofs
    interior to entity n of dimension dim.
    """
    alpha = []
    entity_dofs = []
    for verts_list in entities(d):
        entity_dofs.append([])
        for verts in verts_list:
            dofs = []
            for b in interior_multi_indices(len(verts), p):
                a = [0] * (d + 1)
                for v, bv in zip(verts, b):
                    a[v] = bv
                dofs.append(len(alpha))
                alpha.append(a)
            entity_dofs[-1].append(dofs)
    alpha = np.array(alpha, dtype=int).reshape(-1, d + 1)
    assert alpha.shape[0] == factorial(p + d) // (factorial(p) * factorial(d))
    return alpha, entity_dofs


def gll_points(p):
    """The p+1 Gauss-Lobatto-Legendre points on [-1, 1]: the end points and the roots of P_p'."""
    if p == 1:
        return np.array([-1.0, 1.0])
    L = np.polynomial.legendre.Legendre.basis(p)
    dL, ddL = L.deriv(), L.deriv(2)
    x = np.sort(dL.roots().real)
    for _ in range(3):  # Newton polish of the companion-matrix roots
        x = x - dL(x) / ddL(x)
    x = 0.5 * (x - x[::-1])  # exact antisymmetry
    return np.concatenate(([-1.0], x, [1.0]))


def deflated_warp(p, r):
    """w(r) = [sum_i (gll_i - eq_i) l_i(r)] / (1 - r^2), l_i the Lagrange polynomials of the equispaced
    points eq_i; evaluated without the removable singularity at r = +-1."""
    r = np.asarray(r, dtype=float)
    eq = np.linspace(-1.0, 1.0, p + 1)
    g = gll_points(p)
    w = np.zeros_like(r)
    for i in range(1, p):
        term = (g[i] - eq[i]) / (1.0 - eq[i] ** 2) * np.ones_like(r)
        for j in range(1, p):
            if j != i:
                term = term * (r - eq[j]) / (eq[i] - eq[j])
        w += term
    return w


def _triangle_shift(p, alpha, L, a, b, c):
    """Barycentric shift (nPoints, len(L)) of the triangle with vertices a, b, c (columns of L)."""
    shift = np.zeros_like(L)
    for (x, y, z) in ((a, b, c), (b, c, a), (c, a, b)):
        s = 2.0 * L[:, x] * L[:, y] * deflated_warp(p, L[:, y] - L[:, x]) * (1.0 + (alpha * L[:, z]) ** 2)
        shift[:, y] += s
        shift[:, x] -= s
    return shift


def warp_blend_barycentric(d, p, alpha=None):
    """Barycentric coordinates (dim, d+1) of the warp-and-blend nodes, in the order of ``multi_indices``."""
    mi, _ = multi_indices(d, p)
    L = mi / float(p)
    if p <= 2 or d == 1:
        if d == 1 and p > 2:
            # the 1D GLL points: vertex 0 is xi = 0 (lambda_1 = 0)
            g = 0.5 * (gll_points(p) + 1.0)
            lam1 = g[mi[:, 1]]
            return np.column_stack([1.0 - lam1, lam1])
        return L
    if alpha is None:
        if p > MAX_WARP_BLEND_ORDER:
            raise ValueError("warp-and-blend nodes are available for p <= %d (blend parameter table); "
                             "pass alpha explicitly or use nodes='equispaced'" % MAX_WARP_BLEND_ORDER)
        alpha = (ALPHA_TRIANGLE if d == 2 else ALPHA_TETRAHEDRON)[p - 1]
    if d == 2:
        return L + _triangle_shift(p, alpha, L, 0, 1, 2)
    tol = 1.0e-10
    shift = np.zeros_like(L)
    face_shift = []
    for a in range(4):
        b, c, e = [v for v in range(4) if v != a]
        fs = _triangle_shift(p, alpha, L, b, c, e)
        face_shift.append(fs)
        blend = L[:, b] * L[:, c] * L[:, e]
        denom = (L[:, b] + 0.5 * L[:, a]) * (L[:, c] + 0.5 * L[:, a]) * (L[:, e] + 0.5 * L[:, a])
        ok = denom > tol
        blend[ok] = (1.0 + (alpha * L[ok, a]) ** 2) * blend[ok] / denom[ok]
        shift += blend[:, None] * fs
    # nodes on the edges get the edge warp alone (their blends vanish)
    for a in range(4):
        b, c, e = [v for v in range(4) if v != a]
        on_edge = (L[:, a] < tol) & (((L[:, b] > tol).astype(int) + (L[:, c] > tol) + (L[:, e] > tol)) < 3)
        shift[on_edge] = face_shift[a][on_edge]
    return L + shift


def bernstein(d, q, lam):
    """All Bernstein polynomials of degree q at barycentric points lam (n, d+1): dict alpha -> (n,)."""
    n = lam.shape[0]
    if q < 0:
        return {}
    powers = [[np.ones(n)] for _ in range(d + 1)]
    for i in range(d + 1):
        for _ in range(q):
            powers[i].append(powers[i][-1] * lam[:, i])
    out = {}
    fq = factorial(q)
    for a in itertools.product(range(q + 1), repeat=d + 1):
        if sum(a) != q:
            continue
        coef = fq
        v = np.ones(n)
        for i, ai in enumerate(a):
            coef //= factorial(ai)
            v = v * powers[i][ai]
        out[a] = coef * v
    return out


class Tabulation(object):
    """values (nQ, dim); grads (nQ, dim, d); hessians (nQ, dim, d, d) (None if not requested)."""

    def __init__(self, values, grads=None, hessians=None):
        self.values, self.grads, self.hessians = values, grads, hessians


def face_map(d, f):
    """The affine map x = o + J xBar of ``ReferenceSimplex.boundaryMapList[f]``: returns (o, J)."""
    if d == 1:
        return (np.array([1.0]), np.zeros((1, 1))) if f == 0 else (np.array([0.0]), np.zeros((1, 1)))
    if d == 2:
        return [(np.array([0.0, 1.0]), np.array([[1.0], [-1.0]])),
                (np.array([0.0, 0.0]), np.array([[0.0], [1.0]])),
                (np.array([0.0, 0.0]), np.array([[1.0], [0.0]]))][f]
    return [(np.array([0.0, 0.0, 1.0]), np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])),
            (np.array([0.0, 0.0, 0.0]), np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])),
            (np.array([0.0, 0.0, 0.0]), np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])),
            (np.array([0.0, 0.0, 0.0]), np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))][f]


class SimplexSpace(object):
    """P_p on the reference d-simplex with the Lagrange (nodal) or the Bernstein basis.

    Attributes: ``d``, ``order``, ``family``, ``nodes``, ``dim``, ``multi_index`` (dim, d+1),
    ``entity_dofs[dim][n]``, ``entity_vertices[dim][n]``, ``dof_points`` (dim, d) (nodes, or control points
    for Bernstein), ``lagrange_points`` (dim, d) (a unisolvent set in the same order, equal to ``dof_points``
    for Lagrange; the warp-and-blend nodes for Bernstein), ``C`` (dim, dim) with basis = Bernstein C.
    """

    def __init__(self, d, order, family="lagrange", nodes="gll", alpha=None):
        if d not in (1, 2, 3):
            raise ValueError("d must be 1, 2 or 3")
        if order < 1:
            raise ValueError("order must be >= 1")
        if family not in ("lagrange", "bernstein"):
            raise ValueError("family must be 'lagrange' or 'bernstein'")
        if nodes not in ("gll", "equispaced"):
            raise ValueError("nodes must be 'gll' or 'equispaced'")
        self.d, self.order, self.family, self.nodes = d, order, family, nodes
        self.multi_index, self.entity_dofs = multi_indices(d, order)
        self.entity_vertices = entities(d)
        self.dim = self.multi_index.shape[0]
        self._index = {tuple(a): i for i, a in enumerate(self.multi_index.tolist())}
        if nodes == "gll":
            bary = warp_blend_barycentric(d, order, alpha)
        else:
            bary = self.multi_index / float(order)
        self.lagrange_points = bary[:, 1:].copy()
        if family == "lagrange":
            self.dof_points = self.lagrange_points
            V = self._bernstein_tables(self.lagrange_points, 0)[0]
            self.C = np.linalg.inv(V)
            self.condition_number = np.linalg.cond(V)
        else:
            self.dof_points = self.multi_index[:, 1:] / float(order)
            self.C = np.eye(self.dim)
            self.condition_number = 1.0

    def _bernstein_tables(self, points, derivatives):
        """Bernstein basis of degree p in the local dof order: values, reference gradients, Hessians."""
        points = np.asarray(points, dtype=float).reshape(-1, self.d)
        d, p, n = self.d, self.order, points.shape[0]
        lam = np.column_stack([1.0 - points.sum(axis=1), points])
        Bp = bernstein(d, p, lam)
        vals = np.empty((n, self.dim))
        for a, i in self._index.items():
            vals[:, i] = Bp[a]
        grads = hess = None
        if derivatives >= 1:
            B1 = bernstein(d, p - 1, lam)
            dlam = np.zeros((n, self.dim, d + 1))  # derivatives with respect to lambda_i
            for a, i in self._index.items():
                for k in range(d + 1):
                    if a[k] > 0:
                        s = list(a)
                        s[k] -= 1
                        dlam[:, i, k] = p * B1[tuple(s)]
            grads = dlam[:, :, 1:] - dlam[:, :, :1]
        if derivatives >= 2:
            hess = np.zeros((n, self.dim, d, d))
            if p >= 2:
                B2 = bernstein(d, p - 2, lam)
                d2 = np.zeros((n, self.dim, d + 1, d + 1))
                for a, i in self._index.items():
                    for k in range(d + 1):
                        for l in range(k, d + 1):
                            s = list(a)
                            s[k] -= 1
                            s[l] -= 1
                            if min(s) >= 0:
                                d2[:, i, k, l] = p * (p - 1) * B2[tuple(s)]
                                d2[:, i, l, k] = d2[:, i, k, l]
                hess = (d2[:, :, 1:, 1:] - d2[:, :, 1:, :1] - d2[:, :, :1, 1:] + d2[:, :, :1, :1])
        return vals, grads, hess

    def tabulate(self, points, derivatives=2):
        """Basis values, reference gradients and Hessians at points (nQ, d) of the reference simplex."""
        vals, grads, hess = self._bernstein_tables(points, derivatives)
        if self.family == "bernstein":
            return Tabulation(vals, grads, hess)
        C = self.C
        return Tabulation(vals @ C,
                          None if grads is None else np.einsum("qad,ai->qid", grads, C),
                          None if hess is None else np.einsum("qade,ai->qide", hess, C))

    def tabulate_trace(self, face, points_bar, derivatives=1):
        """The element tabulation at the points of face ``face`` with face coordinates points_bar (nQ, d-1):
        full (not tangential) gradients, evaluated at the mapped points."""
        o, J = face_map(self.d, face)
        xb = np.asarray(points_bar, dtype=float).reshape(-1, J.shape[1])
        return self.tabulate(o + xb @ J.T, derivatives)

    def on_face(self, i, face):
        """True if local dof i lies on (the closure of) face ``face``."""
        return self.multi_index[i, face] == 0

    def lebesgue_constant(self, n=60):
        """max_x sum_i |phi_i(x)| sampled on the lattice with n intervals per edge (a lower bound)."""
        pts = np.array([a[1:] for a in itertools.product(range(n + 1), repeat=self.d + 1) if sum(a) == n]) / float(n)
        return np.abs(self.tabulate(pts, 0).values).sum(axis=1).max()
