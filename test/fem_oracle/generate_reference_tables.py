#!/usr/bin/env python
"""
Generate the pinned reference tables for local finite element spaces (C8 round 1).

Run only to REGENERATE ``reference_tables.npz`` (needs sympy and mpmath). The tests never run
this script: they read the pinned file, so CI does not need sympy.

    python test/fem_oracle/generate_reference_tables.py [--out reference_tables.npz]

Provenance (all derivations are cekees's own)
---------------------------------------------
* ``getBasis`` (Lagrange as the inverse Vandermonde of a monomial basis on nodes), ``getBernsteinBasis``
  (Bernstein-Bezier, multinomial in the barycentric coordinates), ``gll``, ``warpfactor``, ``nodes2d`` and
  ``nodes3d`` (Gauss-Lobatto-Legendre points and Warburton warp-and-blend nodes, tuned ``alpha``
  table) are from ``cekees/AnalyticalADR``, ``FEM_Basis_Construction_Sympy.ipynb`` (cells 50, 47, 77,
  80, 100, 88, 82). The GLL / warp-blend functions are copied below with the print statements removed.
* ``cekees/fem`` ``basis.py`` (``getBasis``) is the same Vandermonde-inverse construction.
* ``LGL2D.txt`` is deliberately NOT used (provenance unsettled).

What differs from the notebook, and why
---------------------------------------
* Everything is evaluated in mpmath at 50 digits (the monomial Vandermonde is ill conditioned at p = 4),
  from the float64 node coordinates taken as exact. Values, gradients and Hessians are exact derivatives of
  the sympy expressions, not finite differences.
* The notebook's cube lives on [0,1]^d; ``FemTools`` uses [-1,1]^d. Quadrilateral and hexahedron tables are
  on [-1,1]^d (derivatives carry the chain-rule factors 1/2 and 1/4).
* The notebook's Bernstein ``nodes`` list for the triangle and tetrahedron does not label its basis
  functions correctly (its functions are indexed by exponents of the barycentric coordinates, its nodes by
  (i/p, j/p)). The tables here use the control-point lattice that matches the functions; see "Layout".

Layout of ``reference_tables.npz``
----------------------------------
Keys are ``<family>/<cell>/p<p>/<item>``.

* family: ``lagrange_equi`` (equispaced nodes), ``lagrange_gll`` (GLL on edges, warp-and-blend inside;
  equal to equispaced for p <= 2; the quadrilateral/hexahedron have equispaced only, which equals GLL
  for Q1, Q2), ``bernstein``.
* cell: ``interval`` ([0,1]), ``triangle`` and ``tetrahedron`` (the unit simplex, vertices at the origin
  and the unit vectors), ``quadrilateral`` and ``hexahedron`` ([-1,1]^d).
* ``lattice`` (n, d): nodes (Lagrange) or Bezier control points (Bernstein). Function j is associated with
  lattice point j. Lagrange equispaced order, from the notebook: triangle (i/p, j/p) for i, j with i+j<=p,
  i outer; tetrahedron (i/p, j/p, m/p), i outer, j, m nested; quad/hex i outer (x slowest). GLL: the
  warp-blend node displaced from the equispaced node with the same index, with barycentric weights
  (L0, L1) = (x, y) (and (L0, L1, L2) = (x, y, z)). Bernstein: the function with exponents
  (a0, a1, ..., ad) on the barycentric coordinates (1-sum x, x, y, z) has control point (a1, ..., ad)/p;
  loops i = a0 outer.
* ``x`` (m, d) interior sample points (the first is the centroid), ``val`` (m, n), ``grad`` (m, n, d),
  ``hess`` (m, n, d, d).
* faces. ``FemTools`` numbers the faces of a simplex by the vertex they are opposite, the faces of the
  cube as documented in ``ReferenceCube``. ``face_xi`` (nf, nq, dxi) are face parameters as
  ``FemTools`` takes them (dxi = d-1, but 1 for d = 1, where the parameter is ignored);
  ``face_x`` (nf, nq, d) the same points in the reference cell; ``face_val``, ``face_grad``, ``face_hess`` as above
  on those points. The 3D gradients and Hessians on faces are the full cell derivatives.
* ``node_val`` (n, n), Lagrange only: the basis evaluated at the lattice (the identity to 1e-15).
* ``meta/seed`` the seed used for the sample points.
"""
import argparse
import os
import sys
from math import pi, sqrt, sin, cos

import numpy as np

SEED = 20261112
MAX_P = {"interval": 5, "triangle": 5, "tetrahedron": 4}  # the degrees CI supports

# alpha table from cell 82 of the notebook, rows p=3..15 (column 0: triangle, column 1: tetrahedron)
ALPHA = np.array([[1.4152, 0.0000],
                  [0.1001, 0.1002],
                  [0.2751, 1.1332],
                  [0.9808, 1.5608],
                  [1.0999, 1.3413],
                  [1.2832, 1.2577],
                  [1.3648, 1.1603],
                  [1.4773, 1.0153],
                  [1.4959, 0.6080],
                  [1.5743, 0.4523],
                  [1.5770, 0.8856],
                  [1.6223, 0.8717],
                  [1.6258, 0.9655]])


def alpha_for(p, column):
    return 0.0 if p < 3 else ALPHA[p - 3, column]


# ---------------------------------------------------------------------------------------------
# GLL and warp-and-blend nodes: cekees/AnalyticalADR FEM_Basis_Construction_Sympy.ipynb
# ---------------------------------------------------------------------------------------------
def gll(p):  # cell 77
    N = p + 1
    x = np.flip(np.cos(pi * np.arange(float(p + 1)) / p))
    V = np.zeros((N, N), 'd')
    xold = 2 * x
    eps = np.finfo(float).eps
    while np.abs(x - xold).max() > eps:
        xold[:] = x
        V[:, 0] = 1.0
        V[:, 1] = x
        for k in range(1, p):
            V[:, k + 1] = ((2 * (k + 1) - 1) * x * V[:, k] - k * V[:, k - 1]) / (k + 1)
        x[:] = xold - (x * V[:, N - 1] - V[:, p - 1]) / (N * V[:, N - 1])
    return x


def warpfactor(p, xnodes, xout):  # cell 80
    warp = np.zeros(xout.shape)
    xeq = np.linspace(-1, 1, p + 1)
    for i in range(p + 1):
        d = (xnodes[i] - xeq[i])
        for j in range(p + 1):
            if i != j:
                d = d * (xout - xeq[j]) / (xeq[i] - xeq[j])
        warp += d
    return warp


def nodes2d(p, alpha):  # cell 100
    N = (p + 1) * (p + 2) // 2
    gaussX = gll(p)
    L = np.zeros((3, N), 'd')
    sk = 0
    for n in range(0, p + 1):
        for m in range(0, p + 1 - n):
            L[0, sk] = n / p
            L[1, sk] = m / p
            sk = sk + 1
    L[2, :] = 1 - L[0, :] - L[1, :]
    v = np.array([[0, 2 / sqrt(3)],
                  [-1, -1 / sqrt(3)],
                  [1, -1 / sqrt(3)]])
    t = np.zeros_like(v)
    for i in range(v.shape[0]):
        t[i, :] = v[(i + 2) % 3] - v[(i + 1) % 3]
        t[i, :] /= np.linalg.norm(t[i])
    X = np.dot(L.transpose(), v)

    def b(l1, l2, l3):
        return 4 * l2 * l3 / ((2 * l3 + l1) * (2 * l2 + l1) + 1.0e-16)

    def w(l1, l2, l3):
        return warpfactor(p, gaussX, l3 - l2) * (1 + (alpha * l1) ** 2)
    g = (b(L[0], L[1], L[2]) * w(L[0], L[1], L[2]))[:, np.newaxis] * t[0] + \
        (b(L[1], L[2], L[0]) * w(L[1], L[2], L[0]))[:, np.newaxis] * t[1] + \
        (b(L[2], L[0], L[1]) * w(L[2], L[0], L[1]))[:, np.newaxis] * t[2]
    XR = X.copy()
    X += g
    return X, g, XR, v


def nodes3d(p, alpha):  # cell 88
    N = (p + 1) * (p + 2) * (p + 3) // 6
    gaussX = gll(p)
    v2d = np.array([[0, 2 / sqrt(3)],
                    [-1, -1 / sqrt(3)],
                    [1, -1 / sqrt(3)]])
    t2d = np.zeros_like(v2d)
    for i in range(v2d.shape[0]):
        t2d[i, :] = v2d[(i + 2) % 3] - v2d[(i + 1) % 3]
        t2d[i, :] /= np.linalg.norm(t2d[i])

    def warp2d(l2, l3):
        return warpfactor(p, gaussX, l3 - l2)

    def blend2d(l1, l2, l3):
        return (1 + (alpha * l1) ** 2) * 4 * l3 * l2 / ((2 * l3 + l1) * (2 * l2 + l1) + 1.0e-16)

    def g2d(l1, l2, l3):
        return (blend2d(l1, l2, l3)[:, np.newaxis] * warp2d(l2, l3)[:, np.newaxis] * t2d[0] +
                blend2d(l2, l3, l1)[:, np.newaxis] * warp2d(l3, l1)[:, np.newaxis] * t2d[1] +
                blend2d(l3, l1, l2)[:, np.newaxis] * warp2d(l1, l2)[:, np.newaxis] * t2d[2])
    L = np.zeros((4, N), 'd')
    sk = 0
    for n in range(0, p + 1):
        for m in range(0, p + 1 - n):
            for k in range(0, p + 1 - n - m):
                L[0, sk] = n / p
                L[1, sk] = m / p
                L[2, sk] = k / p
                sk = sk + 1
    L[3, :] = 1 - L[0] - L[1] - L[2]
    v = 2 / (8 / 3) ** 0.5 * np.array([
        [0, -(2 / 9) ** 0.5, -(2 / 9) ** 0.5, (8 / 9) ** 0.5],
        [0, -(2 / 3) ** 0.5, (2 / 3) ** 0.5, 0],
        [1, -(1 / 3), -(1 / 3), -(1 / 3)]]).transpose()
    X = np.dot(L.transpose(), v)

    def blend(L1, L2, L3, L4):
        return (1.0 + (alpha * L1) ** 2) * 8 * L2 * L3 * L4 / ((2 * L2 + L1) * (2 * L3 + L1) * (2 * L4 + L1) + 1.0e-16)
    blend1 = blend(L[0], L[1], L[2], L[3])
    blend2 = blend(L[1], L[2], L[3], L[0])
    blend3 = blend(L[2], L[3], L[0], L[1])
    blend4 = blend(L[3], L[0], L[1], L[2])
    t = np.zeros((v.shape[0], 2, v.shape[1]), 'd')
    for i in range(v.shape[0]):
        t1 = v[(i + 3) % 4] - v[(i + 1) % 4]
        t1 /= np.linalg.norm(t1)
        t2 = v[(i + 2) % 4] - v[(i + 1) % 4]
        t2 -= np.dot(t1, t2) * t1
        t2 /= np.linalg.norm(t2)
        t[i, 0, :] = t1
        t[i, 1, :] = t2
    G1 = g2d(L[2], L[1], L[3])
    G2 = g2d(L[3], L[2], L[0])
    G3 = g2d(L[0], L[3], L[1])
    G4 = g2d(L[1], L[0], L[2])
    warpfactor1 = G1[:, 0, np.newaxis] * t[0, 0] + G1[:, 1, np.newaxis] * t[0, 1]
    warpfactor2 = G2[:, 0, np.newaxis] * t[1, 0] + G2[:, 1, np.newaxis] * t[1, 1]
    warpfactor3 = G3[:, 0, np.newaxis] * t[2, 0] + G3[:, 1, np.newaxis] * t[2, 1]
    warpfactor4 = G4[:, 0, np.newaxis] * t[3, 0] + G4[:, 1, np.newaxis] * t[3, 1]
    warp1 = blend1[:, np.newaxis] * warpfactor1
    warp2 = blend2[:, np.newaxis] * warpfactor2
    warp3 = blend3[:, np.newaxis] * warpfactor3
    warp4 = blend4[:, np.newaxis] * warpfactor4
    g = warp1 + warp2 + warp3 + warp4
    for i in range(L.shape[1]):
        if abs(L[0, i]) < 1.0e-8:
            g[i] = warpfactor1[i]
        elif abs(L[1, i]) < 1.0e-8:
            g[i] = warpfactor2[i]
        elif abs(L[2, i]) < 1.0e-8:
            g[i] = warpfactor3[i]
        elif abs(L[3, i]) < 1.0e-8:
            g[i] = warpfactor4[i]
    X += g
    return X, g, X - g, v


def gll_reference_nodes(cell, p):
    """GLL / warp-and-blend nodes mapped to the reference cell, same index order as the equispaced nodes."""
    if cell == "interval":
        return ((gll(p) + 1.0) / 2.0)[:, None]
    if cell == "triangle":
        X, g, XR, v = nodes2d(p, alpha_for(p, 0))
    elif cell == "tetrahedron":
        X, g, XR, v = nodes3d(p, alpha_for(p, 1))
    else:
        raise ValueError(cell)
    d = v.shape[1]
    # barycentric coordinates of X with respect to the equilateral vertices v (rows), then
    # reference coordinates (L0, ..., L_{d-1}) as for the equispaced lattice
    A = np.vstack([v.T, np.ones((1, d + 1))])
    rhs = np.vstack([X.T, np.ones((1, X.shape[0]))])
    Lb = np.linalg.solve(A, rhs)
    return Lb[:d].T.copy()


# ---------------------------------------------------------------------------------------------
# space definitions
# ---------------------------------------------------------------------------------------------
DIM = {"interval": 1, "triangle": 2, "tetrahedron": 3, "quadrilateral": 2, "hexahedron": 3}


def equispaced_lattice(cell, p):
    if cell == "interval":
        return np.array([[i / p] for i in range(p + 1)])
    if cell == "triangle":
        return np.array([[i / p, j / p] for i in range(p + 1) for j in range(p + 1 - i)])
    if cell == "tetrahedron":
        return np.array([[i / p, j / p, m / p] for i in range(p + 1) for j in range(p + 1 - i)
                         for m in range(p + 1 - i - j)])
    if cell == "quadrilateral":
        return np.array([[i / p, j / p] for i in range(p + 1) for j in range(p + 1)])
    if cell == "hexahedron":
        return np.array([[i / p, j / p, k / p] for i in range(p + 1) for j in range(p + 1) for k in range(p + 1)])
    raise ValueError(cell)


def symbols(d):
    import sympy
    return sympy.symbols('x y z')[:d]


def monomials(cell, p):
    """Monomial generating set, as ``getBasis`` in the notebook (cells 50)."""
    X = symbols(DIM[cell])
    if cell == "interval":
        return [X[0] ** i for i in range(p + 1)]
    if cell == "triangle":
        return [X[0] ** j * X[1] ** i for i in range(p + 1) for j in range(p + 1 - i)]
    if cell == "tetrahedron":
        return [X[0] ** j * X[1] ** i * X[2] ** m for i in range(p + 1) for j in range(p + 1 - i)
                for m in range(p + 1 - i - j)]
    if cell == "quadrilateral":
        return [X[0] ** i * X[1] ** j for i in range(p + 1) for j in range(p + 1)]
    if cell == "hexahedron":
        return [X[0] ** i * X[1] ** j * X[2] ** k for i in range(p + 1) for j in range(p + 1) for k in range(p + 1)]
    raise ValueError(cell)


def bernstein_functions(cell, p):
    """Bernstein-Bezier functions and their control points, as ``getBernsteinBasis`` (cell 47).

    Function (i, j, ...) is the multinomial times (1-x-y-z)^i x^j y^k z^(p-i-j-k); its control point is
    (j, k, m)/p. For the interval, binomial(p,i) (1-x)^(p-i) x^i has control point i/p.
    """
    import sympy
    from sympy import factorial, binomial
    X = symbols(DIM[cell])
    if cell == "interval":
        x, = X
        return ([binomial(p, i) * (1 - x) ** (p - i) * x ** i for i in range(p + 1)],
                np.array([[i / p] for i in range(p + 1)]))
    if cell == "triangle":
        x, y = X
        fns, pts = [], []
        for i in range(p + 1):
            for j in range(p + 1 - i):
                k = p - i - j
                fns.append(factorial(p) / (factorial(i) * factorial(j) * factorial(k)) * (1 - x - y) ** i * x ** j * y ** k)
                pts.append([j / p, k / p])
        return fns, np.array(pts)
    if cell == "tetrahedron":
        x, y, z = X
        fns, pts = [], []
        for i in range(p + 1):
            for j in range(p + 1 - i):
                for k in range(p + 1 - i - j):
                    m = p - i - j - k
                    fns.append(factorial(p) / (factorial(i) * factorial(j) * factorial(k) * factorial(m)) *
                               (1 - x - y - z) ** i * x ** j * y ** k * z ** m)
                    pts.append([j / p, k / p, m / p])
        return fns, np.array(pts)
    raise ValueError(cell)


class Space(object):
    """phi_i = sum_k C[k, i] g_k, with g_k the generating functions (sympy, in unit coordinates)."""

    def __init__(self, family, cell, p):
        import mpmath as mp
        import sympy
        self.family, self.cell, self.p = family, cell, p
        self.d = DIM[cell]
        self.cube = cell in ("quadrilateral", "hexahedron")
        X = symbols(self.d)
        if family == "bernstein":
            self.gens, lattice = bernstein_functions(cell, p)
            self.n = len(self.gens)
            self.C = mp.eye(self.n)
            self.lattice_unit = lattice
        else:
            self.gens = monomials(cell, p)
            self.n = len(self.gens)
            if family == "lagrange_equi" or self.cube:
                lattice = equispaced_lattice(cell, p)
            elif family == "lagrange_gll":
                lattice = gll_reference_nodes(cell, p) if p > 2 else equispaced_lattice(cell, p)
            else:
                raise ValueError(family)
            self.lattice_unit = lattice
            gen_f = [sympy.lambdify(X, g, 'mpmath') for g in self.gens]
            V = mp.matrix(self.n, self.n)
            for i in range(self.n):
                pt = [mp.mpf(float(c)) for c in lattice[i]]
                for k in range(self.n):
                    V[i, k] = gen_f[k](*pt)
            self.C = mp.inverse(V)
        # derivative function tables
        self.f0 = [sympy.lambdify(X, g, 'mpmath') for g in self.gens]
        self.f1 = [[sympy.lambdify(X, g.diff(X[a]), 'mpmath') for g in self.gens] for a in range(self.d)]
        self.f2 = [[[sympy.lambdify(X, g.diff(X[a]).diff(X[b]), 'mpmath') for g in self.gens]
                    for b in range(self.d)] for a in range(self.d)]

    @property
    def lattice(self):
        """Lattice in the oracle's reference coordinates ([-1,1]^d for the cube cells)."""
        return 2.0 * self.lattice_unit - 1.0 if self.cube else self.lattice_unit

    def evaluate(self, x):
        """Values (m,n), gradients (m,n,d), Hessians (m,n,d,d) at points x (m,d) in oracle coordinates."""
        import mpmath as mp
        x = np.asarray(x, dtype=float)
        m, d, n = x.shape[0], self.d, self.n
        xu = (x + 1.0) / 2.0 if self.cube else x
        pts = [[mp.mpf(float(c)) for c in row] for row in xu]

        def table(fns):
            M = mp.matrix(m, n)
            for r in range(m):
                for k in range(n):
                    M[r, k] = fns[k](*pts[r])
            return M
        val = np.array((table(self.f0) * self.C).tolist(), dtype=float).reshape(m, n)
        grad = np.zeros((m, n, d))
        hess = np.zeros((m, n, d, d))
        s = 0.5 if self.cube else 1.0
        for a in range(d):
            grad[:, :, a] = np.array((table(self.f1[a]) * self.C).tolist(), dtype=float).reshape(m, n) * s
            for b in range(d):
                hess[:, :, a, b] = np.array((table(self.f2[a][b]) * self.C).tolist(), dtype=float).reshape(m, n) * s * s
        return val, grad, hess


# ---------------------------------------------------------------------------------------------
# sample points and faces
# ---------------------------------------------------------------------------------------------
def sample_points(cell, m, rs):
    d = DIM[cell]
    if cell in ("quadrilateral", "hexahedron"):
        pts = rs.uniform(-1.0, 1.0, size=(m, d))
        pts[0] = 0.0
    elif cell == "interval":
        pts = rs.uniform(0.0, 1.0, size=(m, 1))
        pts[0] = 0.5
    else:
        pts = rs.dirichlet(np.ones(d + 1), size=m)[:, 1:]
        pts[0] = 1.0 / (d + 1)
    return pts


def face_maps(cell):
    """Affine face maps x = o + J xi in FemTools' numbering (ReferenceSimplex / ReferenceCube)."""
    def aff(o, J):
        return np.array(o, float), np.array(J, float).reshape(len(o), -1)
    if cell == "interval":  # ReferenceSimplex(1): face i opposite vertex i
        return [aff([1.0], [[0.0]]), aff([0.0], [[0.0]])]
    if cell == "triangle":
        return [aff([0, 1], [[1], [-1]]), aff([0, 0], [[0], [1]]), aff([0, 0], [[1], [0]])]
    if cell == "tetrahedron":
        return [aff([0, 0, 1], [[1, 0], [0, 1], [-1, -1]]),
                aff([0, 0, 0], [[0, 0], [0, 1], [1, 0]]),
                aff([0, 0, 0], [[1, 0], [0, 0], [0, 1]]),
                aff([0, 0, 0], [[1, 0], [0, 1], [0, 0]])]
    if cell == "quadrilateral":  # ReferenceCube(2); face parameter in [-1,1]
        return [aff([0, -1], [[1], [0]]), aff([1, 0], [[0], [1]]), aff([0, 1], [[-1], [0]]), aff([-1, 0], [[0], [-1]])]
    if cell == "hexahedron":
        return [aff([0, 0, -1], [[1, 0], [0, 1], [0, 0]]),
                aff([0, -1, 0], [[1, 0], [0, 0], [0, 1]]),
                aff([1, 0, 0], [[0, 0], [1, 0], [0, 1]]),
                aff([0, 1, 0], [[1, 0], [0, 0], [0, 1]]),
                aff([-1, 0, 0], [[0, 0], [1, 0], [0, 1]]),
                aff([0, 0, 1], [[1, 0], [0, 1], [0, 0]])]
    raise ValueError(cell)


def face_parameters(cell, rs):
    d = DIM[cell]
    cube = cell in ("quadrilateral", "hexahedron")
    if d == 1:
        return np.zeros((1, 1))
    if d == 2:
        return rs.uniform(-1.0, 1.0, size=(8, 1)) if cube else rs.uniform(0.0, 1.0, size=(8, 1))
    if cube:
        return rs.uniform(-1.0, 1.0, size=(12, 2))
    return rs.dirichlet(np.ones(3), size=12)[:, 1:]


def build(out):
    rs = np.random.RandomState(SEED)
    tables = {"meta/seed": np.array(SEED)}
    specs = []
    for cell in ("interval", "triangle", "tetrahedron"):
        for p in range(1, MAX_P[cell] + 1):
            for family in ("lagrange_equi", "lagrange_gll", "bernstein"):
                specs.append((family, cell, p))
    for cell in ("quadrilateral", "hexahedron"):
        for p in (1, 2):
            specs.append(("lagrange_equi", cell, p))
    # sample points are seeded per (cell, p), so every family at one (cell, p) shares them
    for family, cell, p in specs:
        sp = Space(family, cell, p)
        m = max(12, sp.n + 3)
        key = "%s/%s/p%d/" % (family, cell, p)
        local = np.random.RandomState(SEED + 1000 * DIM[cell] + 10 * p + (0 if cell in ('interval', 'triangle', 'tetrahedron') else 1))
        x = sample_points(cell, m, local)
        xi = face_parameters(cell, local)
        maps = face_maps(cell)
        fx = np.array([[o + J @ q for q in xi] for o, J in maps])
        fxi = np.array([xi for _ in maps])
        val, grad, hess = sp.evaluate(x)
        tables[key + "lattice"] = sp.lattice
        tables[key + "x"] = x
        tables[key + "val"] = val
        tables[key + "grad"] = grad
        tables[key + "hess"] = hess
        tables[key + "face_xi"] = fxi
        tables[key + "face_x"] = fx
        fv, fg, fh = [], [], []
        for f in range(len(maps)):
            v_, g_, h_ = sp.evaluate(fx[f])
            fv.append(v_)
            fg.append(g_)
            fh.append(h_)
        tables[key + "face_val"] = np.array(fv)
        tables[key + "face_grad"] = np.array(fg)
        tables[key + "face_hess"] = np.array(fh)
        if family != "bernstein":
            nv, _, _ = sp.evaluate(sp.lattice)
            tables[key + "node_val"] = nv
        print("%-14s %-13s p=%d  n=%2d  m=%2d" % (family, cell, p, sp.n, m))
        sys.stdout.flush()
    np.savez_compressed(out, **tables)
    print("wrote", out, "%.1f kB" % (os.path.getsize(out) / 1024.0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_tables.npz"))
    args = ap.parse_args()
    build(args.out)
