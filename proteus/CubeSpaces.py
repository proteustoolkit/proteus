"""
Local tensor-product spaces Q_p of any degree on the reference square and cube: node sets, entity-grouped
dof layout, entity orientation tables and tabulation (values, gradients, Hessians, traces).

This module is numpy only and knows nothing about meshes; ``FemTools`` wraps it (``PolynomialOnCube``,
``CubeEntityDOFMap``, ``C0_AffinePolynomialOnCube``). It is the counterpart of :mod:`proteus.SimplexSpaces`.

Reference cell: ``FemTools.ReferenceCube``, [-1,1]^d, d = 2 or 3. Vertices (-1,-1), (1,-1), (1,1), (-1,1) in
2D; in 3D those at z = -1 (0..3) and then at z = +1 (4..7). Faces as ``ReferenceCube`` numbers them: the
edges (0,1), (1,2), (2,3), (3,0) in 2D; in 3D 0: z = -1 (0-1-2-3), 1: y = -1 (0-1-5-4), 2: x = +1 (1-2-6-5),
3: y = +1 (2-3-7-6), 4: x = -1 (3-0-4-7), 5: z = +1 (4-5-6-7).

A local dof is the value at the tensor node ``(x_{i_1}, .., x_{i_d})``, identified by its multi-index
``(i_1, .., i_d)``, 0 <= i <= p, with ``x`` the p+1 Gauss-Lobatto-Legendre points (``nodes='gll'``, the
default) or equispaced points (``nodes='equispaced'``, for comparison; they coincide for p <= 2).

Local order (grouped by entity): the vertices; the edges (0,1), (1,2), (2,3), (3,0) and, in 3D,
(0,4), (1,5), (2,6), (3,7), (4,5), (5,6), (6,7), (7,4); in 3D the faces 0..5; the cell interior.
On an edge (a, b) the p-1 dofs count from a to b. On a face (v0, v1, v2, v3) the (p-1)^2 dofs are indexed
(r, s), r along v0 -> v1 and s along v0 -> v3, r outer. In the interior x is the fastest index.
For p = 1 this is the numbering of ``LinearOnCubeWithNodalBasis`` and for p = 2 in 3D that of
``LagrangeOnCubeWithNodalBasis``; the 2D Q2 class of FemTools lists the vertices as (0, 3, 2, 1) and the edges
as (3, 2, 1, 0), so there the space is the same and the local order is not.

Orientation (used by the dof map)
---------------------------------
The dofs of an edge or a face shared by several elements are indexed in the entity's own frame, built from
*global* vertex numbers, so that every element refers to the same physical node by the same index:

* edge: from the vertex with the smaller global number to the other;
* quadrilateral face: the origin is the vertex with the smallest global number; the first axis points to
  whichever of its two neighbours on the face has the smaller global number, the second axis to the other.

A reversed axis maps node i to p - i, which is the same point because both node sets are symmetric.
``edge_tables`` and ``face_tables`` give, for every ordering of the entity's vertices by global number, the
entity index of each local position. The interior dofs of a cell are indexed the same way
(``cell_entity_index``: origin at the vertex with the smallest global number, axes to its neighbours sorted
by global number), so that the ranks holding a ghost cell agree on them whatever their listing of it.

Tabulation
----------
The 1D Lagrange functions are expanded in Legendre polynomials (inverse of the Legendre Vandermonde matrix
at the nodes; its condition number at p = 7 is 4.4 for GLL and 10.6 for equispaced nodes); values and
derivatives are tensor products. The test
oracle uses monomials and a 50-digit solve, so the two paths are independent.
"""
import itertools

import numpy as np
from numpy.polynomial import legendre

from .SimplexSpaces import Tabulation, gll_points

# corner of vertex v as 0/1 per direction (0: -1, 1: +1)
CORNERS = {2: ((0, 0), (1, 0), (1, 1), (0, 1)),
           3: ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))}
EDGES = {2: ((0, 1), (1, 2), (2, 3), (3, 0)),
         3: ((0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5), (2, 6), (3, 7), (4, 5), (5, 6), (6, 7), (7, 4))}
FACES = {2: EDGES[2],
         3: ((0, 1, 2, 3), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7))}
# face f lies in the plane x[axis] = -1 (side 0) or +1 (side 1)
FACE_PLANE = {2: ((1, 0), (0, 1), (1, 1), (0, 0)),
              3: ((2, 0), (1, 0), (0, 1), (1, 1), (0, 0), (2, 1))}


def entities(d):
    """entity_vertices[dim][n]: local vertices of entity n of dimension dim (dim = 0..d)."""
    ev = [tuple((v,) for v in range(2 ** d)), EDGES[d]]
    if d == 3:
        ev.append(FACES[3])
    ev.append((tuple(range(2 ** d)),))
    return ev


def multi_indices(d, p):
    """The local dofs in entity-grouped order: (index, entity_dofs) with index (dim, d) int in 0..p and
    entity_dofs[dim][n] the local dofs interior to entity n of dimension dim."""
    corner = np.array(CORNERS[d])
    index, entity_dofs = [], []

    def add(group):
        dofs = list(range(len(index), len(index) + len(group)))
        index.extend(group)
        return dofs
    entity_dofs.append([add([tuple(p * corner[v])]) for v in range(2 ** d)])
    entity_dofs.append([add([tuple(p * corner[a] + m * (corner[b] - corner[a])) for m in range(1, p)])
                        for a, b in EDGES[d]])
    if d == 3:
        entity_dofs.append([add([tuple(p * corner[v0] + r * (corner[v1] - corner[v0]) + s * (corner[v3] - corner[v0]))
                                 for r in range(1, p) for s in range(1, p)])
                            for v0, v1, v2, v3 in FACES[3]])
    interior = [t[::-1] for t in itertools.product(range(1, p), repeat=d)]  # x fastest
    entity_dofs.append([add(interior)])
    index = np.array(index, dtype=int).reshape(-1, d)
    assert index.shape[0] == (p + 1) ** d and len({tuple(i) for i in index.tolist()}) == (p + 1) ** d
    return index, entity_dofs


def edge_tables(p):
    """(codes, tables) for an edge: ``tables[q][k]`` is the entity index of the dof at local position k when
    the two vertices sort by global number with permutation q; ``codes[perm[0] + 2 perm[1]] = q``."""
    codes = np.zeros((4,), 'i8')
    tables = np.zeros((2, p - 1), 'i8')
    codes[0 + 2 * 1] = 0
    codes[1 + 2 * 0] = 1
    tables[0] = np.arange(p - 1)
    tables[1] = np.arange(p - 1)[::-1]
    return codes, tables


def face_tables(p):
    """(codes, tables) for a quadrilateral face (v0, v1, v2, v3), vertices in cyclic order: for each
    permutation ``perm`` sorting the four vertices by global number, ``tables[q][k]`` is the entity index of
    the dof at local position k = (r-1)(p-1) + (s-1); ``codes[sum_i perm[i] 4^i] = q``."""
    c = np.array([(0, 0), (1, 0), (1, 1), (0, 1)])
    perms = list(itertools.permutations(range(4)))
    codes = np.zeros((4 ** 4,), 'i8')
    tables = np.zeros((len(perms), (p - 1) ** 2), 'i8')
    for q, perm in enumerate(perms):
        codes[sum(perm[i] * 4 ** i for i in range(4))] = q
        o = perm[0]
        n1, n2 = (o + 1) % 4, (o + 3) % 4
        A, B = (n1, n2) if perm.index(n1) < perm.index(n2) else (n2, n1)
        for r in range(1, p):
            for s in range(1, p):
                rel = np.array([r, s]) - p * c[o]
                rg, sg = int(rel @ (c[A] - c[o])), int(rel @ (c[B] - c[o]))
                assert 1 <= rg <= p - 1 and 1 <= sg <= p - 1
                tables[q, (r - 1) * (p - 1) + (s - 1)] = (rg - 1) * (p - 1) + (sg - 1)
        assert sorted(tables[q]) == list(range((p - 1) ** 2))
    return codes, tables


def cell_entity_index(d, p, g):
    """Entity index (n, (p-1)^d) of the interior dofs of cells whose vertices, in reference order, have the
    global numbers ``g`` (n, 2^d): the dof at local position k (x fastest) gets its index in the cell's
    global frame. The origin is the vertex with the smallest global number; the axes point to its d
    neighbours along the cell's edges, sorted by global number; the first axis is the slowest index.
    Two ranks that list a ghost cell through different symmetries of the cell then agree on its dofs."""
    g = np.asarray(g)
    corner = np.array(CORNERS[d])
    # neighbour of vertex v along axis a
    neighbour = np.array([[int(np.abs(corner - (c + (1 - 2 * c) * np.eye(d, dtype=int)[a])).sum(axis=1).argmin())
                           for a in range(d)] for c in corner])
    interior = np.array([t[::-1] for t in itertools.product(range(1, p), repeat=d)])   # (m, d), local positions
    orders = list(itertools.permutations(range(d)))
    tables = np.zeros((2 ** d, len(orders), len(interior)), 'i8')
    for o in range(2 ** d):
        for q, order in enumerate(orders):
            index = np.zeros(len(interior), 'i8')
            for a in order:   # the axis of the neighbour with the m-th smallest global number
                t = (interior - p * corner[o]) @ (corner[neighbour[o, a]] - corner[o])
                assert t.min() >= 1 and t.max() <= p - 1
                index = index * (p - 1) + (t - 1)
            tables[o, q] = index
            assert sorted(index) == list(range(len(interior)))
    rows = np.arange(g.shape[0])
    origin = np.argmin(g, axis=1)
    sort = np.argsort(g[rows[:, None], neighbour[origin]], axis=1, kind='stable')     # (n, d): axes by global number
    code = {order: q for q, order in enumerate(orders)}
    q = np.array([code[tuple(r)] for r in sort.tolist()], dtype='i8').reshape(-1)
    return tables[origin, q]


def face_map(d, f):
    """The affine map x = o + J xBar of ``ReferenceCube.boundaryMapList[f]`` (xBar in [-1,1]^(d-1)): (o, J)."""
    if d == 2:
        return [(np.array([0.0, -1.0]), np.array([[1.0], [0.0]])),
                (np.array([1.0, 0.0]), np.array([[0.0], [1.0]])),
                (np.array([0.0, 1.0]), np.array([[-1.0], [0.0]])),
                (np.array([-1.0, 0.0]), np.array([[0.0], [-1.0]]))][f]
    return [(np.array([0.0, 0.0, -1.0]), np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])),
            (np.array([0.0, -1.0, 0.0]), np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])),
            (np.array([1.0, 0.0, 0.0]), np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])),
            (np.array([0.0, 1.0, 0.0]), np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])),
            (np.array([-1.0, 0.0, 0.0]), np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])),
            (np.array([0.0, 0.0, 1.0]), np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))][f]


def vertex_weights(d, points):
    """The multilinear (Q1) shape functions of the 2^d vertices at reference points (n, d): (n, 2^d)."""
    points = np.asarray(points, dtype=float).reshape(-1, d)
    sign = 2.0 * np.array(CORNERS[d]) - 1.0
    return np.prod(0.5 * (1.0 + sign[None, :, :] * points[:, None, :]), axis=2)


def subcells(d, p):
    """The p^d linear sub-cells of the tensor lattice as local dof indices (n, 2^d), vertices in the order of
    the reference cell: the piecewise-multilinear view of the dofs that the archive writes."""
    mi, _ = multi_indices(d, p)
    index = {tuple(a): i for i, a in enumerate(mi.tolist())}
    corner = np.array(CORNERS[d])
    cells = [[index[tuple(np.array(base[::-1]) + c)] for c in corner]
             for base in itertools.product(range(p), repeat=d)]
    return np.array(cells, dtype=int).reshape(p ** d, 2 ** d)


class CubeSpace(object):
    """Q_p on the reference square or cube with the nodal (Lagrange) basis on tensor nodes.

    Attributes: ``d``, ``order``, ``family`` (``'lagrange'``), ``nodes``, ``dim``, ``multi_index`` (dim, d),
    ``entity_dofs[dim][n]``, ``entity_vertices[dim][n]``, ``nodes1d`` (p+1,), ``dof_points`` (dim, d),
    ``lagrange_points`` (the same array), ``condition_number`` (of the 1D Legendre Vandermonde matrix).
    """
    family = "lagrange"

    def __init__(self, d, order, nodes="gll"):
        if d not in (2, 3):
            raise ValueError("d must be 2 or 3 (the interval is the 1D simplex)")
        if order < 1:
            raise ValueError("order must be >= 1")
        if nodes not in ("gll", "equispaced"):
            raise ValueError("nodes must be 'gll' or 'equispaced'")
        self.d, self.order, self.nodes = d, order, nodes
        self.multi_index, self.entity_dofs = multi_indices(d, order)
        self.entity_vertices = entities(d)
        self.dim = self.multi_index.shape[0]
        self.nodes1d = gll_points(order) if nodes == "gll" else np.linspace(-1.0, 1.0, order + 1)
        self.dof_points = self.nodes1d[self.multi_index]
        self.lagrange_points = self.dof_points
        V = legendre.legvander(self.nodes1d, order)
        self.condition_number = np.linalg.cond(V)
        # columns: Legendre coefficients of the 1D Lagrange functions and of their first two derivatives
        self._c = [np.linalg.inv(V)]
        for m in (1, 2):
            dc = legendre.legder(self._c[0], m=m, axis=0) if order >= m else np.zeros((1, order + 1))
            self._c.append(dc)

    def _line(self, x, m):
        """m-th derivative of the 1D Lagrange functions at x (n,): (n, p+1)."""
        c = self._c[m]
        return legendre.legvander(x, c.shape[0] - 1) @ c

    def tabulate(self, points, derivatives=2):
        """Basis values, reference gradients and Hessians at points (nQ, d) of the reference cell."""
        d, mi = self.d, self.multi_index
        pts = np.asarray(points, dtype=float).reshape(-1, d)
        L = [[self._line(pts[:, a], m)[:, mi[:, a]] for m in range(derivatives + 1)] for a in range(d)]

        def product(orders):
            out = L[0][orders[0]]
            for a in range(1, d):
                out = out * L[a][orders[a]]
            return out
        vals = product([0] * d)
        grads = hess = None
        if derivatives >= 1:
            grads = np.zeros(vals.shape + (d,))
            for a in range(d):
                o = [0] * d
                o[a] = 1
                grads[:, :, a] = product(o)
        if derivatives >= 2:
            hess = np.zeros(vals.shape + (d, d))
            for a in range(d):
                for b in range(a, d):
                    o = [0] * d
                    o[a] += 1
                    o[b] += 1
                    hess[:, :, a, b] = hess[:, :, b, a] = product(o)
        return Tabulation(vals, grads, hess)

    def tabulate_trace(self, face, points_bar, derivatives=1):
        """The element tabulation at the points of face ``face`` with face coordinates points_bar
        (nQ, d-1) in [-1,1]^(d-1): full (not tangential) gradients, evaluated at the mapped points."""
        o, J = face_map(self.d, face)
        xb = np.asarray(points_bar, dtype=float).reshape(-1, J.shape[1])
        return self.tabulate(o + xb @ J.T, derivatives)

    def on_face(self, i, face):
        """True if local dof i lies on (the closure of) face ``face``."""
        axis, side = FACE_PLANE[self.d][face]
        return self.multi_index[i, axis] == side * self.order

    def lebesgue_constant(self, n=200):
        """max_x sum_i |phi_i(x)|: the d-th power of the 1D constant, sampled with n intervals."""
        x = np.linspace(-1.0, 1.0, n + 1)
        return float(np.abs(self._line(x, 0)).sum(axis=1).max() ** self.d)
