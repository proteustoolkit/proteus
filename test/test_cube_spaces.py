"""
Local tensor-product spaces Q_p of any degree on quadrilaterals and hexahedra (C8 round 4): node sets,
layout, orientation tables, tabulation.

`proteus.CubeSpaces` (numpy only) and its `FemTools` wrapper `PolynomialOnCube` are checked

* against the pinned oracle tables of `test/fem_oracle` (an independent path: monomials and a 50-digit
  solve there, a Legendre expansion and a double inverse here), through the same
  `ParametricFiniteElementSpace` routines the kernels use, on the cell and on every face: Q1..Q4 on the
  quadrilateral, Q1..Q3 on the hexahedron;
* against today's Q1/Q2 classes, including the local dof order where it is the same;
* on the properties that need no oracle, up to p = 7: nodality, partition of unity, exact reproduction of
  every tensor monomial of degree <= p per direction with gradients and Hessians, the entity layout, the
  face maps, the sub-cells;
* the orientation tables: every listing of one physical edge or face by an element gives the same index to
  the same physical node.
"""
import itertools
import os

import numpy as np
import pytest

from proteus import CubeSpaces as Q
from proteus import FemTools

HERE = os.path.dirname(os.path.abspath(__file__))
_data = np.load(os.path.join(HERE, "fem_oracle", "reference_tables.npz"))
ALL = {k: _data[k] for k in _data.files}
CELL = {2: "quadrilateral", 3: "hexahedron"}
# (d, p, nodes) -> oracle family; Q1, Q2: equispaced = GLL, stored as lagrange_equi
ORACLE = {}
for _d in (2, 3):
    for _p in (1, 2):
        for _n in ("gll", "equispaced"):
            ORACLE[(_d, _p, _n)] = "lagrange_equi"
for _p in (3, 4):
    ORACLE[(2, _p, "gll")] = "lagrange_gll"
    ORACLE[(2, _p, "equispaced")] = "lagrange_equi"
ORACLE[(3, 3, "gll")] = "lagrange_gll"
CASES = sorted(ORACLE)
CASE_IDS = ["%s-q%d-%s" % (CELL[d], p, n) for d, p, n in CASES]
GEN_MAX_P = {2: 7, 3: 5}
GEN_CASES = [(d, p, n) for d in (2, 3) for p in range(1, GEN_MAX_P[d] + 1) for n in ("gll", "equispaced")]
GEN_IDS = ["%s-q%d-%s" % (CELL[d], p, n) for d, p, n in GEN_CASES]
ATOL = 1.0e-12


def err(a, b):
    """Maximum difference scaled by the size of the reference."""
    return np.abs(a - b).max() / (1.0 + np.abs(b).max())


def T(d, p, nodes, item):
    return ALL["%s/%s/p%d/%s" % (ORACLE[(d, p, nodes)], CELL[d], p, item)]


def _match(points, lattice):
    """perm[i] = index in ``points`` of oracle lattice point i (a permutation)."""
    perm = np.array([np.abs(points - r).max(axis=1).argmin() for r in lattice])
    assert sorted(perm) == list(range(len(points)))
    assert np.abs(points[perm] - lattice).max() < 1.0e-13
    return perm


class _ReferenceOnlySpace(object):
    def __init__(self, lfs):
        self.referenceFiniteElement = FemTools.ReferenceFiniteElement(lfs, None)

    getBasisValuesRef = FemTools.ParametricFiniteElementSpace.getBasisValuesRef
    getBasisGradientValuesRef = FemTools.ParametricFiniteElementSpace.getBasisGradientValuesRef
    getBasisHessianValuesRef = FemTools.ParametricFiniteElementSpace.getBasisHessianValuesRef
    getBasisValuesTraceRef = FemTools.ParametricFiniteElementSpace.getBasisValuesTraceRef
    getBasisGradientValuesTraceRef = FemTools.ParametricFiniteElementSpace.getBasisGradientValuesTraceRef


# ----------------------------------------------------------------------------------------------
# against the oracle
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("d,p,nodes", CASES, ids=CASE_IDS)
def test_tabulation_matches_the_oracle_on_the_cell_and_every_face(d, p, nodes):
    s = Q.CubeSpace(d, p, nodes)
    perm = _match(s.dof_points, T(d, p, nodes, "lattice"))
    t = s.tabulate(T(d, p, nodes, "x"), 2)
    assert err(t.values[:, perm], T(d, p, nodes, "val")) < ATOL
    assert err(t.grads[:, perm], T(d, p, nodes, "grad")) < ATOL
    assert err(t.hessians[:, perm], T(d, p, nodes, "hess")) < ATOL
    face_xi = T(d, p, nodes, "face_xi")
    for f in range(face_xi.shape[0]):
        tf = s.tabulate_trace(f, face_xi[f], 2)
        assert err(tf.values[:, perm], T(d, p, nodes, "face_val")[f]) < ATOL, f
        assert err(tf.grads[:, perm], T(d, p, nodes, "face_grad")[f]) < ATOL, f
        assert err(tf.hessians[:, perm], T(d, p, nodes, "face_hess")[f]) < ATOL, f


@pytest.mark.parametrize("d,p,nodes", CASES, ids=CASE_IDS)
def test_femtools_routines_match_the_oracle(d, p, nodes):
    lfs = FemTools.PolynomialOnCube(d, p, nodes)
    assert lfs.nonzeroHessians and lfs.dim == (p + 1) ** d
    perm = _match(lfs.space.dof_points, T(d, p, nodes, "lattice"))
    ref = _ReferenceOnlySpace(lfs)
    x = T(d, p, nodes, "x")
    assert err(ref.getBasisValuesRef(x)[:, perm], T(d, p, nodes, "val")) < ATOL
    assert err(ref.getBasisGradientValuesRef(x)[:, perm], T(d, p, nodes, "grad")) < ATOL
    assert err(ref.getBasisHessianValuesRef(x)[:, perm], T(d, p, nodes, "hess")) < ATOL
    face_xi, face_x = T(d, p, nodes, "face_xi"), T(d, p, nodes, "face_x")
    elem = lfs.referenceElement
    for f in range(face_xi.shape[0]):
        for k in range(face_xi.shape[1]):
            assert np.abs(elem.boundaryMapList[f](face_xi[f, k]) - face_x[f, k]).max() < 1.0e-14
    psi = ref.getBasisValuesTraceRef(face_xi[0])
    grad = ref.getBasisGradientValuesTraceRef(face_xi[0])
    for f in range(face_xi.shape[0]):
        assert err(psi[f][:, perm], T(d, p, nodes, "face_val")[f]) < ATOL, f
        assert err(grad[f][:, perm], T(d, p, nodes, "face_grad")[f]) < ATOL, f
    # the array interface of the local space is the same tabulation
    assert np.abs(lfs.tabulate(x, 2).hessians - lfs.space.tabulate(x[:, :d], 2).hessians).max() == 0.0


# ----------------------------------------------------------------------------------------------
# against the existing classes
# ----------------------------------------------------------------------------------------------
def _values(lfs, pts):
    return (np.array([[b(x) for b in lfs.basis] for x in pts]),
            np.array([[g(x) for g in lfs.basisGradients] for x in pts]),
            np.array([[h(x) for h in lfs.basisHessians] for x in pts]))


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("nodes", ["gll", "equispaced"])
def test_q1_reproduces_the_existing_class_in_the_same_order(d, nodes):
    pts = np.random.RandomState(1).uniform(-1.0, 1.0, size=(7, d))
    old = _values(FemTools.LinearOnCubeWithNodalBasis(d), pts)
    new = _values(FemTools.PolynomialOnCube(d, 1, nodes), pts)
    for a, b in zip(new, old):
        assert np.abs(a - b).max() < 1.0e-14


@pytest.mark.parametrize("d", [2, 3])
def test_q2_is_the_space_of_the_existing_class(d):
    """Same functions; the same local order in 3D. In 2D the Q2 class of FemTools lists the vertices as
    (0, 3, 2, 1) and the edges as (3, 2, 1, 0) of the reference square."""
    pts = np.random.RandomState(2).uniform(-1.0, 1.0, size=(7, d))
    old_space = FemTools.LagrangeOnCubeWithNodalBasis(d, order=2)
    new_space = FemTools.PolynomialOnCube(d, 2)
    # old function j is 1 at the new space's node perm[j]
    K = np.array([[b(x) for b in old_space.basis] for x in new_space.space.dof_points])
    perm = K.argmax(axis=0)
    assert sorted(perm) == list(range(3 ** d)) and np.abs(K[perm, np.arange(3 ** d)] - 1.0).max() < 1.0e-13
    old, new = _values(old_space, pts), _values(new_space, pts)
    for a, b in zip(new, old):
        assert np.abs(a[:, perm] - b).max() < 1.0e-13
    if d == 3:
        assert list(perm) == list(range(27))
    else:
        assert list(perm) == [0, 3, 2, 1, 7, 6, 5, 4, 8]


# ----------------------------------------------------------------------------------------------
# properties that need no oracle, to p = 7 (quadrilateral) and p = 5 (hexahedron)
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("d,p,nodes", GEN_CASES, ids=GEN_IDS)
def test_layout_partition_of_unity_and_nodality(d, p, nodes):
    s = Q.CubeSpace(d, p, nodes)
    assert s.dim == (p + 1) ** d
    # the entities partition the dofs, with the counts 1, p-1, (p-1)^2, (p-1)^3
    listed = [i for group in s.entity_dofs for dofs in group for i in dofs]
    assert listed == list(range(s.dim))
    assert [len(g) for g in s.entity_dofs] == ([4, 4, 1] if d == 2 else [8, 12, 6, 1])
    for dim, group in enumerate(s.entity_dofs):
        assert all(len(dofs) == (p - 1) ** dim for dofs in group)
    # every dof point lies in the interior of its entity: its vertex weights vanish exactly off the entity
    w = Q.vertex_weights(d, s.dof_points)
    for dim, (vertex_lists, group) in enumerate(zip(s.entity_vertices, s.entity_dofs)):
        for verts, dofs in zip(vertex_lists, group):
            off = [v for v in range(2 ** d) if v not in verts]
            for i in dofs:
                assert (w[i, list(verts)] > 1.0e-6).all() and (not off or np.abs(w[i, off]).max() < 1.0e-15)
    # edge dofs count from the first vertex of the edge to the second
    corners = 2.0 * np.array(Q.CORNERS[d]) - 1.0
    for (a, b), dofs in zip(s.entity_vertices[1], s.entity_dofs[1]):
        along = (s.dof_points[dofs] - corners[a]) @ (corners[b] - corners[a])
        assert (np.diff(along) > 0.0).all()
    t = s.tabulate(s.dof_points, 2)
    assert np.abs(t.values - np.eye(s.dim)).max() < 1.0e-11
    pts = np.random.RandomState(3).uniform(-1.0, 1.0, size=(9, d))
    t = s.tabulate(pts, 2)
    assert np.abs(t.values.sum(axis=1) - 1.0).max() < 1.0e-11
    assert np.abs(t.grads.sum(axis=1)).max() < 1.0e-10
    assert np.abs(t.hessians.sum(axis=1)).max() < 1.0e-9
    # symmetric node sets (the orientation rule relies on it), and well-conditioned 1D tables
    assert np.abs(s.nodes1d + s.nodes1d[::-1]).max() < 1.0e-15
    assert s.condition_number < (5.0 if nodes == "gll" else 11.0)   # measured at p = 7: 4.44 and 10.62


@pytest.mark.parametrize("d,p,nodes", GEN_CASES, ids=GEN_IDS)
def test_every_tensor_polynomial_of_degree_p_is_reproduced(d, p, nodes):
    s = Q.CubeSpace(d, p, nodes)
    rs = np.random.RandomState(4)
    exps = list(itertools.product(range(p + 1), repeat=d))
    coef = rs.uniform(-1.0, 1.0, size=len(exps))

    def poly(x):
        q, g, h = np.zeros(len(x)), np.zeros((len(x), d)), np.zeros((len(x), d, d))
        for c, e in zip(coef, exps):
            e = np.array(e)

            def mono(de):
                r = e - de
                if (r < 0).any():
                    return np.zeros(len(x))
                f = np.prod([np.prod(np.arange(e[i] - de[i] + 1, e[i] + 1)) for i in range(d)])
                return f * np.prod(x ** r, axis=1)
            q += c * mono(np.zeros(d, int))
            for i in range(d):
                di = np.zeros(d, int)
                di[i] = 1
                g[:, i] += c * mono(di)
                for j in range(d):
                    dij = di.copy()
                    dij[j] += 1
                    h[:, i, j] += c * mono(dij)
        return q, g, h
    nodal = poly(s.dof_points)[0]
    pts = rs.uniform(-1.0, 1.0, size=(8, d))
    t = s.tabulate(pts, 2)
    q, g, h = poly(pts)
    scale = 1.0 + np.abs(coef).sum()
    assert np.abs(t.values @ nodal - q).max() < 1.0e-11 * scale
    assert np.abs(np.einsum("qid,i->qd", t.grads, nodal) - g).max() < 1.0e-10 * scale * p ** 2
    assert np.abs(np.einsum("qide,i->qde", t.hessians, nodal) - h).max() < 1.0e-9 * scale * p ** 4
    # negative control: x^(p+1) is not in the space
    high = pts[:, 0] ** (p + 1)
    assert np.abs(t.values @ (s.dof_points[:, 0] ** (p + 1)) - high).max() > 1.0e-4


@pytest.mark.parametrize("d", [2, 3])
def test_face_maps_and_on_face_are_those_of_the_reference_cube(d):
    elem = FemTools.ReferenceCube(d)
    xb = np.random.RandomState(5).uniform(-1.0, 1.0, size=(5, d - 1))
    s = Q.CubeSpace(d, 3)
    for f in range(2 * d):
        o, J = Q.face_map(d, f)
        axis, side = Q.FACE_PLANE[d][f]
        for x in xb:
            mapped = elem.boundaryMapList[f](x)
            assert np.abs(mapped - (o + J @ x)).max() < 1.0e-15
            assert mapped[axis] == 2.0 * side - 1.0
        assert np.abs(J - elem.boundaryJacobianList[f]).max() == 0.0
        # the face's vertices are the ones FACES lists, and on_face is the closure of the face
        corners = 2.0 * np.array(Q.CORNERS[d]) - 1.0
        on = sorted(v for v in range(2 ** d) if corners[v, axis] == 2.0 * side - 1.0)
        assert on == sorted(Q.FACES[d][f])
        for i in range(s.dim):
            assert s.on_face(i, f) == (abs(s.dof_points[i, axis] - (2.0 * side - 1.0)) < 1.0e-14)


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("p", [1, 2, 3, 5])
def test_subcells_tile_the_reference_cell(d, p):
    s = Q.CubeSpace(d, p, "equispaced")
    cells = Q.subcells(d, p)
    assert cells.shape == (p ** d, 2 ** d) and sorted(set(cells.ravel())) == list(range(s.dim))
    x = s.dof_points[cells]                       # (n, 2^d, d), vertices in reference-cell order
    if d == 2:
        e1, e2 = x[:, 1] - x[:, 0], x[:, 3] - x[:, 0]
        measure = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        assert np.abs(x[:, 2] - x[:, 1] - e2).max() < 1.0e-14   # a parallelogram, counter-clockwise
    else:
        e1, e2, e3 = x[:, 1] - x[:, 0], x[:, 3] - x[:, 0], x[:, 4] - x[:, 0]
        measure = np.einsum("ni,ni->n", np.cross(e1, e2), e3)
        assert np.abs(x[:, 6] - x[:, 0] - e1 - e2 - e3).max() < 1.0e-14
    assert (measure > 0.0).all() and abs(measure.sum() - 2.0 ** d) < 1.0e-12


# ----------------------------------------------------------------------------------------------
# orientation tables
# ----------------------------------------------------------------------------------------------
def _square_symmetries():
    """The eight listings (v0, v1, v2, v3), in cyclic order, of the corners of one physical square."""
    out = []
    for start in range(4):
        out.append([(start + i) % 4 for i in range(4)])
        out.append([(start - i) % 4 for i in range(4)])
    return out


@pytest.mark.parametrize("p", [2, 3, 4, 5, 7])
def test_edge_and_face_tables_give_one_index_per_physical_node(p):
    """An element sees a physical edge or face through its own listing of the vertices. Whatever the
    listing, the table must send the same physical node to the same entity index."""
    rs = np.random.RandomState(6)
    c = np.array([(0, 0), (1, 0), (1, 1), (0, 1)])       # physical corners of the face, lattice units / p
    codes, tables = Q.face_tables(p)
    assert tables.shape == (24, (p - 1) ** 2)
    for trial in range(6):
        g = rs.permutation(1000)[:4]                     # global numbers of the four physical corners
        index_of = []
        for listing in _square_symmetries():
            perm = np.argsort(g[listing])
            table = tables[codes[sum(int(perm[i]) * 4 ** i for i in range(4))]]
            o, a, b = c[listing[0]], c[listing[1]], c[listing[3]]
            found = {}
            for r in range(1, p):
                for s in range(1, p):
                    point = tuple(p * o + r * (a - o) + s * (b - o))
                    found[point] = int(table[(r - 1) * (p - 1) + (s - 1)])
            assert sorted(found.values()) == list(range((p - 1) ** 2))
            index_of.append(found)
        assert all(f == index_of[0] for f in index_of[1:])
        # the frame is the documented one: index 0 is next to the corner with the smallest global number,
        # and the first axis points to its neighbour with the smaller global number
        origin = int(np.argmin(g))
        n1, n2 = (origin + 1) % 4, (origin + 3) % 4
        first, second = (n1, n2) if g[n1] < g[n2] else (n2, n1)
        point = lambda rg, sg: tuple(p * c[origin] + rg * (c[first] - c[origin]) + sg * (c[second] - c[origin]))
        assert index_of[0][point(1, 1)] == 0
        if p >= 3:
            assert index_of[0][point(1, 2)] == 1 and index_of[0][point(2, 1)] == p - 1
    codes, tables = Q.edge_tables(p)
    assert list(tables[codes[0 + 2 * 1]]) == list(range(p - 1))          # first vertex has the smaller number
    assert list(tables[codes[1 + 2 * 0]]) == list(range(p - 1))[::-1]


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("p", [3, 4, 5])
def test_cell_interior_index_does_not_depend_on_the_listing_of_the_cell(d, p):
    """Two ranks can list a ghost cell through different symmetries of the square (8) or cube (48); the
    interior dof at one physical point must get one index."""
    corners = 2.0 * np.array(Q.CORNERS[d]) - 1.0
    listings = []
    for axes in itertools.permutations(range(d)):
        for signs in itertools.product((-1.0, 1.0), repeat=d):
            image = corners[:, list(axes)] * np.array(signs)
            listings.append([int(np.abs(corners - c).sum(axis=1).argmin()) for c in image])
    assert len(listings) == {2: 8, 3: 48}[d]
    s = Q.CubeSpace(d, p, "equispaced")
    interior = s.entity_dofs[d][0]
    w = Q.vertex_weights(d, s.dof_points[interior])            # (m, 2^d)
    rs = np.random.RandomState(8)
    for trial in range(4):
        g = rs.permutation(1000)[:2 ** d]                      # global numbers of the physical vertices
        maps = []
        for listing in listings:
            index = Q.cell_entity_index(d, p, g[listing][None, :])[0]
            assert sorted(index) == list(range((p - 1) ** d))
            points = np.round(w @ corners[listing] * p).astype(int)   # physical lattice point of each local dof
            maps.append({tuple(pt): int(k) for pt, k in zip(points.tolist(), index)})
        assert all(m == maps[0] for m in maps[1:])
        # the frame: index 0 is the interior node nearest the vertex with the smallest global number
        o = int(np.argmin(g))
        assert maps[0][tuple((corners[o] * (p - 2)).astype(int))] == 0


# ----------------------------------------------------------------------------------------------
# node sets
# ----------------------------------------------------------------------------------------------
# 1D Lebesgue constants (the d-dimensional one is the d-th power), measured 2026-11-14 on 2001 points
LEBESGUE_1D = {1: (1.000, 1.000), 2: (1.250, 1.250), 3: (1.631, 1.500), 4: (2.208, 1.636), 5: (3.106, 1.779),
               6: (4.549, 1.874), 7: (6.930, 1.972)}


@pytest.mark.parametrize("p", sorted(LEBESGUE_1D))
def test_lebesgue_constants(p):
    eq = Q.CubeSpace(2, p, "equispaced").lebesgue_constant(2000) ** 0.5
    gl = Q.CubeSpace(2, p, "gll").lebesgue_constant(2000) ** 0.5
    assert eq == pytest.approx(LEBESGUE_1D[p][0], abs=2.0e-3)
    assert gl == pytest.approx(LEBESGUE_1D[p][1], abs=2.0e-3)
    assert gl <= eq + 1.0e-12
    assert Q.CubeSpace(3, p, "gll").lebesgue_constant(2000) == pytest.approx(gl ** 3, rel=1.0e-12)


def test_arguments_are_checked():
    for bad in (dict(d=1, order=2), dict(d=2, order=0), dict(d=2, order=2, nodes="fekete")):
        with pytest.raises(ValueError):
            Q.CubeSpace(**bad)
