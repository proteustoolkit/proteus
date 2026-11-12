"""
Local polynomial spaces of any degree on simplices (C8 round 2): node sets, layout, tabulation.

`proteus.SimplexSpaces` (numpy only) and its `FemTools` wrapper `PolynomialOnSimplex` are checked

* against the pinned oracle tables of `test/fem_oracle` (an independent path: monomial basis and a
  50-digit solve there, Bernstein prime basis and a double LU here), through the same
  `ParametricFiniteElementSpace` routines the kernels use, on the cell and on every face;
* against today's P1/P2 classes, including the local dof order, so that no existing pin can move;
* on the properties that need no oracle, up to p = 7 (the generator range): nodality, partition of unity,
  exact reproduction of every monomial of degree <= p with gradients and Hessians, symmetry and
  equivariance of the node sets, GLL points on the edges, the entity layout.

It also records the measured Lebesgue constants and the comparison that decided the tetrahedron node set
(published warp-and-blend construction against the notebook's `nodes3d`).
"""
import importlib.util
import itertools
import os
from math import factorial

import numpy as np
import pytest

from proteus import FemTools
from proteus import SimplexSpaces as S

HERE = os.path.dirname(os.path.abspath(__file__))
_data = np.load(os.path.join(HERE, "fem_oracle", "reference_tables.npz"))
ALL = {k: _data[k] for k in _data.files}
CELL = {1: "interval", 2: "triangle", 3: "tetrahedron"}
CI_MAX_P = {1: 5, 2: 5, 3: 4}      # degrees with oracle tables (the CI-supported range)
GEN_MAX_P = 7                      # the generator range
ORACLE_FAMILY = {("lagrange", "gll"): "lagrange_gll", ("lagrange", "equispaced"): "lagrange_equi",
                 ("bernstein", "gll"): "bernstein"}
ATOL = 1.0e-12


def err(a, b):
    """Maximum difference scaled by the size of the reference (second derivatives at p = 5 reach 1e3)."""
    return np.abs(a - b).max() / (1.0 + np.abs(b).max())

CASES = [(d, p, family, nodes) for d in (1, 2, 3) for p in range(1, CI_MAX_P[d] + 1)
         for (family, nodes) in ORACLE_FAMILY]
CASE_IDS = ["%s-%s-%s-p%d" % (CELL[d], f, n, p) for d, p, f, n in CASES]
GEN_CASES = [(d, p, family, nodes) for d in (1, 2, 3) for p in range(1, GEN_MAX_P + 1)
             for (family, nodes) in ORACLE_FAMILY]
GEN_IDS = ["%s-%s-%s-p%d" % (CELL[d], f, n, p) for d, p, f, n in GEN_CASES]


def T(d, p, family, nodes, item):
    return ALL["%s/%s/p%d/%s" % (ORACLE_FAMILY[(family, nodes)], CELL[d], p, item)]


def _match(points, lattice):
    """perm[i] = index in ``points`` of oracle lattice point i (a permutation)."""
    perm = np.array([np.abs(points - r).max(axis=1).argmin() for r in lattice])
    assert sorted(perm) == list(range(len(points)))
    assert np.abs(points[perm] - lattice).max() < 1.0e-13, np.abs(points[perm] - lattice).max()
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
@pytest.mark.parametrize("d,p,family,nodes", CASES, ids=CASE_IDS)
def test_tabulation_matches_the_oracle_on_the_cell_and_every_face(d, p, family, nodes):
    lfs = FemTools.PolynomialOnSimplex(d, p, family, nodes)
    perm = _match(lfs.space.dof_points, T(d, p, family, nodes, "lattice"))
    tab = lfs.tabulate(T(d, p, family, nodes, "x"), 2)
    assert err(tab.values[:, perm], T(d, p, family, nodes, "val")) < ATOL
    assert err(tab.grads[:, perm], T(d, p, family, nodes, "grad")) < ATOL
    assert err(tab.hessians[:, perm], T(d, p, family, nodes, "hess")) < ATOL
    face_xi = T(d, p, family, nodes, "face_xi")
    for f in range(d + 1):
        tr = lfs.tabulate_trace(f, face_xi[f], 2)
        assert err(tr.values[:, perm], T(d, p, family, nodes, "face_val")[f]) < ATOL
        assert err(tr.grads[:, perm], T(d, p, family, nodes, "face_grad")[f]) < ATOL
        assert err(tr.hessians[:, perm], T(d, p, family, nodes, "face_hess")[f]) < ATOL


@pytest.mark.parametrize("d,p,family,nodes", CASES, ids=CASE_IDS)
def test_femtools_routines_match_the_oracle(d, p, family, nodes):
    """The per-function callables, through the routines that feed the kernels."""
    lfs = FemTools.PolynomialOnSimplex(d, p, family, nodes)
    perm = _match(lfs.space.dof_points, T(d, p, family, nodes, "lattice"))
    ref = _ReferenceOnlySpace(lfs)
    x = T(d, p, family, nodes, "x")
    assert err(ref.getBasisValuesRef(x)[:, perm], T(d, p, family, nodes, "val")) < ATOL
    assert err(ref.getBasisGradientValuesRef(x)[:, perm], T(d, p, family, nodes, "grad")) < ATOL
    assert err(ref.getBasisHessianValuesRef(x)[:, perm], T(d, p, family, nodes, "hess")) < ATOL
    xi = T(d, p, family, nodes, "face_xi")[0]
    assert err(ref.getBasisValuesTraceRef(xi)[:, :, perm], T(d, p, family, nodes, "face_val")) < ATOL
    assert err(ref.getBasisGradientValuesTraceRef(xi)[:, :, perm], T(d, p, family, nodes, "face_grad")) < ATOL
    # the trace lists kept for old callers
    f = d
    assert lfs.basisTrace[f][0](xi[0]) == pytest.approx(T(d, p, family, nodes, "face_val")[f, 0, np.where(perm == 0)[0][0]], abs=ATOL)


# ----------------------------------------------------------------------------------------------
# against today's P1 and P2 classes, including the dof order
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("nodes", ["gll", "equispaced"])
def test_p1_p2_reproduce_the_existing_classes_in_the_same_order(d, p, nodes):
    old = FemTools.LinearOnSimplexWithNodalBasis(d) if p == 1 else FemTools.QuadraticOnSimplexWithNodalBasis(d)
    new = FemTools.PolynomialOnSimplex(d, p, "lagrange", nodes)
    assert new.dim == old.dim
    x = T(d, p, "lagrange", "gll", "x")
    xi = T(d, p, "lagrange", "gll", "face_xi")[0]
    a, b = _ReferenceOnlySpace(old), _ReferenceOnlySpace(new)
    assert np.abs(a.getBasisValuesRef(x) - b.getBasisValuesRef(x)).max() < 1.0e-14
    assert np.abs(a.getBasisGradientValuesRef(x) - b.getBasisGradientValuesRef(x)).max() < 1.0e-13
    assert np.abs(a.getBasisHessianValuesRef(x) - b.getBasisHessianValuesRef(x)).max() < 1.0e-12
    assert np.abs(a.getBasisValuesTraceRef(xi) - b.getBasisValuesTraceRef(xi)).max() < 1.0e-14
    assert np.abs(a.getBasisGradientValuesTraceRef(xi) - b.getBasisGradientValuesTraceRef(xi)).max() < 1.0e-13
    # and the interpolation points of the existing conditions are the new dof points, in order
    ic = (FemTools.NodalInterpolationConditions if p == 1 else FemTools.QuadraticLagrangeNodalInterpolationConditions)(old.referenceElement)
    assert np.abs(ic.quadraturePointArray[:, :d] - new.space.dof_points).max() < 1.0e-15


@pytest.mark.parametrize("d", [1, 2, 3])
def test_bernstein_p2_reproduces_the_existing_class_in_the_same_order(d):
    old = FemTools.BernsteinOnSimplex(d)
    new = FemTools.PolynomialOnSimplex(d, 2, "bernstein")
    x = T(d, 2, "bernstein", "gll", "x")
    a, b = _ReferenceOnlySpace(old), _ReferenceOnlySpace(new)
    assert np.abs(a.getBasisValuesRef(x) - b.getBasisValuesRef(x)).max() < 1.0e-14
    assert np.abs(a.getBasisGradientValuesRef(x) - b.getBasisGradientValuesRef(x)).max() < 1.0e-13


# ----------------------------------------------------------------------------------------------
# properties that need no oracle, to p = 7
# ----------------------------------------------------------------------------------------------
def _random_points(d, n, seed=3):
    return np.random.RandomState(seed).dirichlet(np.ones(d + 1), size=n)[:, 1:]


@pytest.mark.parametrize("d,p,family,nodes", GEN_CASES, ids=GEN_IDS)
def test_layout_partition_of_unity_and_nodality(d, p, family, nodes):
    s = S.SimplexSpace(d, p, family, nodes)
    assert s.dim == factorial(p + d) // (factorial(p) * factorial(d))
    # entity counts: 1 per vertex, p-1 per edge, (p-1)(p-2)/2 per triangle, (p-1)(p-2)(p-3)/6 per tet
    per = {0: 1, 1: p - 1, 2: (p - 1) * (p - 2) // 2, 3: (p - 1) * (p - 2) * (p - 3) // 6}
    seen = []
    for dim, ents in enumerate(s.entity_dofs):
        assert len(ents) == len(s.entity_vertices[dim])
        for dofs, verts in zip(ents, s.entity_vertices[dim]):
            assert len(dofs) == per[dim]
            for i in dofs:  # the dof's multi-index is supported exactly on the entity's vertices
                assert set(np.nonzero(s.multi_index[i])[0]) == set(verts)
            seen += dofs
    assert seen == list(range(s.dim))  # grouped by entity, in order
    assert (s.multi_index.sum(axis=1) == p).all()
    tab = s.tabulate(_random_points(d, 20), 2)
    assert np.abs(tab.values.sum(axis=1) - 1.0).max() < ATOL
    assert np.abs(tab.grads.sum(axis=1)).max() < 1.0e-10
    assert np.abs(tab.hessians.sum(axis=1)).max() < 1.0e-9
    if family == "lagrange":
        assert np.abs(s.tabulate(s.dof_points, 0).values - np.eye(s.dim)).max() < 1.0e-11
    else:
        assert tab.values.min() >= 0.0
        assert np.abs(tab.values @ s.dof_points - _random_points(d, 20)).max() < ATOL  # linear precision


def _poly(exps, coef, x):
    m, d = x.shape
    q, gq, hq = np.zeros(m), np.zeros((m, d)), np.zeros((m, d, d))
    for c, a in zip(coef, exps):
        a = np.array(a)

        def mono(da):
            e = a - da
            if (e < 0).any():
                return np.zeros(m)
            f = 1.0
            for i in range(d):
                for s_ in range(da[i]):
                    f *= a[i] - s_
            return f * np.prod(x ** e, axis=1)
        q += c * mono(np.zeros(d, int))
        for i in range(d):
            di = np.zeros(d, int)
            di[i] = 1
            gq[:, i] += c * mono(di)
            for j in range(d):
                dij = di.copy()
                dij[j] += 1
                hq[:, i, j] += c * mono(dij)
    return q, gq, hq


@pytest.mark.parametrize("d,p,family,nodes", [c for c in GEN_CASES if c[2] == "lagrange"],
                         ids=[i for c, i in zip(GEN_CASES, GEN_IDS) if c[2] == "lagrange"])
def test_every_polynomial_of_degree_p_is_reproduced(d, p, family, nodes):
    s = S.SimplexSpace(d, p, family, nodes)
    exps = [a for a in itertools.product(range(p + 1), repeat=d) if sum(a) <= p]
    coef = np.random.RandomState(11).uniform(-1.0, 1.0, size=len(exps))
    c = _poly(exps, coef, s.dof_points)[0]  # nodal interpolant
    x = _random_points(d, 15, seed=5)
    q, gq, hq = _poly(exps, coef, x)
    tab = s.tabulate(x, 2)
    assert np.abs(tab.values @ c - q).max() < 1.0e-11
    assert np.abs(np.einsum("j,mjd->md", c, tab.grads) - gq).max() < 1.0e-10
    assert np.abs(np.einsum("j,mjde->mde", c, tab.hessians) - hq).max() < 1.0e-9
    for f in range(d + 1):
        xb = _random_points(max(d - 1, 1), 5, seed=7) if d > 1 else np.zeros((1, 1))
        o, J = S.face_map(d, f)
        fq, fgq, _ = _poly(exps, coef, o + xb @ J.T)
        tr = s.tabulate_trace(f, xb, 1)
        assert np.abs(tr.values @ c - fq).max() < 1.0e-11
        assert np.abs(np.einsum("j,mjd->md", c, tr.grads) - fgq).max() < 1.0e-10


@pytest.mark.parametrize("d", [1, 2, 3])
def test_face_maps_are_the_reference_simplex_maps(d):
    elem = FemTools.ReferenceSimplex(d)
    for f in range(d + 1):
        o, J = S.face_map(d, f)
        for xb in (_random_points(max(d - 1, 1), 4) if d > 1 else np.zeros((1, 1))):
            assert np.abs(elem.boundaryMapList[f](xb) - (o + J @ xb)).max() < 1.0e-15


# ----------------------------------------------------------------------------------------------
# node sets
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("p", range(1, GEN_MAX_P + 1))
def test_warp_blend_nodes_are_equivariant_and_gll_on_the_edges(d, p):
    """X(sigma alpha) = sigma X(alpha) for every vertex permutation: what entity orientation relies on."""
    mi, entity_dofs = S.multi_indices(d, p)
    X = S.warp_blend_barycentric(d, p)
    index = {tuple(a): i for i, a in enumerate(mi.tolist())}
    assert np.abs(X.sum(axis=1) - 1.0).max() < 1.0e-14 and X.min() > -1.0e-14
    worst = 0.0
    for sigma in itertools.permutations(range(d + 1)):
        sigma = list(sigma)
        image = np.array([index[tuple(a)] for a in mi[:, sigma].tolist()])
        worst = max(worst, np.abs(X[image] - X[:, sigma]).max())
    assert worst < 1.0e-14, worst
    g = 0.5 * (S.gll_points(p) + 1.0)
    for dofs, (a, b) in zip(entity_dofs[1] if p > 1 else [], S.EDGES[d]):
        # the k-th dof of edge (a, b) is the k-th interior GLL point counted from a
        assert np.abs(X[dofs, b] - g[1:-1]).max() < 1.0e-14
        assert np.abs(X[dofs, a] + X[dofs, b] - 1.0).max() < 1.0e-14
    if p <= 2:
        assert np.abs(X - mi / float(p)).max() == 0.0


def test_gll_points():
    for p in range(1, 13):
        x = S.gll_points(p)
        assert len(x) == p + 1 and x[0] == -1.0 and x[-1] == 1.0
        assert np.abs(x + x[::-1]).max() == 0.0
        # the interior points are the roots of P_p'
        dL = np.polynomial.legendre.Legendre.basis(p).deriv()
        assert p == 1 or np.abs(dL(x[1:-1])).max() < 1.0e-11
    assert np.abs(S.gll_points(3) - np.array([-1.0, -1.0 / np.sqrt(5.0), 1.0 / np.sqrt(5.0), 1.0])).max() < 1.0e-15


def test_warp_blend_needs_a_blend_parameter_above_p7():
    with pytest.raises(ValueError, match="warp-and-blend"):
        S.SimplexSpace(2, 8, "lagrange", "gll")
    S.SimplexSpace(2, 8, "lagrange", "equispaced")
    S.SimplexSpace(1, 12, "lagrange", "gll")  # 1D GLL needs no table


# Lebesgue constants sampled on a lattice (120 intervals per edge on the triangle, 40 on the tetrahedron);
# sampling can only underestimate. Measured 2026-11-12.
LEBESGUE = {
    # (d, p): (equispaced, warp-and-blend)
    (2, 1): (1.000, 1.000), (2, 2): (1.667, 1.667), (2, 3): (2.270, 2.112), (2, 4): (3.473, 2.661),
    (2, 5): (5.443, 3.121), (2, 6): (8.726, 3.697), (2, 7): (14.335, 4.273),
    (3, 1): (1.000, 1.000), (3, 2): (2.000, 2.000), (3, 3): (3.019, 2.922), (3, 4): (4.859, 4.056),
    (3, 5): (8.052, 5.292), (3, 6): (13.396, 6.952), (3, 7): (23.299, 9.204),
}


@pytest.mark.parametrize("d,p", sorted(LEBESGUE))
def test_lebesgue_constants(d, p):
    n = 120 if d == 2 else 40
    eq = S.SimplexSpace(d, p, "lagrange", "equispaced").lebesgue_constant(n)
    wb = S.SimplexSpace(d, p, "lagrange", "gll").lebesgue_constant(n)
    assert eq == pytest.approx(LEBESGUE[(d, p)][0], abs=1.0e-2)
    assert wb == pytest.approx(LEBESGUE[(d, p)][1], abs=1.0e-2)
    assert wb <= eq + 1.0e-12


def _notebook_nodes(cell, p):
    spec = importlib.util.spec_from_file_location("generate_reference_tables",
                                                  os.path.join(HERE, "fem_oracle", "generate_reference_tables.py"))
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)  # numpy only at import; sympy is needed only to build tables
    return gen.gll_reference_nodes(cell, p)


def _lebesgue_of_nodes(d, p, pts, n):
    s = S.SimplexSpace(d, p, "lagrange", "equispaced")
    C = np.linalg.inv(s._bernstein_tables(pts, 0)[0])
    lat = np.array([a[1:] for a in itertools.product(range(n + 1), repeat=d + 1) if sum(a) == n]) / float(n)
    return np.abs(s._bernstein_tables(lat, 0)[0] @ C).sum(axis=1).max()


def _same_set(a, b):
    return max(np.abs(b - r).max(axis=1).min() for r in a)


@pytest.mark.parametrize("d,p", [(2, 3), (2, 4), (2, 5), (2, 7), (3, 3), (3, 4)])
def test_node_sets_equal_the_notebook_where_the_two_constructions_agree(d, p):
    """Triangle (same blend parameter) and tetrahedron through p = 4: the node set of the oracle."""
    assert _same_set(S.warp_blend_barycentric(d, p)[:, 1:], _notebook_nodes(CELL[d], p)) < 1.0e-14


@pytest.mark.parametrize("p,published,notebook", [(5, 5.292, 5.582), (6, 6.952, 7.324), (7, 9.204, 9.655)])
def test_tetrahedron_interior_nodes_follow_the_published_construction(p, published, notebook):
    """For p >= 5 the notebook's nodes3d and the published construction differ in the interior of the
    tetrahedron; the published one has the lower Lebesgue constant, so it is the one used."""
    mine = S.warp_blend_barycentric(3, p)[:, 1:]
    theirs = _notebook_nodes("tetrahedron", p)
    assert _same_set(mine, theirs) > 1.0e-3
    l_mine, l_theirs = _lebesgue_of_nodes(3, p, mine, 40), _lebesgue_of_nodes(3, p, theirs, 40)
    assert l_mine == pytest.approx(published, abs=1.0e-2)
    assert l_theirs == pytest.approx(notebook, abs=1.0e-2)
    assert l_mine < l_theirs


def test_triangle_p6_blend_parameter():
    """The notebook has 0.9808 where the published table has 0.9800; the difference is immaterial."""
    pub = _lebesgue_of_nodes(2, 6, S.warp_blend_barycentric(2, 6)[:, 1:], 120)
    nb = _lebesgue_of_nodes(2, 6, S.warp_blend_barycentric(2, 6, alpha=0.9808)[:, 1:], 120)
    assert pub <= nb and nb - pub < 1.0e-3
