"""The space interface (C0 round 12): continuity, family, order, cell, `isSpace`, `dofCoordinates`, `sameSpaceAs`.

Callers (Transport's viewers, the mprans models, RDLS, RANS3PF, ...) used to test the concrete class of a space
(`isinstance(space, C0_AffineQuadraticOnSimplexWithNodalBasis)`), which rejects or mis-dispatches every space that
is not one of those classes (any-order `C0_AffinePolynomialOnSimplex`, the H(div) spaces). They now ask the
interface. The first test is the guarantee that nothing changed for the existing spaces: for every class, each old
`isinstance` condition and its interface replacement give the same answer.
"""
import numpy as np
import pytest

from proteus import Comm
Comm.init()
from proteus import FemTools as F, MeshTools

OLD_AND_NEW = {
    # old test (class) : interface replacement used in Transport.py (the viewers)
    "p1": (lambda s: isinstance(s, F.C0_AffineLinearOnSimplexWithNodalBasis),
           lambda s: s.isSpace('C0', ('lagrange', 'bernstein'), 1, 'simplex')),
    "p2": (lambda s: isinstance(s, F.C0_AffineQuadraticOnSimplexWithNodalBasis),
           lambda s: s.isSpace('C0', ('lagrange', 'bernstein'), 2, 'simplex')),
    "nc": (lambda s: isinstance(s, F.NC_AffineLinearOnSimplexWithNodalBasis),
           lambda s: s.isSpace('NC')),
    "dg1": (lambda s: isinstance(s, F.DG_AffineLinearOnSimplexWithNodalBasis),
            lambda s: s.isSpace('DG', 'lagrange', 1)),
    "dg2": (lambda s: isinstance(s, F.DG_AffineQuadraticOnSimplexWithNodalBasis),
            lambda s: s.isSpace('DG', 'lagrange', 2)),
    "dgk": (lambda s: isinstance(s, F.DG_AffinePolynomialsOnSimplexWithMonomialBasis),
            lambda s: s.isSpace('DG', 'monomial')),
    "dg0": (lambda s: isinstance(s, (F.DG_Constants, F.DG_AffineP0_OnSimplexWithMonomialBasis)),
            lambda s: s.isSpace('DG', 'constant') or s.isSpace('DG', 'monomial', 0)),
    "bubble": (lambda s: isinstance(s, F.C0_AffineP1BubbleOnSimplexWithNodalBasis),
               lambda s: s.isSpace('C0', 'p1-bubble')),
    "p1p0bubble": (lambda s: isinstance(s, F.C0_AffineP1P0BubbleOnSimplexWithNodalBasis),
                   lambda s: s.isSpace('C0', 'p1p0-bubble')),
    "bernstein-p2": (lambda s: isinstance(s, F.C0_AffineBernsteinOnSimplex),
                     lambda s: s.isSpace('C0', 'bernstein', 2, 'simplex') and type(s).__name__ == "C0_AffineBernsteinOnSimplex"),
}

CONSTRUCT = {
    "P1": lambda m, nd: F.C0_AffineLinearOnSimplexWithNodalBasis(m, nd),
    "P2": lambda m, nd: F.C0_AffineQuadraticOnSimplexWithNodalBasis(m, nd),
    "P2-bernstein": lambda m, nd: F.C0_AffineBernsteinOnSimplex(m, nd),
    "DG-P1": lambda m, nd: F.DG_AffineLinearOnSimplexWithNodalBasis(m, nd),
    "DG-P2": lambda m, nd: F.DG_AffineQuadraticOnSimplexWithNodalBasis(m, nd),
    "DG-P0-mono": lambda m, nd: F.DG_AffineP0_OnSimplexWithMonomialBasis(m, nd),
    "DG-P1-mono": lambda m, nd: F.DG_AffineP1_OnSimplexWithMonomialBasis(m, nd),
    "DG-P3-mono": lambda m, nd: F.DG_AffineP3_OnSimplexWithMonomialBasis(m, nd),
    "DG-const": lambda m, nd: F.DG_Constants(m, nd),
    "NC": lambda m, nd: F.NC_AffineLinearOnSimplexWithNodalBasis(m, nd),
    "P1-bubble": lambda m, nd: F.C0_AffineP1BubbleOnSimplexWithNodalBasis(m, nd),
    "P1P0-bubble": lambda m, nd: F.C0_AffineP1P0BubbleOnSimplexWithNodalBasis(m, nd),
}
EXISTING = list(CONSTRUCT)
# the classes whose old test is not an old isinstance test of an old class: the new any-order spaces
NEW = {
    "poly1": lambda m, nd: F.C0_AffinePolynomialOnSimplex(m, nd, 1),
    "poly2": lambda m, nd: F.C0_AffinePolynomialOnSimplex(m, nd, 2),
    "poly3": lambda m, nd: F.C0_AffinePolynomialOnSimplex(m, nd, 3),
    "poly3-bernstein": lambda m, nd: F.C0_AffinePolynomialOnSimplex(m, nd, 3, 'bernstein'),
    "rt0": lambda m, nd: F.Hdiv_AffineOnSimplex(m, nd, 'rt0'),
    "bdm1": lambda m, nd: F.Hdiv_AffineOnSimplex(m, nd, 'bdm1'),
}


@pytest.fixture(scope="module")
def mesh():
    m = MeshTools.TriangularMesh()
    m.generateTriangularMeshFromRectangularGrid(4, 4, 1.0, 1.0)
    m.partitionMesh(nLayersOfOverlap=0, parallelPartitioningType=MeshTools.MeshParallelPartitioningTypes.element)
    return m.subdomainMesh


@pytest.fixture(scope="module")
def spaces(mesh):
    return {k: f(mesh, 2) for k, f in {**CONSTRUCT, **NEW}.items()}


@pytest.mark.parametrize("name", EXISTING)
@pytest.mark.parametrize("test", list(OLD_AND_NEW))
def test_every_old_class_test_and_its_interface_replacement_agree_on_the_existing_spaces(spaces, name, test):
    old, new = OLD_AND_NEW[test]
    assert bool(old(spaces[name])) == bool(new(spaces[name])), (name, test)


def test_the_interface_of_each_space(spaces):
    want = {"P1": ("C0", "lagrange", 1), "P2": ("C0", "lagrange", 2), "P2-bernstein": ("C0", "bernstein", 2),
            "DG-P1": ("DG", "lagrange", 1), "DG-P2": ("DG", "lagrange", 2), "DG-P0-mono": ("DG", "monomial", 0),
            "DG-P1-mono": ("DG", "monomial", 1), "DG-P3-mono": ("DG", "monomial", 3), "DG-const": ("DG", "constant", 0),
            "NC": ("NC", "crouzeix-raviart", 1), "P1-bubble": ("C0", "p1-bubble", 1),
            "P1P0-bubble": ("C0", "p1p0-bubble", 1), "poly1": ("C0", "lagrange", 1), "poly2": ("C0", "lagrange", 2),
            "poly3": ("C0", "lagrange", 3), "poly3-bernstein": ("C0", "bernstein", 3), "rt0": ("Hdiv", "rt0", 0),
            "bdm1": ("Hdiv", "bdm1", 1)}
    for name, (c, f, o) in want.items():
        s = spaces[name]
        assert (s.continuity, s.family, s.order) == (c, f, o), name
        assert s.cell == "simplex"


def test_a_new_space_is_accepted_where_the_old_class_tests_refused_it(spaces):
    s = spaces["poly3"]
    assert not isinstance(s, F.C0_AffineQuadraticOnSimplexWithNodalBasis)
    assert s.isSpace("C0", ("lagrange", "bernstein")) and s.order > 1          # what `Hess` now asks
    # the same numbering as the classic P2: it is the P2 of the interface although it is another class
    assert spaces["poly2"].isSpace("C0", ("lagrange", "bernstein"), 2, "simplex")
    assert spaces["poly2"].sameSpaceAs(spaces["poly2"]) and not spaces["poly2"].sameSpaceAs(spaces["poly3"])
    assert not spaces["poly3"].sameSpaceAs(spaces["poly3-bernstein"])


def test_same_space_distinguishes_what_class_names_distinguished(spaces):
    names = EXISTING
    for a in names:
        for b in names:
            assert spaces[a].sameSpaceAs(spaces[b]) == (type(spaces[a]).__name__ == type(spaces[b]).__name__), (a, b)


def test_cube_classes_say_they_are_cubes():
    for cls in (F.C0_LinearOnCubeWithNodalBasis, F.C0_AffineLinearOnCubeWithNodalBasis, F.C0_LagrangeOnCubeWithNodalBasis,
                F.C0_AffineLagrangeOnCubeWithNodalBasis, F.C0_BernsteinOnCube, F.C0_AffineBernsteinOnCube,
                F.C0_AffinePolynomialOnCube):
        assert cls.cell == "cube", cls
    assert F.C0_AffineLinearOnSimplexWithNodalBasis.cell == "simplex"


def test_dof_coordinates(spaces, mesh):
    assert np.array_equal(spaces["P1"].dofCoordinates(), mesh.nodeArray)
    for name in ("P2", "poly2", "poly3"):
        x = spaces[name].dofCoordinates()
        assert x is spaces[name].dofMap.lagrangeNodesArray and x.shape[0] == spaces[name].dofMap.nDOF
    # P1 and the order-1 any-order space number the vertices alike
    assert np.allclose(spaces["poly1"].dofCoordinates()[:mesh.nNodes_global], mesh.nodeArray)
