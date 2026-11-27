"""C49 R4: proteus.MoveMeshMonitor.Coefficients, the parts the one integration test (test/MoveMeshMonitor) does not
run: the area function with its grading, scaling and level-set options, the reaction term, the element search, the
recovery interpolation and the mesh preparation (corner nodes, fixed material types).

The mesh is a real generated mesh (`partitionMesh` + `subdomainMesh`), and the answers are known in closed form: a
function of x evaluated through the documented formulas, a P1 interpolant of a constant, a point in an element.
"""
import numpy as np
import numpy.testing as npt
import pytest

from proteus import Comm
Comm.init()
from proteus import FemTools, MeshTools
from proteus import MoveMeshMonitor as mmm


def _func(x, t):
    return 0.1 + x[0] + 2. * x[1] + t


def _coeff(**kw):
    kw.setdefault('he_min', 0.05)
    kw.setdefault('he_max', 0.5)
    return mmm.Coefficients(_func, ME_MODEL=0, **kw)


X = np.array([0.2, 0.3, 0.])


# ---------------------------------------------------------------------------------------------------------------------
# the area function
# ---------------------------------------------------------------------------------------------------------------------

def test_area_function_without_grading_is_the_user_function_clipped_to_the_bounds():
    c = _coeff(grading_type=0)
    f = _func(X, 0.)
    assert c.evaluateFunAtX(X) == pytest.approx(c.he_max)          # 0.9 > he_max = 0.5: clipped
    c.he_max = 5.
    assert c.evaluateFunAtX(X) == pytest.approx(f)
    c.t = 0.25
    assert c.evaluateFunAtX(X) == pytest.approx(f + 0.25)          # the function sees the time of the model


def test_area_function_epsFact_density_shifts_and_floors_the_distance():
    c = _coeff(epsFact_density=2., he_max=5.)
    assert c.evaluateFunAtX(X) == pytest.approx(_func(X, 0.) - 2. * 0.05)
    # a point nearer than epsFact * he_min to the feature has zero distance (the finest cell size)
    c = _coeff(epsFact_density=100., he_max=5.)
    assert c.evaluateFunAtX(X) == 0.


def test_area_function_hyperbolic_and_logarithmic_grading():
    hmin, g = 0.05, 1.2
    c = _coeff(grading=g, grading_type=1, he_max=5.)
    d = _func(X, 0.)
    assert c.evaluateFunAtX(X) == pytest.approx(hmin * g ** (d / hmin))
    c = _coeff(grading=g, grading_type=2, he_max=5.)
    assert c.evaluateFunAtX(X) == pytest.approx(((g - 1.) * d + hmin) / g)
    # at distance zero both give he_min-ish: hyperbolic exactly he_min, log he_min / grading
    zero = np.array([-0.1, 0., 0.])
    assert _func(zero, 0.) == 0.
    assert _coeff(grading=g, grading_type=1).evaluateFunAtX(zero) == pytest.approx(hmin)
    assert _coeff(grading=g, grading_type=2).evaluateFunAtX(zero) == pytest.approx(hmin / g)
    # grading never exceeds he_max
    assert _coeff(grading=g, grading_type=1, he_max=0.3).evaluateFunAtX(np.array([5., 5., 0.])) == 0.3


def test_area_function_takes_the_smaller_of_level_set_and_function():
    c = _coeff(he_max=5.)
    assert c.evaluateFunAtX(X, ls_phi=-0.2) == pytest.approx(0.2)
    assert c.evaluateFunAtX(X, ls_phi=3.) == pytest.approx(_func(X, 0.))


@pytest.mark.parametrize('nd, power', [(2, 2), (3, 3)])
def test_scale_with_nd_raises_the_size_to_the_dimension(nd, power):
    c = _coeff(scale_with_nd=True, nd=nd, he_max=5.)
    assert c.evaluateFunAtX(X) == pytest.approx(_func(X, 0.) ** power)
    c = _coeff(scale_with_nd=False, nd=nd, he_max=5.)
    assert c.evaluateFunAtX(X) == pytest.approx(_func(X, 0.))


def test_reaction_term_is_reciprocal_target_minus_reciprocal_current_area():
    c = _coeff(he_max=5.)
    c.C = 3.
    c.areas = np.array([0.5, 0.25])
    f = c.evaluateFunAtX(X)
    assert c.uOfX(X) == pytest.approx(1. / (f * 3.) - 1.)          # no element: unit area
    assert c.uOfX(X, eN=1) == pytest.approx(1. / (f * 3.) - 4.)
    # the Poisson coefficients are set up with that reaction term and the identity diffusion
    assert c.uOfX is not None and c.nd == 2


def test_epsTimeStep_must_be_in_the_unit_interval():
    with pytest.raises(AssertionError):
        _coeff(epsTimeStep=0.)
    with pytest.raises(AssertionError):
        _coeff(epsTimeStep=1.5)
    _coeff(epsTimeStep=1.)
    _coeff(epsTimeStep=0.1)
    # useLS False forgets the level-set model
    assert _coeff(LS_MODEL=3, useLS=False).LS_MODEL is None
    assert _coeff(LS_MODEL=3, useLS=True).LS_MODEL == 3


def test_area_function_at_nodes_and_quadrature_points_uses_the_level_set_when_there_is_one():
    c = _coeff(grading=1.2, grading_type=1, epsFact_density=1., he_max=0.4)

    class Mesh:
        nodeArray = np.array([[0., 0., 0.], [0.3, 0.1, 0.], [1., 1., 0.]])
        elementNodesArray = np.zeros((2, 3), dtype=int)

    class Model:
        q = {'x': np.array([[[0.1, 0.1, 0.], [0.5, 0.5, 0.]], [[0.9, 0.2, 0.], [0.2, 0.9, 0.]]])}

    c.mesh, c.model = Mesh(), Model()
    c.t = 0.1
    c.uOfXTatNodes = np.zeros(3)
    c.uOfXTatQuadrature = np.zeros((2, 2))
    c.LS_MODEL = None

    def expect(d):
        d = d - 1. * c.he_min
        d = max(d, 0.)
        return min(c.he_max, c.he_min * c.grading ** (d / c.he_min))

    c.evaluateFunAtNodes()
    npt.assert_allclose(c.uOfXTatNodes, [expect(_func(x, 0.1)) for x in Mesh.nodeArray])
    c.evaluateFunAtQuadraturePoints()
    npt.assert_allclose(c.uOfXTatQuadrature, [[expect(_func(x, 0.1)) for x in row] for row in Model.q['x']])
    # with a level-set model the distance is min(|phi|, func)
    c.LS_MODEL = 1
    c.u_phi = np.array([0.05, 1., 0.3])
    c.q_phi = np.array([[0.02, 5.], [0.3, 0.01]])
    c.evaluateFunAtNodes()
    npt.assert_allclose(c.uOfXTatNodes, [expect(min(abs(p), _func(x, 0.1))) for p, x in zip(c.u_phi, Mesh.nodeArray)])
    c.evaluateFunAtQuadraturePoints()
    exp = [[expect(min(abs(c.q_phi[e, k]), _func(Model.q['x'][e, k], 0.1))) for k in range(2)] for e in range(2)]
    npt.assert_allclose(c.uOfXTatQuadrature, exp)


def test_calculate_area_is_the_weighted_sum_of_the_jacobian_determinants():
    w = np.array([0.5, 0.25, 0.25])
    detJ = np.array([2., 4., 8.])
    assert mmm.calculate_area(np.zeros(3), detJ, w) == pytest.approx(0.5 * 2. + 0.25 * 4. + 0.25 * 8.)


@pytest.mark.xfail(strict=True, reason="calculate_areas fills `areas` per element but returns the last element's "
                                       "`area` (a float) instead of the array (found by C49 R4; C0)")
def test_calculate_areas_returns_one_area_per_element():
    w = np.array([0.5, 0.5])
    detJ = np.array([[2., 2.], [4., 4.]])
    npt.assert_allclose(mmm.calculate_areas(np.zeros(2), detJ, w), [2., 4.])


# ---------------------------------------------------------------------------------------------------------------------
# on a real mesh
# ---------------------------------------------------------------------------------------------------------------------

def _mesh(nx=5, ny=4):
    mesh = MeshTools.TriangularMesh()
    mesh.generateTriangularMeshFromRectangularGrid(nx, ny, 1., 1.)
    mesh.partitionMesh(nLayersOfOverlap=0, parallelPartitioningType=MeshTools.MeshParallelPartitioningTypes.element)
    return mesh.subdomainMesh


def test_initializeMesh_sets_up_the_normals_and_fixes_the_corner_nodes():
    mesh = _mesh()
    c = _coeff()
    c.initializeMesh(mesh)
    assert c.areas.shape == (mesh.nElements_global,)
    assert mesh.elementBoundaryNormalsArray.shape == (len(mesh.elementBoundariesArray), 3, 3)
    # the four corners of the unit square are fixed, nothing else
    fixed = set(np.flatnonzero(mesh.fixedNodesBoolArray).tolist())
    corners = [i for i, p in enumerate(mesh.nodeArray) if np.all(np.isclose(np.abs(p[:2] - 0.5), 0.5))]
    assert len(corners) == 4 and set(corners) <= fixed
    # the corner search also fixes a few more boundary nodes (where the boundary flag changes), never an interior one
    for node in fixed:
        assert np.any(np.isclose(np.abs(mesh.nodeArray[node, :2] - 0.5), 0.5))
    assert len(fixed) < len(mesh.nodeArray) / 2
    assert fixed == set(np.unique(mesh.cornerNodes).tolist())
    # unit normals of the three sides of every triangle
    n = np.asarray(mesh.elementBoundaryNormalsArray)
    norms = np.linalg.norm(n[:, :, :2], axis=2)
    npt.assert_allclose(norms[norms > 0], 1., atol=1e-12)


def test_initializeMesh_fixes_nodes_by_material_type():
    mesh = _mesh()
    types = np.unique(mesh.nodeMaterialTypes)
    assert len(types) > 1
    fixedTypes = np.zeros(types.max() + 1, dtype=int)
    fixedTypes[types[-1]] = 1
    c = _coeff(fixedNodeMaterialTypes=fixedTypes)
    c.initializeMesh(mesh)
    for node, mt in enumerate(mesh.nodeMaterialTypes):
        if mt == types[-1]:
            assert mesh.fixedNodesBoolArray[node] == 1
    # fixing an element material type fixes every node of those elements
    mesh = _mesh()
    etypes = np.zeros(max(2, int(mesh.elementMaterialTypes.max()) + 1), dtype=int)
    etypes[int(mesh.elementMaterialTypes[0])] = 1
    c = _coeff(fixedElementMaterialTypes=etypes)
    c.initializeMesh(mesh)
    assert np.all(mesh.fixedNodesBoolArray[mesh.elementNodesArray[mesh.elementMaterialTypes == np.flatnonzero(etypes)[0]]] == 1)


def _search_setup():
    mesh = _mesh(6, 6)
    c = _coeff(he_max=5.)
    c.initializeMesh(mesh)
    space = FemTools.C0_AffineLinearOnSimplexWithNodalBasis(mesh, 2)

    class U:
        femSpace = space
        dof = np.zeros(space.dim)

    class Model:
        u = {0: U()}

    c.model = Model()
    return mesh, c, space


def _inside(mesh, eN, x):
    nodes = mesh.nodeArray[mesh.elementNodesArray[eN]][:, :2]
    T = np.array([nodes[1] - nodes[0], nodes[2] - nodes[0]]).T
    b = np.linalg.solve(T, np.asarray(x[:2]) - nodes[0])
    return b[0] >= -1e-10 and b[1] >= -1e-10 and b.sum() <= 1. + 1e-10


@pytest.mark.xfail(strict=True, raises=TypeError,
                   reason="Coefficients.dist_search compares element = None with 0 (`if element < 0`) whenever the "
                          "guessed element is not given or does not contain the point, i.e. on every normal call "
                          "(found by C49 R4; nothing in the tree calls it; kernel territory: C0)")
@pytest.mark.parametrize('x', [[0.51, 0.49, 0.], [0.05, 0.93, 0.], [0.34, 0.66, 0.]])
def test_dist_search_finds_the_element_that_contains_the_point_from_any_node(x):
    mesh, c, space = _search_setup()
    x = np.array(x)
    for start in (0, mesh.nNodes_global // 2, mesh.nNodes_global - 1):
        element, node, xi = c.dist_search(x, start)
        assert element >= 0 and _inside(mesh, element, x)


def test_recovery_interpolation_reproduces_constants_and_linear_fields():
    mesh, c, space = _search_setup()
    c.grads = np.zeros((space.dim, 2))
    c.grads[:] = [0.7, -0.3]
    c.areas_nodes = np.full(space.dim, 0.04)
    c.u_phi = mesh.nodeArray[:, 0] + 2. * mesh.nodeArray[:, 1]          # a linear field
    for eN in (0, 7, mesh.nElements_global - 1):
        xi = np.array([0.2, 0.3, 0.])
        npt.assert_allclose(c.getGradientValue(eN, xi), [0.7, -0.3], atol=1e-13)
        npt.assert_allclose(c.getAreaValue(eN, xi), 0.04, atol=1e-13)
        # the level set is P1-interpolated: exact for a linear field at the mapped point
        nodes = mesh.nodeArray[mesh.elementNodesArray[eN]]
        x = (1. - xi[0] - xi[1]) * nodes[0] + xi[0] * nodes[1] + xi[1] * nodes[2]
        npt.assert_allclose(c.getLevelSetValue(eN, xi), x[0] + 2. * x[1], atol=1e-12)
