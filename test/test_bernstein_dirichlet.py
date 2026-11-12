"""
Strong Dirichlet data and initial data for the Bernstein spaces (C8 round 2, decision Q4).

A Bernstein dof is a coefficient, not a point value. `C0_AffinePolynomialOnSimplex(family='bernstein')`
converts by default: the boundary dofs that `DOFBoundaryConditions` hands out are the Bernstein coefficients
of the degree-p interpolant of the data on the boundary entity, so degree-p data is reproduced exactly and
the boundary error is of order p+1. `bernsteinOperator=True` keeps control-point values (bounds
preserving, second order). Lagrange spaces are nodal and exact as before.

Today's P2 `C0_AffineBernsteinOnSimplex` uses control-point (edge midpoint) values; that is measured and
pinned here as it is, not changed (it would move the pins of the cases that use it).
"""
import numpy as np
import pytest

from proteus import Comm
Comm.init()
from proteus import FemTools, MeshTools


def _mesh(nd):
    if nd == 2:
        mesh = MeshTools.TriangularMesh()
        mesh.generateTriangularMeshFromRectangularGrid(5, 4, 1.0, 1.0)
    else:
        mesh = MeshTools.TetrahedralMesh()
        mesh.generateTetrahedralMeshFromRectangularGrid(3, 3, 3, 1.0, 1.0, 1.0)
    mesh.partitionMesh(nLayersOfOverlap=0, parallelPartitioningType=MeshTools.MeshParallelPartitioningTypes.element)
    return mesh.subdomainMesh


def _poly(x, p, nd):
    c = np.array([1.0, -0.7, 0.4])[:nd]
    return (0.3 + x[..., :nd] @ c) ** p + 0.5 * x[..., 0] ** p


def _dirichlet_values(space, p, nd):
    """Boundary dof numbers and the values the Dirichlet conditions give them, with data on the whole boundary."""
    def conditions(x, flag):
        if flag != 0:  # every exterior face of the generated meshes has a positive flag
            return lambda x, t: _poly(np.asarray(x), p, nd)
        return None
    bc = FemTools.DOFBoundaryConditions(space, conditions)
    dofs = np.array(sorted(bc.DOFBoundaryConditionsDict))
    values = np.array([bc.DOFBoundaryConditionsDict[n](bc.DOFBoundaryPointDict[n], 0.0) for n in dofs])
    return dofs, values


def _on_boundary(points, nd):
    return np.any(np.isclose(points[:, :nd], 0.0) | np.isclose(points[:, :nd], 1.0), axis=1)


CASES = [(2, p) for p in (1, 2, 3, 4, 5)] + [(3, p) for p in (1, 2, 3, 4)]


@pytest.mark.parametrize("nd,p", CASES)
@pytest.mark.parametrize("family", ["lagrange", "bernstein"])
def test_degree_p_dirichlet_data_is_reproduced_exactly(nd, p, family):
    mesh = _mesh(nd)
    space = FemTools.C0_AffinePolynomialOnSimplex(mesh, nd, p, family)
    u = FemTools.FiniteElementFunction(space, name="u")
    u.projectFromInterpolationConditions(_poly(space.interpolationPoints, p, nd))  # the exact interpolant
    dofs, values = _dirichlet_values(space, p, nd)
    # the Dirichlet dofs are exactly the dofs whose point (node or control point) is on the boundary
    assert np.array_equal(dofs, np.where(_on_boundary(space.dofMap.lagrangeNodesArray, nd))[0])
    assert np.abs(values - u.dof[dofs]).max() < 1.0e-12, np.abs(values - u.dof[dofs]).max()
    # with those boundary dofs (and the interior ones of the interpolant) u_h is the polynomial on the boundary
    u.dof[dofs] = values
    xi = np.random.RandomState(1).dirichlet(np.ones(nd + 1), size=5)[:, 1:]
    en = np.asarray(mesh.elementNodesArray)
    bary = np.column_stack([1.0 - xi.sum(axis=1), xi])
    xq = np.einsum("qv,evc->eqc", bary, np.asarray(mesh.nodeArray)[en])
    uh = np.einsum("qi,ei->eq", space.getBasisValuesRef(xi), u.dof[space.dofMap.l2g])
    assert np.abs(uh - _poly(xq, p, nd)).max() < 1.0e-11


@pytest.mark.parametrize("nd,p", [(2, 2), (2, 4), (3, 3)])
def test_bernstein_operator_is_opt_in_and_uses_control_point_values(nd, p):
    mesh = _mesh(nd)
    space = FemTools.C0_AffinePolynomialOnSimplex(mesh, nd, p, "bernstein", bernsteinOperator=True)
    dofs, values = _dirichlet_values(space, p, nd)
    control_points = space.dofMap.lagrangeNodesArray[dofs]
    assert np.abs(values - _poly(control_points, p, nd)).max() < 1.0e-13
    exact = FemTools.C0_AffinePolynomialOnSimplex(mesh, nd, p, "bernstein")
    u = FemTools.FiniteElementFunction(exact, name="u")
    u.projectFromInterpolationConditions(_poly(exact.interpolationPoints, p, nd))
    assert np.abs(values - u.dof[dofs]).max() > 1.0e-3  # not the coefficients of the data: second order


@pytest.mark.parametrize("nd", [2, 3])
def test_todays_p2_bernstein_space_uses_control_point_values(nd):
    """Documented, unchanged behaviour: the edge dofs of C0_AffineBernsteinOnSimplex are set to g at the
    edge midpoint, where the exact coefficient is 2 g(m) - (g(a) + g(b))/2: a boundary error h^2 g''/8."""
    mesh = _mesh(nd)
    old = FemTools.C0_AffineBernsteinOnSimplex(mesh, nd)
    new = FemTools.C0_AffinePolynomialOnSimplex(mesh, nd, 2, "bernstein")
    assert np.array_equal(np.asarray(old.dofMap.l2g), new.dofMap.l2g)  # same dofs, so values are comparable
    dofs_old, values_old = _dirichlet_values(old, 2, nd)
    dofs_new, values_new = _dirichlet_values(new, 2, nd)
    assert np.array_equal(dofs_old, dofs_new)
    midpoints = old.dofMap.lagrangeNodesArray[dofs_old]
    assert np.abs(values_old - _poly(midpoints, 2, nd)).max() < 1.0e-13
    difference = np.abs(values_old - values_new).max()
    assert 1.0e-3 < difference < 1.0, difference
    print("P2 Bernstein, nd=%d: today's Dirichlet edge dofs differ from the exact coefficients by %.3e" % (nd, difference))
