"""
The reference trace gradients of a P2 space must be evaluated on the face they belong to.

`ParametricFiniteElementSpace.getBasisGradientValuesTraceRef` feeds `u_grad_trial_trace_ref`
/ `u_grad_test_trace_ref` in every kernel. Its value counterpart maps each boundary
quadrature point onto face ebN with `boundaryMapList[ebN]` before evaluating; the gradient
routine did not, so for p >= 2 every face used the points of one face (invisible for P1,
whose gradients are constant). The check is independent of the implementation: the nodal
interpolant of u(xi) = sum_I xi_I^2 is exact in P2, so the reconstructed value and gradient
on each face must equal u and grad u at the mapped point.
"""
import numpy as np
import pytest
from pytest import approx

from proteus import FemTools, Quadrature


class _ReferenceOnlySpace(object):
    """The two trace routines under test use only `referenceFiniteElement`."""

    def __init__(self, nd):
        lfs = FemTools.QuadraticOnSimplexWithNodalBasis(nd)
        ic = FemTools.QuadraticLagrangeNodalInterpolationConditions(lfs.referenceElement)
        self.referenceFiniteElement = FemTools.ReferenceFiniteElement(lfs, ic)

    getBasisValuesTraceRef = FemTools.ParametricFiniteElementSpace.getBasisValuesTraceRef
    getBasisGradientValuesTraceRef = FemTools.ParametricFiniteElementSpace.getBasisGradientValuesTraceRef


@pytest.mark.parametrize("nd", [2, 3])
def test_p2_trace_gradients_are_taken_on_their_own_face(nd):
    space = _ReferenceOnlySpace(nd)
    ref = space.referenceFiniteElement
    nodes = ref.interpolationConditions.quadraturePointArray[:, :nd]
    u = np.array([np.sum(x ** 2) for x in nodes])  # nodal interpolant of sum_I xi_I^2, exact in P2
    quad = Quadrature.GaussEdge(3) if nd == 2 else Quadrature.GaussTriangle(3)
    xi = np.array(quad.points)
    psi = space.getBasisValuesTraceRef(xi)
    grad_psi = space.getBasisGradientValuesTraceRef(xi)
    n_faces = ref.referenceElement.nElementBoundaries
    assert psi.shape == (n_faces, len(xi), ref.localFunctionSpace.dim)
    assert grad_psi.shape == (n_faces, len(xi), ref.localFunctionSpace.dim, nd)
    worst = 0.0
    for ebN in range(n_faces):
        for k in range(len(xi)):
            x_hat = ref.referenceElement.boundaryMapList[ebN](xi[k])
            assert np.dot(u, psi[ebN, k, :]) == approx(np.sum(x_hat ** 2), abs=1e-12)
            grad = np.dot(u, grad_psi[ebN, k, :, :])
            worst = max(worst, np.max(np.abs(grad - 2.0 * x_hat)))
    assert worst == approx(0.0, abs=1e-12), \
        "P2 trace gradient off by up to {0:g} on some face (nd={1})".format(worst, nd)
