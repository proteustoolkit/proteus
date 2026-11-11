"""Polynomial MOMENTS of the equivalent polynomials (2D, P1 level set on a cut triangle).

test_equivalent_polynomials.py checks the zeroth moments: sum_q w_q H_q, (1 - H)_q and D_q against the exact area or interface length.  What the
finite-element integrals need are higher moments, and the properties below are what a user of H and D as weights relies on (the MCorr Jacobian was inexact
for lack of the third):

  * H_q reproduces int_{phi>0} N_i dx exactly (a degree-1 integrand) for equivalent polynomials of order nP once the quadrature is exact to degree nP + 1;
  * D_q reproduces int_Gamma N_i N_j dGamma (a degree-2 integrand) exactly once the quadrature is exact to degree nP + 2 (= 4 for nP = 2, 5 for nP = 3), and not one
    degree below: the rule is the degree of the equivalent polynomial plus the degree of the integrand (= 2 nP only when the two are equal);
  * D_q is the SURFACE Dirac (it integrates to the interface measure dGamma), so the derivative of the Heaviside with respect to the level set,
    delta(phi) = D / |grad phi|, equals it only where |grad phi| = 1.
"""
import numpy as np
import pytest
from proteus import equivalent_polynomials as eqp
from proteus.Quadrature import GaussTriangle

H_EL = 0.05
V = np.array([[0.0, 0.0], [H_EL, 0.0], [0.0, H_EL]])
JAC = np.array([V[1] - V[0], V[2] - V[0]]).T
DET = abs(np.linalg.det(JAC))
TOL = 1.0e-11

CUTS = {"one node alone": [-0.003, 0.012, 0.020], "two nodes below": [0.004, -0.015, -0.009], "near a vertex": [-0.0004, 0.02, 0.03]}
DISTANCE_FUNCTION = [V[i, 0] - 0.012 for i in range(3)]   # |grad phi| = 1


def _bary(x):
    l = np.linalg.solve(JAC, np.asarray(x) - V[0])
    return np.array([1.0 - l[0] - l[1], l[0], l[1]])


def _cut_points(phi):
    pts = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        if phi[a] * phi[b] < 0:
            t = phi[a] / (phi[a] - phi[b])
            pts.append(V[a] + t * (V[b] - V[a]))
    return pts


def _exact_h_moments(phi):
    poly = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        if phi[a] > 0:
            poly.append(V[a])
        if phi[a] * phi[b] < 0:
            t = phi[a] / (phi[a] - phi[b])
            poly.append(V[a] + t * (V[b] - V[a]))
    M = np.zeros(3)
    for k in range(1, len(poly) - 1):       # fan triangulation; a linear integrand is exact at the centroid
        tri = np.array([poly[0], poly[k], poly[k + 1]])
        M += 0.5 * abs(np.linalg.det(np.array([tri[1] - tri[0], tri[2] - tri[0]]))) * _bary(tri.mean(0))
    return M


def _exact_d_moments(phi, per_gradient):
    """int_Gamma N_i N_j dGamma (per_gradient=False) or int_Gamma N_i N_j / |grad phi| dGamma = int delta(phi) N_i N_j dx (True)."""
    p0, p1 = _cut_points(phi)
    L = np.linalg.norm(p1 - p0)
    grad = np.linalg.solve(np.array([V[1] - V[0], V[2] - V[0]]), np.array([phi[1] - phi[0], phi[2] - phi[0]]))
    scale = 1.0 / np.linalg.norm(grad) if per_gradient else 1.0
    x, w = np.polynomial.legendre.leggauss(3)    # degree 5 on the segment
    M = np.zeros((3, 3))
    for s, ws in zip(x, w):
        N = _bary(p0 + 0.5 * (s + 1) * (p1 - p0))
        M += 0.5 * ws * L * scale * np.outer(N, N)
    return M


def _weighted_moments(phi, nP, quad_order):
    quad = GaussTriangle(order=quad_order)
    gf = eqp.Simplex(nSpace=2, nP=nP, nQ=len(quad.points))
    gf.calculate(np.array(phi), np.array([[V[i, 0], V[i, 1], 0.0] for i in range(3)]), np.array(quad.points))
    MH, MD = np.zeros(3), np.zeros((3, 3))
    for k in range(len(quad.points)):
        gf.set_quad(k)
        N = _bary(V[0] + JAC @ np.array(quad.points[k][:2]))
        MH += quad.weights[k] * DET * gf.H * N
        MD += quad.weights[k] * DET * gf.D * np.outer(N, N)
    return MH, MD


def _rel(a, b):
    return np.abs(a - b).max() / np.abs(b).max()


@pytest.mark.parametrize("cut", sorted(CUTS))
@pytest.mark.parametrize("nP", [1, 2, 3])
def test_heaviside_reproduces_the_p1_moments_once_the_quadrature_is_exact_to_degree_nP_plus_one(cut, nP):
    phi = CUTS[cut]
    exact = _exact_h_moments(phi)
    for quad_order in range(nP + 1, 9):
        assert _rel(_weighted_moments(phi, nP, quad_order)[0], exact) < TOL, (nP, quad_order)


@pytest.mark.parametrize("cut", sorted(CUTS))
@pytest.mark.parametrize("nP", [2, 3])
def test_dirac_reproduces_the_quadratic_surface_moments_at_degree_nP_plus_two_and_not_below(cut, nP):
    phi = CUTS[cut]
    exact = _exact_d_moments(phi, per_gradient=False)
    for quad_order in range(nP + 2, 9):
        assert _rel(_weighted_moments(phi, nP, quad_order)[1], exact) < TOL, (nP, quad_order)
    # one degree below, the integral of a quadratic is not reproduced
    assert _rel(_weighted_moments(phi, nP, nP + 1)[1], exact) > 1e-3


@pytest.mark.parametrize("cut", sorted(CUTS))
def test_dirac_is_the_surface_measure_and_delta_phi_is_the_dirac_over_the_gradient(cut):
    phi = CUTS[cut]
    MD = _weighted_moments(phi, 2, 5)[1]
    assert _rel(MD, _exact_d_moments(phi, per_gradient=False)) < TOL            # D integrates dGamma
    assert _rel(MD, _exact_d_moments(phi, per_gradient=True)) > 1e-2            # so it is not delta(phi) where |grad phi| != 1
    grad = np.linalg.solve(np.array([V[1] - V[0], V[2] - V[0]]), np.array([phi[1] - phi[0], phi[2] - phi[0]]))
    assert _rel(MD / np.linalg.norm(grad), _exact_d_moments(phi, per_gradient=True)) < TOL


def test_dirac_and_delta_phi_coincide_for_a_distance_function():
    phi = DISTANCE_FUNCTION
    MD = _weighted_moments(phi, 2, 5)[1]
    assert _rel(MD, _exact_d_moments(phi, per_gradient=False)) < TOL
    assert _rel(MD, _exact_d_moments(phi, per_gradient=True)) < TOL
