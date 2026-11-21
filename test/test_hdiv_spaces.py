"""
The H(div) local spaces RT0 and BDM1 on simplices (C8 round 5): `proteus.HdivSpaces`.

* Against the pinned oracle `test/fem_oracle/hdiv_tables.npz` (an independent path: a monomial generating
  set and an exact rational inverse of the dof matrix there, closed forms here): values, divergences, the
  normal flux on every face, the area-weighted normals, on the triangle and the tetrahedron.
* Without the oracle, in 1, 2 and 3 dimensions: the basis is dual to the degrees of freedom; a basis
  function has no normal component on the faces it does not belong to; the BDM1 functions of a face add up to
  its RT0 function; RT0 reproduces a + b x and BDM1 every linear vector field from their dofs.
* The Piola map: on a random affine element, of either orientation, the mapped basis is dual to the physical
  dofs (so the outward flux through each face is preserved) and its divergence is the reference one over
  |det J|.
"""
import os

import numpy as np
import pytest

from proteus import HdivSpaces as H

HERE = os.path.dirname(os.path.abspath(__file__))
_data = np.load(os.path.join(HERE, "fem_oracle", "hdiv_tables.npz"))
ALL = {k: _data[k] for k in _data.files}
CELL = {2: "triangle", 3: "tetrahedron"}
CASES = [(d, family) for d in (1, 2, 3) for family in ("rt0", "bdm1")]
IDS = ["%dd-%s" % c for c in CASES]


def T(d, family, item):
    return ALL["%s/%s/%s" % (family, CELL[d], item)]


def _match(s, d, family):
    """perm[i] = index in the space of the oracle's function i, matched by (face, vertex)."""
    mine = {(int(f), int(a)): i for i, (f, a) in enumerate(zip(s.dof_face, s.dof_vertex))}
    perm = np.array([mine[(int(f), int(a))] for f, a in zip(T(d, family, "dof_face"), T(d, family, "dof_vertex"))])
    assert sorted(perm) == list(range(s.dim))
    return perm


@pytest.mark.parametrize("d,family", [c for c in CASES if c[0] > 1], ids=[i for c, i in zip(CASES, IDS) if c[0] > 1])
def test_against_the_oracle(d, family):
    s = H.HdivSpace(d, family)
    perm = _match(s, d, family)
    assert np.abs(H.reference_area_normals(d) - T(d, family, "area_normal")).max() < 1.0e-15
    t = s.tabulate(T(d, family, "x"))
    assert np.abs(t.values[:, perm] - T(d, family, "val")).max() < 1.0e-13
    assert np.abs(t.grads[:, perm] - T(d, family, "div")).max() < 1.0e-13
    face_x, face_flux = T(d, family, "face_x"), T(d, family, "face_flux")
    N = H.reference_area_normals(d)
    for f in range(d + 1):
        flux = np.einsum("d,qnd->qn", N[f], s.tabulate(face_x[f]).values)
        assert np.abs(flux[:, perm] - face_flux[f]).max() < 1.0e-13, f


@pytest.mark.parametrize("d,family", CASES, ids=IDS)
def test_layout_duality_and_normal_traces(d, family):
    s = H.HdivSpace(d, family)
    assert s.dim == (d + 1 if family == "rt0" else d * (d + 1))
    assert [len(x) for x in s.face_dofs] == [1 if family == "rt0" else d] * (d + 1)
    assert sorted(i for dofs in s.face_dofs for i in dofs) == list(range(s.dim))
    # the dof points lie on their faces: lambda_f = 0
    lam = np.column_stack([1.0 - s.dof_points.sum(axis=1), s.dof_points])
    assert np.abs(lam[np.arange(s.dim), s.dof_face]).max() < 1.0e-15
    # dual basis
    A = s.dofs_from_values(np.array([s.tabulate(s.dof_points).values[i, :, :] for i in range(s.dim)]).transpose(1, 0, 2))
    # A[j, i] = sigma_i(phi_j)
    assert np.abs(A - np.eye(s.dim)).max() < 1.0e-14
    # no normal component on the other faces; on its own face the flux density is 1 (rt0) or lambda_a (bdm1)
    N = H.reference_area_normals(d)
    rs = np.random.RandomState(11)
    for f in range(d + 1):
        w = rs.dirichlet(np.ones(d), size=6)
        x = w @ s.vertices[list(s.faces[f])]
        flux = np.einsum("d,qnd->qn", N[f], s.tabulate(x).values)
        for i in range(s.dim):
            if s.dof_face[i] != f:
                assert np.abs(flux[:, i]).max() < 1.0e-14
            elif family == "rt0":
                assert np.abs(flux[:, i] - 1.0).max() < 1.0e-14
            else:
                a = list(s.faces[f]).index(s.dof_vertex[i])
                assert np.abs(flux[:, i] - w[:, a]).max() < 1.0e-14
    # the divergence theorem on the reference cell: |K| div phi_i = sum over faces of the flux = 1 (rt0), 1/d (bdm1)
    assert np.abs(s.divergence / np.prod(np.arange(1, d + 1)) - (1.0 if family == "rt0" else 1.0 / d)).max() < 1.0e-14


@pytest.mark.parametrize("d", [1, 2, 3])
def test_bdm1_contains_rt0_and_both_reproduce_their_spaces(d):
    rt, bdm = H.HdivSpace(d, "rt0"), H.HdivSpace(d, "bdm1")
    rs = np.random.RandomState(12)
    x = rs.dirichlet(np.ones(d + 1), size=7)[:, 1:]
    vr, vb = rt.tabulate(x).values, bdm.tabulate(x).values
    for f in range(d + 1):
        assert np.abs(vb[:, bdm.face_dofs[f]].sum(axis=1) - vr[:, rt.face_dofs[f][0]]).max() < 1.0e-14
    a, b, B = rs.uniform(-1, 1, d), rs.uniform(-1, 1), rs.uniform(-1, 1, (d, d))
    raviart = lambda y: a + b * y
    linear = lambda y: a + y @ B.T
    for s, field, div in ((rt, raviart, b * d), (bdm, linear, np.trace(B)), (bdm, raviart, b * d)):
        dofs = s.dofs_from_values(field(s.dof_points))
        t = s.tabulate(x)
        assert np.abs(np.einsum("i,qid->qd", dofs, t.values) - field(x)).max() < 1.0e-13
        assert np.abs(t.grads @ dofs - div).max() < 1.0e-13
    # RT0 does not contain a general linear field: the interpolant differs
    dofs = rt.dofs_from_values(linear(rt.dof_points))
    if d > 1:
        assert np.abs(np.einsum("i,qid->qd", dofs, vr) - linear(x)).max() > 1.0e-3


@pytest.mark.parametrize("d,family", CASES, ids=IDS)
@pytest.mark.parametrize("orientation", [1, -1])
def test_piola_map_preserves_the_dofs(d, family, orientation):
    s = H.HdivSpace(d, family)
    rs = np.random.RandomState(13 + d)
    while True:
        J = np.eye(d) + 0.4 * rs.standard_normal((d, d))
        if abs(np.linalg.det(J)) > 0.3:
            break
    if np.sign(np.linalg.det(J)) != orientation:
        J[:, 0] *= -1.0
    P0 = rs.uniform(-1, 1, d)
    N = H.area_normals(J)
    # outward: N_f points away from the opposite vertex, and |N_f| is the area of the physical face
    P = P0 + s.vertices @ J.T
    for f in range(d + 1):
        centroid = P[list(s.faces[f])].mean(axis=0)
        assert N[f] @ (centroid - P[f]) > 0.0
        if d == 2:
            area = np.linalg.norm(P[s.faces[f][1]] - P[s.faces[f][0]])
        elif d == 3:
            e1, e2 = P[s.faces[f][1]] - P[s.faces[f][0]], P[s.faces[f][2]] - P[s.faces[f][0]]
            area = 0.5 * np.linalg.norm(np.cross(e1, e2))
        else:
            area = 1.0
        assert abs(np.linalg.norm(N[f]) - area) < 1.0e-13
    # sum of the area normals of a closed surface vanishes
    assert np.abs(N.sum(axis=0)).max() < 1.0e-13
    phys = s.piola(J, s.tabulate(s.dof_points).values)          # (nQ = dim, dim, d) at the dof points
    A = s.dofs_from_values(phys.transpose(1, 0, 2), N)          # A[j, i] = sigma_i(phi_j)
    assert np.abs(A - np.eye(s.dim)).max() < 1.0e-12
    # closed form in physical coordinates: (x - P_f) / (d |K|) and lambda_a (P_a - P_f) / (d |K|)
    volume = abs(np.linalg.det(J)) / np.prod(np.arange(1, d + 1))
    xhat = rs.dirichlet(np.ones(d + 1), size=5)[:, 1:]
    x = P0 + xhat @ J.T
    got = s.piola(J, s.tabulate(xhat).values)
    lam = np.column_stack([1.0 - xhat.sum(axis=1), xhat])
    for i, (f, a) in enumerate(zip(s.dof_face, s.dof_vertex)):
        want = (x - P[f]) / (d * volume) if a < 0 else lam[:, a, None] * (P[a] - P[f]) / (d * volume)
        assert np.abs(got[:, i] - want).max() < 1.0e-12
    assert np.abs(s.divergence / abs(np.linalg.det(J)) - (1.0 if family == "rt0" else 1.0 / d) / volume).max() < 1.0e-12


def test_arguments_are_checked():
    for bad in (dict(d=4), dict(d=2, family="rt2"), dict(d=1, family="rt1")):
        with pytest.raises(ValueError):
            H.HdivSpace(**bad)
