"""
RT1 and BDM2 on simplices as local spaces (C8 round 7): `proteus.HdivSpaces`.

Degrees of freedom. The normal component of a function of either space is a polynomial of degree 1 (RT1) or 2
(BDM2) on every face, so a face dof is the flux density ``N_f . v(x)`` at a node of the degree-1 or degree-2
Lagrange lattice of the face (vertices, then edge midpoints), with ``N_f`` the area-weighted outward normal,
as for BDM1. The remaining dofs live inside the element: the mean over the reference element of ``v . w`` for
the test fields ``w`` of the interior (RT1: the constants; BDM2: the Nedelec fields ``a + b x (x - c)``).

* Against the pinned oracle `test/fem_oracle/hdiv_tables.npz` (generators in monomials, exact rational inverse
  of the dof matrix and exact integrals, independent of the construction here): values, divergences at the
  points, the normal flux on every face, and the order and position of the dofs.
* Without the oracle: the dual basis; a basis function has no normal component on the other faces and
  its normal component on its own face is the Lagrange function of its node; the spaces contain what they must
  (RT1 contains BDM1 and a + Bx + (c.x)x; BDM2 contains RT1 and every quadratic field) and reproduce it from
  its dofs, with its divergence; the face flux weights; the generic dual-basis construction reproduces the
  closed forms of RT0 and BDM1 and is invariant under the Piola map.
"""
import os
import itertools

import numpy as np
import pytest

from proteus import HdivSpaces as H

HERE = os.path.dirname(os.path.abspath(__file__))
_data = np.load(os.path.join(HERE, "fem_oracle", "hdiv_tables.npz"))
ALL = {k: _data[k] for k in _data.files}
CELL = {2: "triangle", 3: "tetrahedron"}
NEW = [(d, family) for d in (2, 3) for family in ("rt1", "bdm2")]
NEW_IDS = ["%dd-%s" % c for c in NEW]
ALL_CASES = [(d, f) for d in (2, 3) for f in ("rt0", "bdm1", "rt1", "bdm2")]
ALL_IDS = ["%dd-%s" % c for c in ALL_CASES]
DEG = {"rt0": 0, "bdm1": 1, "rt1": 1, "bdm2": 2}


def T(d, family, item):
    return ALL["%s/%s/%s" % (family, CELL[d], item)]


def lagrange_on_face(alpha, den, lam):
    """The degree-1 or -2 Lagrange function of the lattice node alpha/den (multi-index over the face's vertices)
    at barycentric coordinates lam (.., m) of the face (a closed form, independent of the dual-basis code)."""
    alpha = np.asarray(alpha)
    if den == 1:
        return lam[..., int(np.argmax(alpha))]
    if alpha.max() == 2:                                      # vertex of a P2 lattice
        v = int(np.argmax(alpha))
        return lam[..., v] * (2.0 * lam[..., v] - 1.0)
    a, b = np.where(alpha == 1)[0]
    return 4.0 * lam[..., a] * lam[..., b]


def interior_values(s, field):
    """The values of the reference-frame field at the interior points."""
    xi, _ = s.interior_points
    return field(xi)


def dofs_of(s, field):
    """All dofs of a reference field x (nQ, d) -> (nQ, d): face dofs at the dof points, interior dofs by the rule."""
    face = field(s.dof_points)
    return s.dofs_from_values(face, interior=interior_values(s, field))


@pytest.mark.parametrize("d,family", NEW, ids=NEW_IDS)
def test_against_the_oracle(d, family):
    s = H.HdivSpace(d, family)
    assert s.dim == T(d, family, "dof_face").shape[0]
    # the same dofs in the same order: face, position of the face nodes, interior ones last
    assert (s.dof_face == T(d, family, "dof_face")).all()
    assert (s.dof_vertex == T(d, family, "dof_vertex")).all()
    nf = s.n_face_dofs
    assert np.abs(s.dof_points[:nf] - T(d, family, "dof_x")[:nf]).max() < 1.0e-15
    t = s.tabulate(T(d, family, "x"))
    assert np.abs(t.values - T(d, family, "val")).max() < 1.0e-12
    assert np.abs(t.grads - T(d, family, "divx")).max() < 1.0e-12
    face_x, face_flux = T(d, family, "face_x"), T(d, family, "face_flux")
    N = H.reference_area_normals(d)
    assert np.abs(N - T(d, family, "area_normal")).max() < 1.0e-15
    for f in range(d + 1):
        flux = np.einsum("d,qnd->qn", N[f], s.tabulate(face_x[f]).values)
        assert np.abs(flux - face_flux[f]).max() < 1.0e-12, f


@pytest.mark.parametrize("d,family", [c for c in ALL_CASES if c[1] in H.CLOSED_FORM], ids=[i for c, i in zip(ALL_CASES, ALL_IDS) if c[1] in H.CLOSED_FORM])
def test_generic_construction_reproduces_the_closed_forms(d, family):
    s = H.HdivSpace(d, family)
    x = np.random.RandomState(31).dirichlet(np.ones(d + 1), size=8)[:, 1:]
    a, b = s.tabulate(x), s.tabulate(x, generic=True)
    assert np.abs(a.values - b.values).max() < 1.0e-13
    assert np.abs(a.grads - b.grads).max() < 1.0e-12


@pytest.mark.parametrize("d,family", NEW, ids=NEW_IDS)
def test_layout_and_duality(d, family):
    s = H.HdivSpace(d, family)
    k = DEG[family]
    nodes = 3 if (d == 2 and k == 1) or (d == 3 and k == 1) else None
    per_face = {("rt1", 2): 2, ("rt1", 3): 3, ("bdm2", 2): 3, ("bdm2", 3): 6}[(family, d)]
    interior = {("rt1", 2): 2, ("rt1", 3): 3, ("bdm2", 2): 3, ("bdm2", 3): 6}[(family, d)]
    assert [len(x) for x in s.face_dofs] == [per_face] * (d + 1)
    assert len(s.interior_dofs) == interior == s.n_interior_dofs
    assert s.dim == (d + 1) * per_face + interior == {("rt1", 2): 8, ("rt1", 3): 15, ("bdm2", 2): 12, ("bdm2", 3): 30}[(family, d)]
    assert s.dof_matrix_condition < 1.0e3
    lam = np.column_stack([1.0 - s.dof_points.sum(axis=1), s.dof_points])
    # face nodes lie on their faces; interior dofs are not on one
    assert np.abs(lam[np.arange(s.n_face_dofs), s.dof_face[:s.n_face_dofs]]).max() < 1.0e-15
    assert (s.dof_face[s.n_face_dofs:] == -1).all()
    # dual basis: sigma_i(phi_j) = delta_ij
    xi, _ = s.interior_points
    face_vals = s.tabulate(s.dof_points).values              # (dim_points, dim_basis, d)
    int_vals = s.tabulate(xi).values                         # (nq, dim_basis, d)
    A = np.array([s.dofs_from_values(face_vals[:, j, :], interior=int_vals[:, j, :]) for j in range(s.dim)])
    assert np.abs(A - np.eye(s.dim)).max() < 1.0e-12


@pytest.mark.parametrize("d,family", NEW, ids=NEW_IDS)
def test_normal_traces(d, family):
    """On its own face the normal flux density of a face basis function is the Lagrange function of its node,
    on every other face it vanishes, and an interior basis function has no flux on any face."""
    s = H.HdivSpace(d, family)
    N = H.reference_area_normals(d)
    rs = np.random.RandomState(32)
    for f in range(d + 1):
        w = rs.dirichlet(np.ones(d), size=7)
        x = w @ s.vertices[list(s.faces[f])]
        flux = np.einsum("d,qnd->qn", N[f], s.tabulate(x).values)
        for i in range(s.dim):
            if s.dof_face[i] == f:
                want = lagrange_on_face(s.dof_alpha[i], s.dof_den, w)
                assert np.abs(flux[:, i] - want).max() < 1.0e-12, (f, i)
            else:
                assert np.abs(flux[:, i]).max() < 1.0e-12, (f, i)


@pytest.mark.parametrize("d,family", NEW, ids=NEW_IDS)
def test_the_spaces_contain_what_they_should_and_reproduce_it(d, family):
    rs = np.random.RandomState(33 + d)
    a, B = rs.uniform(-1, 1, d), rs.uniform(-1, 1, (d, d))
    c = rs.uniform(-1, 1, d)
    S = rs.uniform(-1, 1, (d, d, d))
    S = 0.5 * (S + S.transpose(0, 2, 1))
    linear = lambda y: a + y @ B.T
    raviart1 = lambda y: a + y @ B.T + (y @ c)[:, None] * y            # RT1
    quadratic = lambda y: a + y @ B.T + np.einsum("qj,ijk,qk->qi", y, S, y)   # P2^d
    divs = {"linear": (lambda y: np.trace(B) + 0 * y[:, 0]),
            "raviart1": (lambda y: np.trace(B) + (d + 1) * (y @ c)),
            "quadratic": (lambda y: np.trace(B) + 2.0 * np.einsum("ijk,qk,ij->q", S, y, np.eye(d)))}
    fields = {"linear": linear, "raviart1": raviart1, "quadratic": quadratic}
    contains = {"rt1": ("linear", "raviart1"), "bdm2": ("linear", "raviart1", "quadratic")}[family]
    s = H.HdivSpace(d, family)
    x = rs.dirichlet(np.ones(d + 1), size=9)[:, 1:]
    t = s.tabulate(x)
    for name, field in fields.items():
        dofs = dofs_of(s, field)
        rec = np.einsum("i,qid->qd", dofs, t.values)
        err = np.abs(rec - field(x)).max()
        if name in contains:
            assert err < 1.0e-12, name
            assert np.abs(t.grads @ dofs - divs[name](x)).max() < 1.0e-12, name
        else:
            assert err > 1.0e-3, name                           # a quadratic field is not in RT1


@pytest.mark.parametrize("d,family", NEW, ids=NEW_IDS)
def test_face_flux_weights(d, family):
    """sum_n w_n dof_n is the flux through the face (the mean over the face of the normal flux density)."""
    from proteus.Quadrature import SimplexGaussQuadrature
    s = H.HdivSpace(d, family)
    assert abs(s.face_flux_weights.sum() - 1.0) < 1.0e-14
    N = H.reference_area_normals(d)
    q = SimplexGaussQuadrature(d - 1, 8)
    pts = np.array([p[:d - 1] for p in q.points]).reshape(-1, d - 1)
    wts = np.array(q.weights)
    wts = wts / wts.sum()
    lam = np.column_stack([1.0 - pts.sum(axis=1), pts])           # barycentric on the face
    for f in range(d + 1):
        x = lam @ s.vertices[list(s.faces[f])]
        flux = np.einsum("d,qnd->qn", N[f], s.tabulate(x).values)
        mean = wts @ flux                                          # (dim,)
        weights = np.zeros(s.dim)
        weights[s.face_dofs[f]] = s.face_flux_weights
        assert np.abs(mean - weights).max() < 1.0e-12


@pytest.mark.parametrize("d,family", ALL_CASES, ids=ALL_IDS)
@pytest.mark.parametrize("orientation", [1, -1])
def test_piola_map_preserves_the_dofs(d, family, orientation):
    """With the physical face dofs N_f . v and the interior moments of the pulled-back field |det J| J^{-1} v,
    the mapped basis is dual to them, whatever the orientation of the element."""
    s = H.HdivSpace(d, family)
    rs = np.random.RandomState(34 + d)
    while True:
        J = np.eye(d) + 0.4 * rs.standard_normal((d, d))
        if abs(np.linalg.det(J)) > 0.3:
            break
    if np.sign(np.linalg.det(J)) != orientation:
        J[:, 0] *= -1.0
    N = H.area_normals(J)
    detJ = abs(np.linalg.det(J))
    xi, _ = s.interior_points
    nf = s.n_face_dofs
    phys_face = s.piola(J, s.tabulate(s.dof_points).values)       # (dim_points, dim_basis, d)
    phys_int = s.piola(J, s.tabulate(xi).values) if s.n_interior_dofs else None
    for j in range(s.dim):
        kw = {}
        if s.n_interior_dofs:
            kw["interior"] = detJ * np.einsum("Ij,qj->qI", np.linalg.inv(J), phys_int[:, j, :])
        got = s.dofs_from_values(phys_face[:, j, :], N, **kw)
        want = np.zeros(s.dim)
        want[j] = 1.0
        assert np.abs(got - want).max() < 1.0e-11, j
    # the divergence is the reference one over |det J|: central differences of the mapped basis in x = J xi
    x = rs.dirichlet(np.ones(d + 1), size=4)[:, 1:]
    h = 1.0e-5
    Jinv = np.linalg.inv(J)
    div_fd = np.zeros((len(x), s.dim))
    for i in range(d):
        e = np.zeros(d)
        e[i] = h
        dv = (s.piola(J, s.tabulate(x + e).values) - s.piola(J, s.tabulate(x - e).values)) / (2 * h)   # (nQ, dim, d)
        div_fd += np.einsum("qnI,I->qn", dv, Jinv[i, :])
    assert np.abs(div_fd - s.tabulate(x).grads / detJ).max() < 1.0e-6


def test_interior_dofs_need_their_values():
    s = H.HdivSpace(2, "rt1")
    with pytest.raises(ValueError):
        s.dofs_from_values(np.zeros((s.dim, 2)))


def test_argument_checks():
    for bad in (dict(d=1, family="bdm2"), dict(d=1, family="rt1"), dict(d=2, family="rt2"), dict(d=4, family="bdm2")):
        with pytest.raises(ValueError):
            H.HdivSpace(**bad)


# ---- RT0 on the reference square and cube ------------------------------------------------------------------
CUBE = [2, 3]


@pytest.mark.parametrize("d", CUBE)
def test_cube_rt0_is_dual_to_its_face_fluxes_and_has_the_right_traces(d):
    s = H.HdivCubeSpace(d)
    assert s.dim == 2 * d and [len(x) for x in s.face_dofs] == [1] * (2 * d)
    A = np.array([s.dofs_from_values(s.tabulate(s.dof_points).values[:, j, :]) for j in range(s.dim)])
    assert np.abs(A - np.eye(s.dim)).max() < 1.0e-15
    N = H.cube_reference_area_normals(d)
    assert np.abs(np.linalg.norm(N, axis=1) - 2.0 ** (d - 1)).max() < 1.0e-15      # |face| of [-1,1]^d
    assert np.abs(N.sum(axis=0)).max() < 1.0e-15
    rs = np.random.RandomState(40)
    from proteus.CubeSpaces import FACE_PLANE
    for f, (axis, side) in enumerate(FACE_PLANE[d]):
        x = rs.uniform(-1, 1, (6, d))
        x[:, axis] = 1.0 if side else -1.0
        flux = np.einsum("d,qnd->qn", N[f], s.tabulate(x).values)
        want = np.zeros(s.dim)
        want[f] = 1.0
        assert np.abs(flux - want).max() < 1.0e-14           # flux density 1 through its own face, 0 through the others
    # each component is linear in its own variable and constant in the others; divergence 2^-d
    x = rs.uniform(-1, 1, (5, d))
    t = s.tabulate(x)
    assert np.abs(t.grads - 2.0 ** (-d)).max() < 1.0e-15
    h = 1.0e-6
    for i in range(d):
        e = np.zeros(d)
        e[i] = h
        dv = (s.tabulate(x + e).values - s.tabulate(x - e).values) / (2 * h)
        for comp in range(d):
            if comp != i:
                assert np.abs(dv[:, :, comp]).max() < 1.0e-9      # d v_comp / d x_i = 0 for comp != i


@pytest.mark.parametrize("d", CUBE)
@pytest.mark.parametrize("orientation", [1, -1])
def test_cube_rt0_piola_preserves_the_fluxes(d, orientation):
    s = H.HdivCubeSpace(d)
    rs = np.random.RandomState(41 + d)
    while True:
        J = np.eye(d) + 0.4 * rs.standard_normal((d, d))
        if abs(np.linalg.det(J)) > 0.3:
            break
    if np.sign(np.linalg.det(J)) != orientation:
        J[:, 0] *= -1.0
    N = H.cube_area_normals(J)
    assert np.abs(N.sum(axis=0)).max() < 1.0e-13
    phys = s.piola(J, s.tabulate(s.dof_points).values)
    A = np.array([s.dofs_from_values(phys[:, j, :], N) for j in range(s.dim)])
    assert np.abs(A - np.eye(s.dim)).max() < 1.0e-12


def test_cube_rt0_arguments_and_the_affine_check():
    from proteus import FemTools
    for bad in (dict(d=1), dict(d=2, family="bdm1"), dict(d=4)):
        with pytest.raises(ValueError):
            H.HdivCubeSpace(**bad)
    X = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    FemTools.cubeElementJacobians(X, np.array([[0, 1, 2, 3]]), 2)             # a square: fine
    X[2] = [1.4, 1.0, 0.0]                                                    # a trapezoid: not a parallelogram
    with pytest.raises(ValueError):
        FemTools.cubeElementJacobians(X, np.array([[0, 1, 2, 3]]), 2)
