"""
H(div)-conforming local spaces on the reference simplex: the lowest-order Raviart-Thomas space RT0 and the
Brezzi-Douglas-Marini space BDM1, with their degrees of freedom, tabulation and the contravariant Piola map.

This module is numpy only and knows nothing about meshes; ``FemTools`` wraps it (``HdivOnSimplex``,
``FaceDOFMap``, ``Hdiv_AffineOnSimplex``). It is the vector-valued counterpart of :mod:`proteus.SimplexSpaces`.

Reference cell: the unit simplex of ``FemTools.ReferenceSimplex`` (vertices p_0 = 0, p_i = e_i), barycentric
coordinates ``lambda_0 = 1 - sum(x)``, ``lambda_i = x_{i-1}``; face ``f`` is opposite vertex ``f``; |K| = 1/d!.

Spaces and bases (K any simplex with vertices P_a, face F_f opposite P_f, outward unit normal n_f)
-------------------------------------------------------------------------------------------------
* ``rt0``: ``{a + b x}``, d+1 functions, one per face::

      phi_f(x) = (x - P_f) / (d |K|)

  The normal component on F_g is the constant ``delta_fg / |F_f|``; the divergence is ``1/|K|``.
* ``bdm1``: all linear vector fields, d(d+1) functions, one per face f and vertex a of that face::

      phi_{f,a}(x) = lambda_a(x) (P_a - P_f) / (d |K|)

  On F_f the normal component is the linear function ``lambda_a / |F_f|``; on every other face it vanishes
  (``lambda_a = 0`` on the face opposite a, and ``P_a - P_f`` is an edge of the remaining faces). The
  divergence is ``1 / (d |K|)``, and ``sum_a phi_{f,a} = phi_f`` of RT0.

Degrees of freedom
------------------
The normal trace of a function of either space is a polynomial on each face (constant for RT0, linear for
BDM1), so it is fixed by point values, and the dofs are the normal flux densities at points of the face::

      rt0:    sigma_f(v)     = |F_f| (v . n_f)(centroid of F_f)        = the flux through F_f
      bdm1:   sigma_{f,a}(v) = |F_f| (v . n_f)(P_a),  a a vertex of F_f

With them the bases above are the dual bases, ``sigma_i(phi_j) = delta_ij``. The flux of a BDM1 function
through F_f is the mean of its d dofs on that face. Two elements that share a face see the same normal trace,
so the dofs are single valued up to the sign of the normal: that sign is the dof map's business.

Local order: face-major, f = 0..d; inside a face of BDM1 the vertices of the face in increasing local order.

Piola map
---------
For an affine element x = P_0 + J xhat the basis functions are the contravariant Piola images of the
reference ones with |det J| (not det J), ``phi = J phihat / |det J|``, so that the outward flux through each
face is preserved whatever the orientation of the element; ``div phi = divhat phihat / |det J|``. The
area-weighted outward normal of a face is ``N_f = |det J| J^{-T} Nhat_f`` (:func:`area_normals`).
"""
import itertools
from math import factorial

import numpy as np

from .SimplexSpaces import Tabulation, faces


def reference_area_normals(d):
    """Nhat_f = |Fhat_f| nhat_f, the area-weighted outward normals of the reference simplex: (d+1, d)."""
    N = np.zeros((d + 1, d))
    N[0, :] = 1.0
    N[1:, :] = -np.eye(d)
    return N / factorial(d - 1)


def area_normals(J):
    """Area-weighted outward face normals (.., d+1, d) of the elements with Jacobians J (.., d, d),
    J[.., I, i] = d x_I / d xhat_i: N_f = |det J| J^{-T} Nhat_f."""
    J = np.asarray(J, dtype=float)
    d = J.shape[-1]
    det = np.abs(np.linalg.det(J))
    Jinv = np.linalg.inv(J)
    return det[..., None, None] * np.einsum("...iI,fi->...fI", Jinv, reference_area_normals(d))


# --------------------------------------------------------------------------------------------------------------
# generating sets of polynomial vector fields (monomial coefficients), used by the families without a closed
# form here and, as a cross check, by rt0 and bdm1
# --------------------------------------------------------------------------------------------------------------
FAMILIES = {"rt0": ("rt", 0), "bdm1": ("bdm", 1), "rt1": ("rt", 1), "bdm2": ("bdm", 2)}
CLOSED_FORM = ("rt0", "bdm1")
MAX_INTERIOR_DEGREE = 8      # exactness of the rule that evaluates the interior moments of a function


def _monomials(d, deg):
    """Exponent tuples of the monomials of total degree <= deg, by degree then reverse lexicographic."""
    out = [a for a in itertools.product(range(deg + 1), repeat=d) if sum(a) <= deg]
    return sorted(out, key=lambda a: (sum(a), tuple(-x for x in a)))


def _eval_monomials(mono, x):
    """(nQ, nmono) values of the monomials at points x (nQ, d)."""
    x = np.asarray(x, dtype=float)
    return np.array([np.prod(x ** np.array(a), axis=1) for a in mono]).T


def _integral_monomial(a):
    """Integral of x^a over the reference simplex: prod(a_i!) / (|a| + d)!."""
    return float(np.prod([factorial(i) for i in a])) / factorial(sum(a) + len(a))


def _generating_set(d, kind, k):
    """Coefficients G (ngen, d, nmono) of vector fields spanning RT_k (``rt``) or the full P_k^d (``bdm``) over the
    monomials of degree <= k + 1, and the monomial list."""
    mono = _monomials(d, k + 1)
    index = {a: i for i, a in enumerate(mono)}
    gens = []
    for i in range(d):
        for a in _monomials(d, k):
            g = np.zeros((d, len(mono)))
            g[i, index[a]] = 1.0
            gens.append(g)
    if kind == "rt":
        for a in _monomials(d, k):
            if sum(a) == k:                      # x times a homogeneous polynomial of degree k
                g = np.zeros((d, len(mono)))
                for i in range(d):
                    b = list(a)
                    b[i] += 1
                    g[i, index[tuple(b)]] = 1.0
                gens.append(g)
    return np.array(gens), mono


def _interior_weights(d, kind, k):
    """The test fields of the interior dofs as monomial coefficients (nw, d, nmono1), nmono1 over degree <= 1:
    constant vectors (rt: P_{k-1}^d for k = 1, bdm: Nedelec N0 = a + b x x for k = 2). Empty for k = 0 (rt0) and
    for bdm1. In the *reference* frame, centred at the reference centroid c."""
    if (kind, k) in (("rt", 0), ("bdm", 1)):
        return np.zeros((0, d, 0)), []
    if (kind, k) not in (("rt", 1), ("bdm", 2)):
        raise ValueError("interior dofs are implemented for rt1 and bdm2 only")
    mono = _monomials(d, 1)
    index = {a: i for i, a in enumerate(mono)}
    c = np.full(d, 1.0 / (d + 1))
    zero = tuple([0] * d)
    unit = [tuple(1 if j == i else 0 for j in range(d)) for i in range(d)]
    ws = []
    for i in range(d):
        w = np.zeros((d, len(mono)))
        w[i, index[zero]] = 1.0
        ws.append(w)
    if kind == "bdm":
        if d == 2:                               # (-(y - cy), x - cx)
            w = np.zeros((2, len(mono)))
            w[0, index[unit[1]]], w[0, index[zero]] = -1.0, c[1]
            w[1, index[unit[0]]], w[1, index[zero]] = 1.0, -c[0]
            ws.append(w)
        else:                                    # e_i x (x - c), i = 0, 1, 2
            for i in range(3):
                j, l = (i + 1) % 3, (i + 2) % 3        # e_i x v = (component j of result: -v_l ...) via cyclic
                w = np.zeros((3, len(mono)))
                # (e_i x v)_j = -v_l * (+1 for the cyclic order), (e_i x v)_l = v_j
                w[j, index[unit[l]]], w[j, index[zero]] = -1.0, c[l]
                w[l, index[unit[j]]], w[l, index[zero]] = 1.0, -c[j]
                ws.append(w)
    return np.array(ws), mono


def _face_lattice(m, k):
    """Integer multi-indices (N, m), denominator den, of the nodes of the degree-k Lagrange lattice on a face
    with m vertices, vertices first (increasing local vertex) then the others in decreasing lexicographic
    order; k = 0 is the centroid (all ones, denominator m)."""
    if k == 0:
        return np.ones((1, m), dtype=int), m
    nodes = sorted([a for a in itertools.product(range(k + 1), repeat=m) if sum(a) == k], reverse=True)
    vertices = [tuple(k if j == i else 0 for j in range(m)) for i in range(m)]
    rest = [a for a in nodes if a not in vertices]
    return np.array(vertices + rest, dtype=int).reshape(-1, m), k


def _face_flux_weights(m, k, lattice, den):
    """w (N,) with sum 1 such that the mean over a face of a degree-k polynomial is sum_n w_n q(node_n): the
    flux through a face is the sum of the face's dofs weighted by w (k = 0: the one dof)."""
    if k == 0 or m == 1:
        return np.ones(lattice.shape[0])
    e = m - 1                                    # dimension of the face
    mono = _monomials(e, k)
    xi = lattice[:, 1:] / float(den)
    V = _eval_monomials(mono, xi)
    moments = np.array([_integral_monomial(a) for a in mono]) * factorial(e)
    return np.linalg.solve(V.T, moments)


class HdivSpace(object):
    """RT0, RT1, BDM1 or BDM2 on the reference d-simplex (d = 1, 2, 3 for rt0 and bdm1, 2 and 3 for the others).

    Attributes: ``d``, ``family``, ``order`` (the degree of the trace: 0 rt0, 1 bdm1 and rt1, 2 bdm2),
    ``dim``, ``dof_face`` (dim,) (-1 for an interior dof), ``dof_vertex`` (dim,) (the local vertex of a dof at a
    vertex of its face, -1 otherwise), ``dof_alpha`` (dim, d) integer multi-index of the dof's node on its face
    with ``dof_den`` (rows of an interior dof are 0), ``face_dofs[f]``, ``interior_dofs``, ``dof_points``
    (dim, d) (the face node, or the centroid for an interior dof), ``face_lattice`` / ``face_lattice_den``
    (the node set of a face, in the order of ``face_dofs[f]``), ``constant_divergence`` and, if so,
    ``divergence`` (dim,) of the reference basis.
    """

    def __init__(self, d, family="rt0"):
        if family not in FAMILIES:
            raise ValueError("family must be one of %s" % (sorted(FAMILIES),))
        kind, k = FAMILIES[family]
        if d not in (1, 2, 3) or (d == 1 and family not in CLOSED_FORM):
            raise ValueError("d must be 1, 2 or 3 (2 or 3 for %s)" % (family,))
        self.d, self.family, self.kind = d, family, kind
        self.degree = k
        self.order = {"rt0": 0, "bdm1": 1, "rt1": 1, "bdm2": 2}[family]
        self.vertices = np.vstack([np.zeros((1, d)), np.eye(d)])
        self.faces = faces(d)
        m = d
        self.face_lattice, self.face_lattice_den = _face_lattice(m, self.order)
        nodes_per_face = self.face_lattice.shape[0]
        dof_face, dof_vertex, dof_alpha, dof_points = [], [], [], []
        for f in range(d + 1):
            verts = self.faces[f]
            for alpha in self.face_lattice:
                dof_face.append(f)
                vertex = -1
                if self.order > 0 and alpha.max() == self.face_lattice_den:
                    vertex = verts[int(np.argmax(alpha))]
                dof_vertex.append(vertex)
                dof_alpha.append(alpha)
                dof_points.append((alpha / float(self.face_lattice_den)) @ self.vertices[list(verts)])
        self.n_face_dofs = len(dof_face)
        self.interior_test, self._interior_mono = _interior_weights(d, kind, k)
        self.n_interior_dofs = self.interior_test.shape[0]
        centroid = np.full(d, 1.0 / (d + 1))
        for _ in range(self.n_interior_dofs):
            dof_face.append(-1)
            dof_vertex.append(-1)
            dof_alpha.append(np.zeros(m, dtype=int))
            dof_points.append(centroid)
        self.dim = len(dof_face)
        self.dof_face = np.array(dof_face)
        self.dof_vertex = np.array(dof_vertex)
        self.dof_alpha = np.array(dof_alpha, dtype=int).reshape(self.dim, m)
        self.dof_den = self.face_lattice_den
        self.face_dofs = [[i for i in range(self.dim) if self.dof_face[i] == f] for f in range(d + 1)]
        self.interior_dofs = list(range(self.n_face_dofs, self.dim))
        self.dof_points = np.array(dof_points).reshape(self.dim, d)
        self.face_centroids = np.array([self.vertices[list(self.faces[f])].mean(axis=0) for f in range(d + 1)])
        assert all(len(x) == nodes_per_face for x in self.face_dofs)
        self.face_flux_weights = _face_flux_weights(m, self.order, self.face_lattice, self.face_lattice_den)
        # closed forms: phi = scale * lambda_a (p_a - p_f) (bdm1) or scale * (x - p_f) (rt0); 1/(d |K|) = (d-1)!
        self._scale = float(factorial(d - 1))
        self.constant_divergence = family in CLOSED_FORM
        self.divergence = np.full(self.dim, self._scale * (d if family == "rt0" else 1.0)) if self.constant_divergence else None
        # generic construction by the dual basis of a generating set (always built for the families without a
        # closed form, on request for the others)
        self._generic = None
        self._interior_rule = None
        if family not in CLOSED_FORM:
            self._build_generic()

    # ---- generic path --------------------------------------------------------------------------------------
    def _build_generic(self):
        if self._generic is not None:
            return self._generic
        d = self.d
        gens, mono = _generating_set(d, self.kind, self.degree)
        ngen = gens.shape[0]
        assert ngen == self.dim, (ngen, self.dim)
        N = reference_area_normals(d)
        A = np.zeros((self.dim, ngen))
        for i in range(self.n_face_dofs):
            vals = np.einsum("gdm,m->gd", gens, _eval_monomials(mono, self.dof_points[i:i + 1])[0])
            A[i] = vals @ N[self.dof_face[i]]
        volume = 1.0 / factorial(d)
        for j in range(self.n_interior_dofs):
            w, wmono = self.interior_test[j], self._interior_mono
            for g in range(ngen):
                total = 0.0
                for c in range(d):
                    for a_i, a in enumerate(mono):
                        ca = gens[g, c, a_i]
                        if ca == 0.0:
                            continue
                        for b_i, b in enumerate(wmono):
                            cb = w[c, b_i]
                            if cb != 0.0:
                                total += ca * cb * _integral_monomial(tuple(x + y for x, y in zip(a, b)))
                A[self.n_face_dofs + j, g] = total / volume
        C = np.linalg.inv(A)
        self.dof_matrix_condition = float(np.linalg.cond(A))
        # divergence of the generators as monomial coefficients
        index = {a: i for i, a in enumerate(mono)}
        divg = np.zeros((ngen, len(mono)))
        for g in range(ngen):
            for c in range(d):
                for a_i, a in enumerate(mono):
                    if a[c] > 0 and gens[g, c, a_i] != 0.0:
                        b = list(a)
                        b[c] -= 1
                        divg[g, index[tuple(b)]] += a[c] * gens[g, c, a_i]
        self._generic = dict(gens=gens, mono=mono, C=C, divg=divg)
        return self._generic

    def tabulate(self, points, generic=False):
        """Reference basis values (nQ, dim, d) at points (nQ, d); ``.grads`` holds the divergences (nQ, dim) and
        ``.hessians`` is None. ``generic=True`` forces the dual-basis construction (a cross check for rt0, bdm1)."""
        d = self.d
        x = np.asarray(points, dtype=float).reshape(-1, d)
        if self.family in CLOSED_FORM and not generic:
            vals = np.zeros((x.shape[0], self.dim, d))
            if self.family == "rt0":
                for i, f in enumerate(self.dof_face):
                    vals[:, i, :] = self._scale * (x - self.vertices[f])
            else:
                lam = np.column_stack([1.0 - x.sum(axis=1), x])
                for i, (f, a) in enumerate(zip(self.dof_face, self.dof_vertex)):
                    vals[:, i, :] = self._scale * lam[:, a, None] * (self.vertices[a] - self.vertices[f])
            return Tabulation(vals, np.tile(self.divergence, (x.shape[0], 1)), None)
        g = self._build_generic()
        M = _eval_monomials(g["mono"], x)
        vals = np.einsum("gdm,qm,gj->qjd", g["gens"], M, g["C"])
        divs = np.einsum("gm,qm,gj->qj", g["divg"], M, g["C"])
        return Tabulation(vals, divs, None)

    # ---- degrees of freedom --------------------------------------------------------------------------------
    @property
    def interior_points(self):
        """(nq, d) and weights (nq,) (summing to 1) of the rule that evaluates the interior moments of a field."""
        if self._interior_rule is None:
            from .Quadrature import SimplexGaussQuadrature
            q = SimplexGaussQuadrature(self.d, MAX_INTERIOR_DEGREE)
            pts = np.array([p[:self.d] for p in q.points], dtype=float)
            w = np.array(q.weights, dtype=float)
            w = w / w.sum()
            test = _eval_monomials(self._interior_mono, pts) if self.n_interior_dofs else np.zeros((len(pts), 0))
            W = np.einsum("jdm,qm->qjd", self.interior_test, test) if self.n_interior_dofs else np.zeros((len(pts), 0, self.d))
            self._interior_rule = (pts, w, W)
        return self._interior_rule[0], self._interior_rule[1]

    def dofs_from_values(self, values, area_normal=None, interior=None):
        """sigma_i(v) from the values (.., dim, d) of v at the dof points and the area-weighted outward normals
        (.., d+1, d) (the reference ones by default): the face dofs are N_f . v(point). An interior dof is the
        mean over the reference element of v . w_j and needs ``interior`` = the values (.., nq, d) of the
        (reference-frame) field at ``interior_points``; the entries of ``values`` at interior dofs are ignored."""
        N = reference_area_normals(self.d) if area_normal is None else np.asarray(area_normal)
        values = np.asarray(values, dtype=float)
        nf = self.n_face_dofs
        face = np.einsum("...id,...id->...i", N[..., self.dof_face[:nf], :], values[..., :nf, :])
        if self.n_interior_dofs == 0:
            return face
        if interior is None:
            raise ValueError("%s has interior dofs: pass the field's values at interior_points" % (self.family,))
        self.interior_points
        _, w, W = self._interior_rule
        inner = np.einsum("q,...qd,qjd->...j", w, np.asarray(interior, dtype=float), W)
        return np.concatenate([face, inner], axis=-1)

    def piola(self, J, reference_values):
        """Physical basis values (.., nQ, dim, d) = J phihat / |det J| for Jacobians J (.., d, d)."""
        J = np.asarray(J, dtype=float)
        det = np.abs(np.linalg.det(J))
        return np.einsum("...Ii,qni->...qnI", J, reference_values) / det[..., None, None, None]


# --------------------------------------------------------------------------------------------------------------
# RT0 on the reference square and cube [-1, 1]^d (d = 2, 3): the lowest-order Raviart-Thomas space of
# quadrilaterals and hexahedra, Q_{1,0} x Q_{0,1} (x Q_{0,0,1}): the a-th component is linear in x_a and constant
# in the others. One dof per face.
# --------------------------------------------------------------------------------------------------------------
def cube_reference_area_normals(d):
    """Nhat_f = |Fhat_f| nhat_f for the faces of [-1,1]^d in the order of ``CubeSpaces.FACE_PLANE``: (2d, d)."""
    from .CubeSpaces import FACE_PLANE
    N = np.zeros((2 * d, d))
    for f, (axis, side) in enumerate(FACE_PLANE[d]):
        N[f, axis] = (1.0 if side == 1 else -1.0) * 2.0 ** (d - 1)
    return N


class HdivCubeSpace(object):
    """RT0 on the reference square (d = 2) or cube (d = 3) [-1,1]^d.

    The face f lies in the plane x[axis] = -1 or +1 (``CubeSpaces.FACE_PLANE``), n_f = +-e_axis and |Fhat| =
    2^(d-1). With s = +-1 the side of the face, the basis function of face f is::

        phi_f(x) = s (1 + s x_axis) / 2 / 2^(d-1) e_axis

    Its normal component is 1/|Fhat| on face f and 0 on every other face; the divergence is 2^-d, so |K|
    div phi_f = 1 = the flux. The dof is the flux density at the face centre times the face area,
    ``sigma_f(v) = N_f . v(centre of f)`` with N_f the area-weighted outward normal, which for v in the space is
    the flux of v through f. Attributes as :class:`HdivSpace` (``dim`` = 2d, ``dof_face`` = arange, no interior
    dofs, ``dof_points`` the face centres, ``constant_divergence``, ``divergence``).
    """

    family = "rt0"
    kind = "rt"
    order = 0
    degree = 0
    n_interior_dofs = 0
    interior_dofs = []
    constant_divergence = True

    def __init__(self, d, family="rt0"):
        from .CubeSpaces import CORNERS, FACE_PLANE, FACES
        if d not in (2, 3) or family != "rt0":
            raise ValueError("RT0 on cubes is implemented for d = 2 or 3 and family 'rt0'")
        self.d = d
        self.dim = 2 * d
        self.n_face_dofs = self.dim
        self.vertices = 2.0 * np.array(CORNERS[d], dtype=float) - 1.0
        self.faces = FACES[d]
        self._plane = FACE_PLANE[d]
        self.dof_face = np.arange(self.dim)
        self.dof_vertex = -np.ones(self.dim, dtype=int)
        self.face_dofs = [[f] for f in range(self.dim)]
        self.dof_alpha = np.ones((self.dim, 1), dtype=int)
        self.dof_den = 1
        self.face_lattice = np.ones((1, 1), dtype=int)
        self.face_lattice_den = 1
        self.face_flux_weights = np.ones(1)
        self.dof_points = np.array([self.vertices[list(self.faces[f])].mean(axis=0) for f in range(self.dim)])
        self.face_centroids = self.dof_points
        self.divergence = np.full(self.dim, 2.0 ** (-d))

    def tabulate(self, points, generic=False):
        """Reference basis values (nQ, dim, d); ``.grads`` holds the divergences (nQ, dim)."""
        d = self.d
        x = np.asarray(points, dtype=float).reshape(-1, d)
        vals = np.zeros((x.shape[0], self.dim, d))
        for f, (axis, side) in enumerate(self._plane):
            s = 1.0 if side == 1 else -1.0
            vals[:, f, axis] = s * (1.0 + s * x[:, axis]) / 2.0 / 2.0 ** (d - 1)
        return Tabulation(vals, np.tile(self.divergence, (x.shape[0], 1)), None)

    def dofs_from_values(self, values, area_normal=None, interior=None):
        """sigma_f(v) = N_f . v(centre of f) from the values (.., 2d, d) at the dof points and the area-weighted
        outward normals (.., 2d, d) (the reference ones by default)."""
        N = cube_reference_area_normals(self.d) if area_normal is None else np.asarray(area_normal)
        return np.einsum("...id,...id->...i", N, np.asarray(values, dtype=float))

    def piola(self, J, reference_values):
        """Physical basis values (.., nQ, dim, d) = J phihat / |det J| for the Jacobian J (.., d, d) of x(xi)."""
        J = np.asarray(J, dtype=float)
        det = np.abs(np.linalg.det(J))
        return np.einsum("...Ii,qni->...qnI", J, reference_values) / det[..., None, None, None]


def cube_area_normals(J):
    """Area-weighted outward face normals (.., 2d, d) of the parallelepipeds x = c + J xi, xi in [-1,1]^d:
    N_f = |det J| J^{-T} Nhat_f."""
    J = np.asarray(J, dtype=float)
    d = J.shape[-1]
    det = np.abs(np.linalg.det(J))
    Jinv = np.linalg.inv(J)
    return det[..., None, None] * np.einsum("...iI,fi->...fI", Jinv, cube_reference_area_normals(d))
