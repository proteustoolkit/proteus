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


class HdivSpace(object):
    """RT0 or BDM1 on the reference d-simplex (d = 1, 2, 3).

    Attributes: ``d``, ``family``, ``order`` (0 for rt0, 1 for bdm1), ``dim``, ``dof_face`` (dim,),
    ``dof_vertex`` (dim,) (the local vertex of a BDM1 dof, -1 for RT0), ``face_dofs[f]``, ``dof_points``
    (dim, d) (the face centroid for RT0, the vertex for BDM1), ``divergence`` (dim,) of the reference basis.
    """

    def __init__(self, d, family="rt0"):
        if d not in (1, 2, 3):
            raise ValueError("d must be 1, 2 or 3")
        if family not in ("rt0", "bdm1"):
            raise ValueError("family must be 'rt0' or 'bdm1'")
        self.d, self.family = d, family
        self.order = 0 if family == "rt0" else 1
        self.vertices = np.vstack([np.zeros((1, d)), np.eye(d)])
        self.faces = faces(d)
        dof_face, dof_vertex = [], []
        for f in range(d + 1):
            for a in (self.faces[f] if family == "bdm1" else (-1,)):
                dof_face.append(f)
                dof_vertex.append(a)
        self.dof_face = np.array(dof_face)
        self.dof_vertex = np.array(dof_vertex)
        self.dim = len(dof_face)
        self.face_dofs = [[i for i in range(self.dim) if self.dof_face[i] == f] for f in range(d + 1)]
        centroids = np.array([self.vertices[list(self.faces[f])].mean(axis=0) for f in range(d + 1)])
        self.dof_points = np.array([centroids[f] if a < 0 else self.vertices[a]
                                    for f, a in zip(dof_face, dof_vertex)]).reshape(self.dim, d)
        self.face_centroids = centroids
        # phi = scale * lambda_a (p_a - p_f) (bdm1) or scale * (x - p_f) (rt0); 1 / (d |K|) = (d-1)!
        self._scale = float(factorial(d - 1))
        self.divergence = np.full(self.dim, self._scale * (d if family == "rt0" else 1.0))

    def tabulate(self, points):
        """Reference basis values (nQ, dim, d) at points (nQ, d); ``.grads`` holds the (constant)
        divergences (dim,) and ``.hessians`` is None."""
        d = self.d
        x = np.asarray(points, dtype=float).reshape(-1, d)
        vals = np.zeros((x.shape[0], self.dim, d))
        if self.family == "rt0":
            for i, f in enumerate(self.dof_face):
                vals[:, i, :] = self._scale * (x - self.vertices[f])
        else:
            lam = np.column_stack([1.0 - x.sum(axis=1), x])
            for i, (f, a) in enumerate(zip(self.dof_face, self.dof_vertex)):
                vals[:, i, :] = self._scale * lam[:, a, None] * (self.vertices[a] - self.vertices[f])
        return Tabulation(vals, self.divergence.copy(), None)

    def dofs_from_values(self, values, area_normal=None):
        """sigma_i(v) = N_{f(i)} . v(dof point i) from the values (.., dim, d) of v at the dof points and the
        area-weighted outward normals (.., d+1, d) (the reference ones by default)."""
        N = reference_area_normals(self.d) if area_normal is None else np.asarray(area_normal)
        return np.einsum("...id,...id->...i", N[..., self.dof_face, :], np.asarray(values, dtype=float))

    def piola(self, J, reference_values):
        """Physical basis values (.., nQ, dim, d) = J phihat / |det J| for Jacobians J (.., d, d)."""
        J = np.asarray(J, dtype=float)
        det = np.abs(np.linalg.det(J))
        return np.einsum("...Ii,qni->...qnI", J, reference_values) / det[..., None, None, None]
