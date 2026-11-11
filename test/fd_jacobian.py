"""Finite-difference check of a Jacobian against its residual: a small, model-independent capability (seed of a generic FD Jacobian for proteus).

For a residual R(u) and a Jacobian J(u), compare J d with the central difference (R(u + h d) - R(u - h d)) / (2 h) in a random direction d.  If J = dR/du the
relative error falls like h^2 until round-off; an inexact J (a missing or approximated term, a mismatched regularization width, an insufficiently exact
quadrature) shows as an error that does not fall with h.  Models with kinks (a sharp Heaviside, upwinding, limiters) need h small enough that nothing changes
branch within the perturbation: pass the nodal distance of the nearest kink as `kink_distance` to have that checked.

Usage with a proteus LevelModel (the one-call form):  fd_jacobian.levelmodel_error(model, h)
Generic form:  fd_jacobian.relative_error(residual, jacobian_times, u0, h)
"""
import numpy as np


def csr_times(csr, d):
    """y = J d for (rowptr, colind, nzval), as returned by a proteus SparseMat's getCSRrepresentation()."""
    rowptr, colind, nzval = csr
    rows = np.repeat(np.arange(len(rowptr) - 1), np.diff(rowptr))
    return np.bincount(rows, weights=nzval * d[colind], minlength=len(rowptr) - 1)


def relative_error(residual, jacobian_times, u0, h, seed=0, kink_distance=None):
    """||FD - J d|| / ||J d|| for a random direction d (reproducible by `seed`).  kink_distance: smallest distance (in the units of u) from a kink of R at u0."""
    u0 = np.asarray(u0, float)
    d = np.random.default_rng(seed).standard_normal(len(u0))
    if kink_distance is not None:
        assert kink_distance > 100 * h * np.abs(d).max(), "a kink is too close to u0 for this perturbation size"
    Jd = jacobian_times(d)
    fd = (residual(u0 + h * d) - residual(u0 - h * d)) / (2 * h)
    return np.linalg.norm(fd - Jd) / np.linalg.norm(Jd)


def levelmodel_error(model, h, seed=0, kink_distance=None):
    """relative_error for a proteus LevelModel with a single unknown (getResidual(u, r) and getJacobian(jacobian)); restores the model's dofs."""
    u0 = model.u[0].dof.copy()

    def residual(u):
        r = np.zeros(len(u0))
        model.u[0].dof[:] = u
        out = model.getResidual(u.copy(), r)
        return np.array(r if out is None else out)

    def jacobian_times(d):
        model.u[0].dof[:] = u0
        model.getJacobian(model.jacobian)
        return csr_times(model.jacobian.getCSRrepresentation(), d)

    try:
        return relative_error(residual, jacobian_times, u0, h, seed=seed, kink_distance=kink_distance)
    finally:
        model.u[0].dof[:] = u0
