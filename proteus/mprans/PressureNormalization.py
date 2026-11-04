"""The constant of a pressure that no boundary condition fixes: zero mean (decided by cekees, 2026-11-03).

In a closed tank the pressure increment (PresInc), the pressure (Pres) and the initial pressure (PresInit) are
determined only up to a constant, and each run picked one by its own means: PresInc pinned dof 0 in serial and used a
constant null space in parallel, PresInit took whatever its solver returned. So pressures, pressure gauges and
pressure forces depended on the rank count. Each of these models declares zeroMean on its coefficients, and
NumericalSolution.postStep shifts its solution to zero mean when it has no Dirichlet condition anywhere, before the
model's own postStep and so before any other model reads it; the solver's unknowns (uList) are shifted with the dofs. With a Dirichlet
condition (an open top, a prescribed outlet pressure) the pressure is already fixed and is left alone.

The mean is the integral mean of the discrete field, sum_i m_i p_i / sum_i m_i with m_i = int phi_i (the row sums of
the mass matrix), over the owned dofs and reduced over the ranks: it does not depend on the mesh grading, as a plain
average of the dofs would.
"""
import numpy as np
from proteus import Comm
from proteus.Comm import globalMax, globalSum


def _hasDirichlet(model):
    nf = getattr(model, "numericalFlux", None)
    flags = getattr(nf, "isDOFBoundary", {}).get(0) if nf is not None else None
    local = float(flags.max()) if flags is not None and flags.size else 0.0
    for dc in getattr(model, "dirichletConditions", {}).values():
        if dc is not None and len(dc.DOFBoundaryConditionsDict):
            local = 1.0
    return globalMax(local) > 0.0


def _massWeights(model):
    """m_i = int phi_i for every local dof (complete for the owned ones)."""
    fs = model.u[0].femSpace
    xi = model.elementQuadraturePoints
    w = model.elementQuadratureWeights[('u', 0)]
    nE, nQ, nd = model.mesh.nElements_global, len(w), model.nSpace_global
    J = np.zeros((nE, nQ, nd, nd), 'd')
    invJ = np.zeros((nE, nQ, nd, nd), 'd')
    detJ = np.zeros((nE, nQ), 'd')
    fs.elementMaps.getJacobianValues(xi, J, invJ, detJ)
    local = np.einsum('ek,k,ki->ei', np.abs(detJ), w, np.asarray(fs.psi))
    m = np.zeros(model.u[0].dof.shape, 'd')
    np.add.at(m, fs.dofMap.l2g, local)
    return m


def zero_mean(model):
    """Shift the level model's pressure (dofs and its values at the quadrature points) to zero integral mean, unless a
    Dirichlet condition fixes it; returns the shift, which the caller also takes from the solver's unknowns.
    Collective: every rank calls it."""
    if not hasattr(model, "_zeroMean"):
        model._zeroMean = not _hasDirichlet(model)
    if not model._zeroMean:
        return 0.0
    comm = Comm.get()
    if comm.size() > 1:
        dofMap = model.u[0].femSpace.dofMap
        nOwned = dofMap.dof_offsets_subdomain_owned[comm.rank() + 1] - dofMap.dof_offsets_subdomain_owned[comm.rank()]
    else:
        nOwned = model.u[0].dof.shape[0]
    m = _massWeights(model)[:nOwned]
    mean = globalSum(float(np.dot(m, model.u[0].dof[:nOwned]))) / globalSum(float(m.sum()))
    model.u[0].dof -= mean
    for store in (model.q, getattr(model, "ebqe", {}), getattr(model, "ebq", {})):
        if ('u', 0) in store:
            store[('u', 0)] -= mean
    return mean
