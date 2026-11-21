#!/usr/bin/env python
"""Poisson on the unit square or cube with a C0 space of degree p on simplices and WEAK Dirichlet data (Nitsche),
through NS_base and OneLevelTransport; run by test_weak_dirichlet_mms.py, serially and under mpiexec.

    -div(grad u) = f,  u = g on the whole boundary (imposed weakly by Diffusion_{SIPG,IIPG,NIPG}_exterior),

with the manufactured solution of simplex_mms_mpi.py (SOLUTION=smooth|poly). The boundary dofs are ordinary
unknowns (no strong condition). Environment as simplex_mms_mpi.py, plus FLUX (sipg|iipg|nipg), PENALTY (the
numerical flux's penalty_constant; unset: the class default) and SCALING (0: switch off the degree factor
NumericalFlux.penaltyOrderFactor puts on the penalty). DG=1 uses the discontinuous monomial space of degree P with
interior penalty (FLUX=dgsipg, Advection_DiagonalUpwind_Diffusion_SIPG: interior faces and the weak boundary).
Rank 0 prints one line ``RESULT {json}``.
"""
import json
import os

from proteus import runtime
comm = runtime.start()

import numpy as np
from petsc4py import PETSc
from proteus import (Domain, FemTools, MeshTools, NumericalFlux, NumericalSolution, Quadrature,
                     TransportCoefficients, default_s, defaults)
from proteus.LinearAlgebraTools import SparseMatrix
from proteus.LinearSolvers import KSP_petsc4py
from proteus.NonlinearSolvers import Newton
from proteus.TimeIntegration import NoIntegration
from proteus.Transport import OneLevelTransport

nd = int(os.environ.get("ND", "2"))
P = int(os.environ.get("P", "3"))
family = os.environ.get("FAMILY", "lagrange")
nodes = os.environ.get("NODES", "gll")
nn = int(os.environ.get("NN", "5"))
solution = os.environ.get("SOLUTION", "smooth")
part = {"element": MeshTools.MeshParallelPartitioningTypes.element,
        "node": MeshTools.MeshParallelPartitioningTypes.node}[os.environ.get("PART_TYPE", "node")]
layers = int(os.environ.get("NLAYERS", "1"))

K = 1.5 * np.pi   # sin(K (x + 0.2)) is neither zero nor linear on the boundary
S = 0.2
C = np.array([1.0, -0.7, 0.4])[:nd]


def u_exact(x):
    """x (..., >= nd)"""
    if solution == "poly":
        return (0.3 + x[..., :nd] @ C) ** P + 0.5 * x[..., 0] ** P
    return np.prod(np.sin(K * (x[..., :nd] + S)), axis=-1)


def grad_u_exact(x):
    g = np.zeros(x.shape[:-1] + (nd,))
    if solution == "poly":
        g += (P * (0.3 + x[..., :nd] @ C) ** (P - 1))[..., None] * C
        g[..., 0] += 0.5 * P * x[..., 0] ** (P - 1)
        return g
    s, c = np.sin(K * (x[..., :nd] + S)), np.cos(K * (x[..., :nd] + S))
    for i in range(nd):
        g[..., i] = K * c[..., i] * np.prod(np.delete(s, i, axis=-1), axis=-1)
    return g


def source(x):
    """f = -laplacian(u), evaluated at one point (the coefficient class calls it point by point)."""
    x = np.asarray(x)
    if solution == "poly":
        lap = P * (P - 1) * (0.3 + x[:nd] @ C) ** (P - 2) * (C @ C) if P >= 2 else 0.0
        if P >= 2:
            lap += 0.5 * P * (P - 1) * x[0] ** (P - 2)
        return -lap
    return nd * K * K * np.prod(np.sin(K * (x[:nd] + S)))


def identity(x):
    return np.eye(nd)


def dirichlet(x, flag):
    if flag != 0:  # every exterior face of the generated box mesh has a positive flag
        return lambda x, t: float(u_exact(np.asarray(x)))
    return None


class Exact(object):
    def uOfXT(self, x, t):
        return float(u_exact(np.asarray(x)))


L = (1.0,) * nd
domain = Domain.RectangularDomain(L=L, x=(0.0,) * nd, name="box")
o = domain.MeshOptions
o.nnx = o.nny = o.nnz = o.nn = nn
o.triangleFlag = 0
o.parallelPartitioningType = part
o.nLayersOfOverlapForParallel = layers

p = defaults.Physics_base(nd=nd, name="weak_mms")
p.domain = domain
p.LevelModelType = OneLevelTransport
p.coefficients = TransportCoefficients.PoissonEquationCoefficients({0: identity}, {0: source}, 1, nd, l2proj=[False])
p.initialConditions = {0: type("Zero", (), {"uOfXT": lambda self, x, t: 0.0})()}
p.dirichletConditions = {0: dirichlet}
p.advectiveFluxBoundaryConditions = {}
p.diffusiveFluxBoundaryConditions = {0: {}}
p.analyticalSolution = {0: Exact()}

n = defaults.Numerics_base()
n.timeIntegration = NoIntegration
DG = os.environ.get("DG", "0") == "1"
if DG:
    n.femSpaces = {0: getattr(FemTools, "DG_AffineP%d_OnSimplexWithMonomialBasis" % P)}
else:
    n.femSpaces = {0: lambda mesh, nd_: FemTools.C0_AffinePolynomialOnSimplex(mesh, nd_, P, family, nodes)}
n.elementQuadrature = Quadrature.SimplexGaussQuadrature(nd, 2 * P + 2)
n.elementBoundaryQuadrature = Quadrature.SimplexGaussQuadrature(nd - 1, 2 * P + 2)
n.multilevelNonlinearSolver = n.levelNonlinearSolver = Newton
n.maxNonlinearIts = 3
n.fullNewtonFlag = True
n.tolFac = 0.0
n.nl_atol_res = 1.0e-10
n.matrix = SparseMatrix
# the same path serially and in parallel: strong Dirichlet dofs kept in the system, a direct solve
_base = {"dgsipg": NumericalFlux.Advection_DiagonalUpwind_Diffusion_SIPG, "sipg": NumericalFlux.Diffusion_SIPG_exterior, "iipg": NumericalFlux.Diffusion_IIPG_exterior,
         "nipg": NumericalFlux.Diffusion_IIPG_exterior}[os.environ.get("FLUX", "sipg")]


class _Flux(_base):
    def __init__(self, *a, **kw):
        _base.__init__(self, *a, **kw)
        if os.environ.get("FLUX") == "nipg":
            self.includeBoundaryAdjoint = True
            self.boundaryAdjoint_sigma = -1.0
        if "PENALTY" in os.environ:
            self.penalty_constant = float(os.environ["PENALTY"])
        self.penalty_order_scaling = os.environ.get("SCALING", "1") == "1"


n.numericalFluxType = _Flux
n.multilevelLinearSolver = n.levelLinearSolver = KSP_petsc4py
n.linearSmoother = None
n.linearSolverConvergenceTest = "r-true"
n.linTolFac = 0.0
n.l_atol_res = 1.0e-13
n.subgridError = None
n.shockCapturing = None
options = PETSc.Options()
options.setValue("ksp_type", "gmres")
options.setValue("ksp_rtol", 1.0e-14)
options.setValue("ksp_atol", 1.0e-15)
options.setValue("pc_type", "lu")
options.setValue("pc_factor_mat_solver_type", os.environ.get("LU", "superlu_dist" if comm.size() > 1 else "superlu"))
n.parallelPartitioningType = part
n.nLayersOfOverlapForParallel = layers

so = defaults.System_base(name="weak_mms", pnList=[(p, n)], sList=[default_s])
so.tnList = [0.0, 1.0]
ns = NumericalSolution.NS_base(so, [p], [n], so.sList, runtime.RunOptions())
ns.calculateSolution("weak_mms")

model = ns.modelList[0].levelModelList[-1]
space = model.u[0].femSpace
mesh = model.mesh
dof = np.asarray(model.u[0].dof)
l2g = np.asarray(space.dofMap.l2g)
en = np.asarray(mesh.elementNodesArray)
X = np.asarray(mesh.nodeArray)
nOwned = mesh.nElements_owned
if DG:
    # the monomial basis has no tabulate: the model's own quadrature values (degree 2P + 2, exact for the error integrand
    # up to the quadrature of the smooth solution)
    xq = np.asarray(model.q["x"])[:nOwned, :, :nd]
    dV = np.asarray(model.q["dV"])[:nOwned]
    uh = np.asarray(model.q[("u", 0)])[:nOwned]
    grad = np.asarray(model.q[("grad(u)", 0)])[:nOwned, :, :nd]
    err2 = float(np.sum(dV * (uh - u_exact(xq)) ** 2))
    gerr2 = float(np.sum(dV[..., None] * (grad - grad_u_exact(xq)) ** 2))
    linf = float(np.abs(uh - u_exact(xq)).max()) if nOwned > 0 else 0.0
else:
    quad = Quadrature.SimplexGaussQuadrature(nd, min(2 * P + 4, 20))
    xi = np.array([q[:nd] for q in quad.points])
    w = np.array(quad.weights)
    tab = space.referenceFiniteElement.localFunctionSpace.tabulate(xi, 1)
    bary = np.column_stack([1.0 - xi.sum(axis=1), xi])
    e = np.arange(nOwned)
    xq = np.einsum("qv,evc->eqc", bary, X[en[e]])
    J = (X[en[e, 1:]] - X[en[e, :1]])[:, :, :nd].transpose(0, 2, 1)        # (e, nd, nd), columns = edge vectors
    detJ = np.abs(np.linalg.det(J))
    Jinv = np.linalg.inv(J)
    uh = np.einsum("qi,ei->eq", tab.values, dof[l2g[e]])
    grad_ref = np.einsum("qid,ei->eqd", tab.grads, dof[l2g[e]])
    grad = np.einsum("eqd,edk->eqk", grad_ref, Jinv)                         # grad_x = J^-T grad_xi
    err2 = float(np.einsum("q,e,eq->", w, detJ, (uh - u_exact(xq)) ** 2))
    gerr2 = float(np.einsum("q,e,eqk->", w, detJ, (grad - grad_u_exact(xq)) ** 2))
    linf = float(np.abs(uh - u_exact(xq)).max()) if nOwned > 0 else 0.0
from mpi4py import MPI
mc = comm.comm.tompi4py()
err2, gerr2 = mc.allreduce(err2, MPI.SUM), mc.allreduce(gerr2, MPI.SUM)
linf = mc.allreduce(linf, MPI.MAX)
nElements = mc.allreduce(int(nOwned), MPI.SUM)
if comm.rank() == 0:
    print("RESULT " + json.dumps(dict(flux=os.environ.get("FLUX", "sipg"), penalty=float(os.environ.get("PENALTY", "nan")), scaling=os.environ.get("SCALING", "1"), nd=nd, p=P, family=family, nodes=nodes, nn=nn, solution=solution,
                                      ranks=comm.size(), L2=err2 ** 0.5, H1=gerr2 ** 0.5, Linf=linf,
                                      nDOF=int(space.dofMap.nDOF_all_processes), nElements=nElements,
                                      h=1.0 / (nn - 1), space_class=type(space).__name__)), flush=True)
