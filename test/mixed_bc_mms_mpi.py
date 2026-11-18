#!/usr/bin/env python
"""Steady advection-diffusion MMS on the unit square or cube with a DIFFERENT boundary kind on each wall (C12, per-face
strong | weak | none; ``proteus.BoundaryKinds``), through NS_base and OneLevelTransport. Run by
test_mixed_bc_mms.py, serially and under mpiexec.

    -EPS laplacian(u) + B.grad(u) = f   in the box,     u = phi(x) psi(y..),
    phi = cos(pi x) + 2  (so du/dx = 0 on x = 0 and x = 1),   psi = prod_j sin(K (x_j + S)), j >= 2,

so the exact normal derivative vanishes on the walls x = 0 and x = 1: a face declared ``none`` there (no value, no
datum: the upwinded outflow and a zero diffusive flux) is consistent with the exact solution. Each wall is given a
kind through KINDS, e.g. ``left=strong,bottom=weak,top=weak,right=none``; the data of a strong or weak wall is the
exact solution. A wall that is not named has no condition (the same as ``none`` for a weak flux class).

Environment: ND (2|3), P, NN (nodes per direction), ADV (1: B = (1, .5, .25); 0: B = 0, Poisson),
EPS, KINDS, FLUX (iipg|sipg: Advection_DiagonalUpwind_Diffusion_*_exterior (Diffusion_*_exterior when ADV=0), weak by default;
strong: StrongDirichlet, eliminating its Dirichlet dofs, serial; keep: StrongDirichletFactory(keepDOFs=True)),
FLUXBC (noFlow|outFlow: the fluxBoundaryConditions of the strong classes), DECLARE (0: pass the setter plain, the
kinds are then not read: the baseline every existing problem is), UNDECLARED_KINDS (1: with DECLARE=0, still return
strong(g)/OPEN: refused), PART_TYPE, NLAYERS.
Rank 0 prints one line ``RESULT {json}``.
"""
import json
import os

from proteus import runtime
comm = runtime.start()

import numpy as np
from petsc4py import PETSc
from proteus import (BoundaryKinds as bk, Domain, FemTools, MeshTools, NumericalFlux, NumericalSolution, Quadrature,
                     TransportCoefficients, default_s, defaults)
from proteus.LinearAlgebraTools import SparseMatrix
from proteus.LinearSolvers import KSP_petsc4py
from proteus.NonlinearSolvers import Newton
from proteus.TimeIntegration import NoIntegration
from proteus.Transport import OneLevelTransport

nd = int(os.environ.get("ND", "2"))
P = int(os.environ.get("P", "2"))
nn = int(os.environ.get("NN", "9"))
adv = os.environ.get("ADV", "1") == "1"
eps = float(os.environ.get("EPS", "0.2"))
kinds = dict(kv.split("=") for kv in os.environ.get("KINDS", "").split(",") if kv)
fluxName = os.environ.get("FLUX", "iipg")
declare = os.environ.get("DECLARE", "1") == "1"
kinded = declare or os.environ.get("UNDECLARED_KINDS") == "1"   # return strong(g)/OPEN from a setter that was NOT declared (refused)
part = {"element": MeshTools.MeshParallelPartitioningTypes.element,
        "node": MeshTools.MeshParallelPartitioningTypes.node}[os.environ.get("PART_TYPE", "node")]
layers = int(os.environ.get("NLAYERS", "1"))

K = 1.5 * np.pi
S = 0.2
B = np.array([1.0, 0.5, 0.25])[:nd] if adv else np.zeros(nd)


def phi(x):
    return np.cos(np.pi * x) + 2.0


def u_exact(x):
    x = np.asarray(x)
    return phi(x[..., 0]) * np.prod(np.sin(K * (x[..., 1:nd] + S)), axis=-1) if nd > 1 else phi(x[..., 0])


def grad_u_exact(x):
    x = np.asarray(x)
    g = np.zeros(x.shape[:-1] + (nd,))
    psi = np.prod(np.sin(K * (x[..., 1:nd] + S)), axis=-1)
    g[..., 0] = -np.pi * np.sin(np.pi * x[..., 0]) * psi
    for i in range(1, nd):
        rest = np.prod(np.sin(K * (np.delete(x[..., 1:nd], i - 1, axis=-1) + S)), axis=-1)
        g[..., i] = phi(x[..., 0]) * K * np.cos(K * (x[..., i] + S)) * rest
    return g


def source(x):
    """f = -EPS laplacian(u) + B.grad(u), elementwise on x (..., 3)."""
    x = np.asarray(x)
    psi = np.prod(np.sin(K * (x[..., 1:nd] + S)), axis=-1)
    lap = (-np.pi ** 2 * np.cos(np.pi * x[..., 0]) * psi) + phi(x[..., 0]) * (-(nd - 1) * K ** 2) * psi
    return -eps * lap + np.einsum("...i,i->...", grad_u_exact(x), B)


class Coefficients(TransportCoefficients.TC_base):
    def __init__(self):
        TransportCoefficients.TC_base.__init__(self, 1, {}, {0: {0: "linear"}} if adv else {}, {0: {0: {0: "constant"}}},
                                               {0: {0: "u"}}, {0: {0: "constant"}}, {}, sparseDiffusionTensors={})

    def evaluate(self, t, c):
        x = c["x"]
        c[("r", 0)][:] = -source(x)
        c[("dr", 0, 0)][:] = 0.0
        a = c[("a", 0, 0)]
        a[:] = 0.0
        for i in range(nd):
            a[..., i * nd + i] = eps
        if adv:
            c[("f", 0)][:] = c[("u", 0)][..., None] * B
            c[("df", 0, 0)][:] = B


class Exact(object):
    def uOfXT(self, x, t):
        return float(u_exact(np.asarray(x)))


L = (1.0,) * nd
domain = Domain.RectangularDomain(L=L, x=(0.0,) * nd, name="box")
tag = {v: k for k, v in domain.boundaryTags.items()}


def g(x, t):
    return float(u_exact(np.asarray(x)))


_kind = {"strong": bk.strong, "weak": bk.weak}


def dirichlet(x, flag):
    name = tag.get(flag)
    k = kinds.get(name)
    if k is None:
        return None
    if k == "none":
        return bk.OPEN if kinded else None
    if k == "bare":  # a plain function in a declared setter: the class default
        return g
    return _kind[k](g) if kinded else g


o = domain.MeshOptions
o.nnx = o.nny = o.nnz = o.nn = nn
o.triangleFlag = 0
o.parallelPartitioningType = part
o.nLayersOfOverlapForParallel = layers

p = defaults.Physics_base(nd=nd, name="mixed_bc_mms")
p.domain = domain
p.LevelModelType = OneLevelTransport
p.coefficients = Coefficients()
p.initialConditions = {0: type("Zero", (), {"uOfXT": lambda self, x, t: 0.0})()}
p.dirichletConditions = {0: bk.declare(dirichlet) if declare else dirichlet}
p.advectiveFluxBoundaryConditions = {}
p.diffusiveFluxBoundaryConditions = {0: {}}
p.fluxBoundaryConditions = {0: os.environ.get("FLUXBC", "noFlow")}
p.analyticalSolution = {0: Exact()}

n = defaults.Numerics_base()
n.timeIntegration = NoIntegration
n.femSpaces = {0: lambda mesh, nd_: FemTools.C0_AffinePolynomialOnSimplex(mesh, nd_, P, "lagrange", "gll")}
n.elementQuadrature = Quadrature.SimplexGaussQuadrature(nd, 2 * P + 2)
n.elementBoundaryQuadrature = Quadrature.SimplexGaussQuadrature(nd - 1, 2 * P + 2)
n.multilevelNonlinearSolver = n.levelNonlinearSolver = Newton
n.maxNonlinearIts = 3
n.fullNewtonFlag = True
n.tolFac = 0.0
n.nl_atol_res = 1.0e-10
n.matrix = SparseMatrix
if fluxName == "strong":
    n.numericalFluxType = NumericalFlux.StrongDirichletFactory({0: os.environ.get("FLUXBC", "noFlow")})
elif fluxName == "keep":
    n.numericalFluxType = NumericalFlux.StrongDirichletFactory({0: os.environ.get("FLUXBC", "noFlow")}, keepDOFs=True)
else:
    base = {"iipg": NumericalFlux.Advection_DiagonalUpwind_Diffusion_IIPG_exterior if adv else NumericalFlux.Diffusion_IIPG_exterior,
            "sipg": NumericalFlux.Advection_DiagonalUpwind_Diffusion_SIPG_exterior if adv else NumericalFlux.Diffusion_SIPG_exterior}[fluxName]

    class _Flux(base):
        def __init__(self, *a, **kw):
            base.__init__(self, *a, **kw)
            if "PENALTY" in os.environ:
                self.penalty_constant = float(os.environ["PENALTY"])
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

so = defaults.System_base(name="mixed_bc_mms", pnList=[(p, n)], sList=[default_s])
so.tnList = [0.0, 1.0]
ns = NumericalSolution.NS_base(so, [p], [n], so.sList, runtime.RunOptions())
ns.calculateSolution("mixed_bc_mms")

model = ns.modelList[0].levelModelList[-1]
space = model.u[0].femSpace
mesh = model.mesh
dof = np.asarray(model.u[0].dof)
l2g = np.asarray(space.dofMap.l2g)
en = np.asarray(mesh.elementNodesArray)
X = np.asarray(mesh.nodeArray)
nOwned = mesh.nElements_owned
quad = Quadrature.SimplexGaussQuadrature(nd, min(2 * P + 4, 20))
xi = np.array([q[:nd] for q in quad.points])
w = np.array(quad.weights)
tab = space.referenceFiniteElement.localFunctionSpace.tabulate(xi, 1)
bary = np.column_stack([1.0 - xi.sum(axis=1), xi])
e = np.arange(nOwned)
xq = np.einsum("qv,evc->eqc", bary, X[en[e]])
J = (X[en[e, 1:]] - X[en[e, :1]])[:, :, :nd].transpose(0, 2, 1)
detJ = np.abs(np.linalg.det(J))
Jinv = np.linalg.inv(J)
uh = np.einsum("qi,ei->eq", tab.values, dof[l2g[e]])
grad_ref = np.einsum("qid,ei->eqd", tab.grads, dof[l2g[e]])
grad = np.einsum("eqd,edk->eqk", grad_ref, Jinv)
err2 = float(np.einsum("q,e,eq->", w, detJ, (uh - u_exact(xq)) ** 2))
gerr2 = float(np.einsum("q,e,eqk->", w, detJ, (grad - grad_u_exact(xq)) ** 2))

# what the model did: the rows it replaced, and the weak faces it gave the flux
force = getattr(model, "dirichletConditionsForceDOF", {})
force = {0: force} if hasattr(force, "DOFBoundaryConditionsDict") else force
nForced = sum(len(d.DOFBoundaryConditionsDict) for d in force.values())
forcedErr = max([abs(float(dof[dN]) - g_(force[0].DOFBoundaryPointDict[dN], 0.0))
                 for dN, g_ in (force[0].DOFBoundaryConditionsDict.items() if 0 in force else [])] or [0.0])
nf = model.numericalFlux
weakPts = int(nf.isDOFBoundary[0].sum())
strongPts = int(nf.isStrongBoundary[0].sum())
openPts = int(nf.isOpenBoundary[0].sum())
nElim = int(space.dim - model.dirichletConditions[0].nFreeDOF_global)
both = int((nf.isDOFBoundary[0] * nf.isStrongBoundary[0]).sum())
from mpi4py import MPI
mc = comm.comm.tompi4py()
err2, gerr2 = mc.allreduce(err2, MPI.SUM), mc.allreduce(gerr2, MPI.SUM)
nElements = mc.allreduce(int(nOwned), MPI.SUM)
forcedErr = mc.allreduce(forcedErr, MPI.MAX)
weakPts, strongPts, openPts, both = (mc.allreduce(v, MPI.SUM) for v in (weakPts, strongPts, openPts, both))
if comm.rank() == 0:
    print("RESULT " + json.dumps(dict(
        nd=nd, p=P, nn=nn, adv=adv, kinds=kinds, flux=fluxName, declare=declare, ranks=comm.size(),
        L2=err2 ** 0.5, H1=gerr2 ** 0.5, h=1.0 / (nn - 1), nElements=nElements,
        weakPts=weakPts, strongPts=strongPts, openPts=openPts, bothWeakAndStrong=both,
        forcedDofsOnRank0=nForced, forcedValueErr=forcedErr, eliminatedOnRank0=nElim,
        forceStrongConditions=bool(model.forceStrongConditions))), flush=True)
