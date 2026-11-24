#!/usr/bin/env python
"""RANS2P (single phase Navier-Stokes, 2D and 3D) with a DIFFERENT boundary kind on each wall and each component (C12, per-face
strong | weak | none; ``proteus.BoundaryKinds``), against a manufactured solution. Run by test_rans2p_mixed_bc_mms.py,
serially and under mpiexec.

    (u . grad) u - NU laplacian(u) + grad(p) = F,   div(u) = 0   on the unit square or cube,   rho = 1,

    stream function  psi = f(x) y + m(x) q(y) h(z),   u = psi_y,  v = -psi_x,  w = 0   (div u = 0 exactly),
    f = 2 + (1 - x)^4 / 2,   m = (1 - x)^3 cos(pi x) / 2,   q = sin(K y) / K,   h = 1 + 0.3 cos(K z)   (h = 1 in 2D),
    p = cos(pi x) cos(K y) (cos(K z)),

so on the wall x = 1 the velocity is (f(1), 0, 0) with f' = f'' = m = m' = m'' = 0 there: u > 0 (outflow), the full viscous stress
(the symmetric gradient RANS2P uses) has no shear and no normal part, and a face declared ``none`` for the velocity there (no
value, no datum, the natural condition) is consistent with the exact solution; the pressure datum on that wall closes the problem
(no null space). F is evaluated from the exact solution by sympy. Every strong and weak
wall carries the exact solution as its data.

Environment: ND (2|3), NN (nodes per direction), NU, KINDS (velocity: ``left=strong,bottom=weak,...``; a wall that is not
named has no condition), PKINDS (pressure: the same), PDECLARE (0: the pressure setter is not declared while the velocity ones are), DECLARE (0: plain setters, the kinds are not read: every named wall is
then weak, the baseline every existing problem is), ALL (strong: coefficients.forceStrongDirichlet, the legacy all-strong
model), SPACE (p1p1: equal order with the model's stabilization, the decks' choice; p1p2: P2 velocity, which crashes
RANS2P2D::calculateResidual on some meshes, not used by the tests), SGE (1: the subgrid error terms, 0: off), PART_TYPE, NLAYERS.
Rank 0 prints one line ``RESULT {json}``.
"""
import json
import os

from proteus import runtime
comm = runtime.start()
from proteus import Profiling
Profiling.logLevel = int(os.environ.get('LOGLEVEL', '1'))
Profiling.verbose = Profiling.logLevel > 1

import numpy as np
import sympy as sp
from petsc4py import PETSc
from proteus import (BoundaryKinds as bk, Domain, FemTools, MeshTools, NumericalSolution, Quadrature, default_s, defaults)
from proteus.LinearAlgebraTools import SparseMatrix
from proteus.LinearSolvers import KSP_petsc4py
from proteus.NonlinearSolvers import Newton
from proteus.TimeIntegration import NoIntegration
from proteus.kernels import RANS2P

nd = int(os.environ.get("ND", "2"))
nn = int(os.environ.get("NN", "9"))
NU = float(os.environ.get("NU", "0.1"))
kinds = dict(kv.split("=") for kv in os.environ.get("KINDS", "left=strong,bottom=weak,top=strong,right=none").split(",") if kv)
pkinds = dict(kv.split("=") for kv in os.environ.get("PKINDS", "right=weak").split(",") if kv)
declare = os.environ.get("DECLARE", "1") == "1"
allStrong = os.environ.get("ALL") == "strong"
space = os.environ.get("SPACE", "p1p1")
part = {"element": MeshTools.MeshParallelPartitioningTypes.element,
        "node": MeshTools.MeshParallelPartitioningTypes.node}[os.environ.get("PART_TYPE", "node")]
layers = int(os.environ.get("NLAYERS", "1"))

K = 1.5 * np.pi
X, Y, Z = sp.symbols("x y z")
f = 2 + (1 - X) ** 4 / 2
m = (1 - X) ** 3 * sp.cos(sp.pi * X) / 2
q = sp.sin(K * Y) / K
h = 1 + sp.Rational(3, 10) * sp.cos(K * Z) if nd == 3 else sp.Integer(1)
psi = f * Y + m * q * h
u_ = sp.diff(psi, Y)
v_ = -sp.diff(psi, X)
w_ = sp.Integer(0)
p_ = sp.cos(sp.pi * X) * sp.cos(K * Y) * (sp.cos(K * Z) if nd == 3 else 1)
vel_ = [u_, v_, w_][:nd]
coords = [X, Y, Z][:nd]
assert sp.expand(sum(sp.diff(vel_[i], coords[i]) for i in range(nd))) == 0
# rho = 1: (u.grad)u - NU lap(u) + grad(p) = F   (the kernel's  mom_source = -g - force; its convection is always on:
# the coefficients' ``stokes`` flag is not read)
force_ = [sum(vel_[j] * sp.diff(vel_[i], coords[j]) for j in range(nd))
          - NU * sum(sp.diff(vel_[i], c, 2) for c in coords) + sp.diff(p_, coords[i]) for i in range(nd)]
lam = lambda e: sp.lambdify((X, Y, Z), e, "numpy")
u_f, v_f, w_f, p_f = lam(u_), lam(v_), lam(w_), lam(p_)
vel_f = [u_f, v_f, w_f][:nd]
force_f = [lam(e) for e in force_]
grad_vel_f = [[lam(sp.diff(vel_[i], c)) for c in coords] for i in range(nd)]
grad_p_f = [lam(sp.diff(p_, c)) for c in coords]


def _xyz(x):
    x = np.asarray(x)
    return x[..., 0], x[..., 1], (x[..., 2] if nd == 3 else 0.0 * x[..., 0])


domain = Domain.RectangularDomain(L=(1.0,) * nd, x=(0.0,) * nd, name="box")
tag = {v: k for k, v in domain.boundaryTags.items()}
_kind = {"strong": bk.strong, "weak": bk.weak}


def make_setter(table, fun, declared=None):
    declared = declare if declared is None else declared
    data = lambda x, t: float(fun(*_xyz(x)))

    def setter(x, flag):
        name = tag.get(flag)
        k = table.get(name)
        if k is None:
            return None
        if k == "none":
            return bk.OPEN if declare else None
        if k == "bare":
            return data
        return _kind[k](data) if declare else data
    return bk.declare(setter) if declared else setter


o = domain.MeshOptions
o.nnx = o.nny = o.nnz = o.nn = nn
o.triangleFlag = 0
o.parallelPartitioningType = part
o.nLayersOfOverlapForParallel = layers

p = defaults.Physics_base(nd=nd, name="rans2p_mixed_bc")
p.domain = domain
p.LevelModelType = RANS2P.LevelModel
p.coefficients = RANS2P.Coefficients(epsFact=0.0, rho_0=1.0, nu_0=NU, rho_1=1.0, nu_1=NU, g=np.zeros(nd), nd=nd, LS_model=None,
                                     epsFact_density=0.0, stokes=True, useVF=0.0, useRBLES=0.0, useMetrics=1.0,
                                     forceStrongDirichlet=allStrong, NONCONSERVATIVE_FORM=1.0,
                                     MOMENTUM_SGE=float(os.environ.get("SGE", "1.0")), PRESSURE_SGE=float(os.environ.get("SGE", "1.0")),
                                     VELOCITY_SGE=float(os.environ.get("SGE", "1.0")),
                                     PRESSURE_PROJECTION_STABILIZATION=0.0, eb_penalty_constant=float(os.environ.get("PENALTY", "100.0")),
                                     force_x=lambda x: float(force_f[0](*_xyz(x))),
                                     force_y=lambda x: float(force_f[1](*_xyz(x))),
                                     force_z=(lambda x: float(force_f[2](*_xyz(x)))) if nd == 3 else None)
p.initialConditions = {i: type("Zero", (), {"uOfXT": lambda self, x, t: 0.0})() for i in range(nd + 1)}
p.dirichletConditions = {0: make_setter(pkinds, p_f, declared=declare and os.environ.get("PDECLARE", "1") == "1")}
for i, fn in enumerate(vel_f):
    p.dirichletConditions[i + 1] = make_setter(kinds, fn)
p.advectiveFluxBoundaryConditions = {}
p.diffusiveFluxBoundaryConditions = {i: {} for i in range(nd + 1)}
p.fluxBoundaryConditions = {i: "outFlow" for i in range(nd + 1)}

n = defaults.Numerics_base()
n.timeIntegration = NoIntegration
if space == "p1p1":
    basis = lambda mesh, nd_: FemTools.C0_AffineLinearOnSimplexWithNodalBasis(mesh, nd_)
    n.femSpaces = {i: basis for i in range(nd + 1)}
else:
    n.femSpaces = {0: FemTools.C0_AffineLinearOnSimplexWithNodalBasis}
    n.femSpaces.update({i + 1: FemTools.C0_AffineQuadraticOnSimplexWithNodalBasis for i in range(nd)})
n.elementQuadrature = Quadrature.SimplexGaussQuadrature(nd, 5)
n.elementBoundaryQuadrature = Quadrature.SimplexGaussQuadrature(nd - 1, 5)
n.multilevelNonlinearSolver = n.levelNonlinearSolver = Newton
n.maxNonlinearIts = 20
n.fullNewtonFlag = True
n.tolFac = 0.0
n.nl_atol_res = 1.0e-9
n.matrix = SparseMatrix
n.numericalFluxType = RANS2P.NumericalFlux
n.multilevelLinearSolver = n.levelLinearSolver = KSP_petsc4py
n.linear_solver_options_prefix = "rans2p_"
n.linearSmoother = None
n.linearSolverConvergenceTest = "r-true"
n.linTolFac = 0.0
n.l_atol_res = 1.0e-11
n.subgridError = RANS2P.SubgridError(p.coefficients, nd, lag=False, hFactor=1.0)
n.shockCapturing = RANS2P.ShockCapturing(p.coefficients, nd, 0.0, lag=False)
options = PETSc.Options()
options.setValue("rans2p_ksp_type", "gmres")
options.setValue("rans2p_ksp_rtol", 1.0e-10)
options.setValue("rans2p_ksp_atol", 1.0e-11)
options.setValue("rans2p_ksp_max_it", 50)
options.setValue("rans2p_pc_type", "lu")
options.setValue("rans2p_pc_factor_mat_solver_type", os.environ.get("LU", "superlu_dist" if comm.size() > 1 else "superlu"))
n.parallelPartitioningType = part
n.nLayersOfOverlapForParallel = layers

so = defaults.System_base(name="rans2p_mixed_bc", pnList=[(p, n)], sList=[default_s])
so.tnList = [0.0, 1.0]
ns = NumericalSolution.NS_base(so, [p], [n], so.sList, runtime.RunOptions())
# the constant-pressure null space flag at the FIRST Jacobian (zero velocity: where it can be True); later ones see the solution
_model = ns.modelList[0].levelModelList[-1]
_firstNullSpace = []
_getJacobian = _model.getJacobian


def _recordingGetJacobian(jacobian):
    out = _getJacobian(jacobian)
    if not _firstNullSpace:
        _firstNullSpace.append(bool(_model.pp_hasConstantNullSpace))
    return out


_model.getJacobian = _recordingGetJacobian
if os.environ.get("FAULT"):
    import faulthandler
    faulthandler.enable(all_threads=True)   # replaces the PUMI signal handler, which prints no Python frames
ns.calculateSolution("rans2p_mixed_bc")

model = ns.modelList[0].levelModelList[-1]
mesh = model.mesh
nOwned = mesh.nElements_owned
en = np.asarray(mesh.elementNodesArray)
Xn = np.asarray(mesh.nodeArray)
e = np.arange(nOwned)
bary_nodes = Xn[en[e]]
J = (Xn[en[e, 1:]] - Xn[en[e, :1]])[:, :, :nd].transpose(0, 2, 1)
detJ = np.abs(np.linalg.det(J))
Jinv = np.linalg.inv(J)
quad = Quadrature.SimplexGaussQuadrature(nd, 8)
xi = np.array([q[:nd] for q in quad.points])
wq = np.array(quad.weights)
bary = np.column_stack([1.0 - xi.sum(axis=1), xi])
xq = np.einsum("qv,evc->eqc", bary, bary_nodes)
xs = [xq[..., i] for i in range(3)] if nd == 3 else [xq[..., 0], xq[..., 1], 0.0 * xq[..., 0]]


def errors(ci, exact, exact_grads):
    fs = model.u[ci].femSpace
    dof = np.asarray(model.u[ci].dof)
    l2g = np.asarray(fs.dofMap.l2g)
    lfs = fs.referenceFiniteElement.localFunctionSpace
    vals = np.array([[b(x) for b in lfs.basis] for x in xi])
    grads = np.array([[gb(x)[:nd] for gb in lfs.basisGradients] for x in xi])
    uh = np.einsum("qi,ei->eq", vals, dof[l2g[e]])
    grad_ref = np.einsum("qid,ei->eqd", grads, dof[l2g[e]])
    grad = np.einsum("eqd,edk->eqk", grad_ref, Jinv)
    gex = np.stack([exact_grads[k](*xs) * np.ones_like(uh) for k in range(nd)], axis=-1)
    return (float(np.einsum("q,e,eq->", wq, detJ, (uh - exact(*xs)) ** 2)),
            float(np.einsum("q,e,eqk->", wq, detJ, (grad - gex) ** 2)))


res = {}
res["p"] = errors(0, p_f, grad_p_f)
for i in range(nd):
    res["u%d" % i] = errors(i + 1, vel_f[i], grad_vel_f[i])

# what the model did: the rows it replaced, and the faces it gave the flux
nf = model.numericalFlux
force = model.dirichletConditionsForceDOF
forcedErr = 0.0
nForced = 0
for cj, d in (force.items() if hasattr(force, "items") else []):
    dof = np.asarray(model.u[cj].dof)
    nForced += len(d.DOFBoundaryConditionsDict)
    for dN, g_ in d.DOFBoundaryConditionsDict.items():
        forcedErr = max(forcedErr, abs(float(dof[dN]) - g_(d.DOFBoundaryPointDict[dN], 0.0)))
faces = {}
for ci in range(nd + 1):
    faces[ci] = dict(weak=int(nf.isDOFBoundary[ci].sum()), strong=int(nf.isStrongBoundary[ci].sum()),
                     open=int(nf.isOpenBoundary[ci].sum()),
                     both=int((nf.isDOFBoundary[ci] * nf.isStrongBoundary[ci]).sum()))
from mpi4py import MPI
mc = comm.comm.tompi4py()
tot = {k: (mc.allreduce(v[0], MPI.SUM) ** 0.5, mc.allreduce(v[1], MPI.SUM) ** 0.5) for k, v in res.items()}
forcedErr = mc.allreduce(forcedErr, MPI.MAX)
nForcedAll = mc.allreduce(nForced, MPI.SUM)
for ci in faces:
    for k in faces[ci]:
        faces[ci][k] = mc.allreduce(faces[ci][k], MPI.SUM)
nElements = mc.allreduce(int(nOwned), MPI.SUM)
# first and last Newton residual of the (single) solve, for the Jacobian-consistency reading
if comm.rank() == 0:
    print("RESULT " + json.dumps(dict(
        nd=nd, nn=nn, kinds=kinds, pkinds=pkinds, declare=declare, all=allStrong, space=space, ranks=comm.size(),
        h=1.0 / (nn - 1), nElements=nElements,
        L2={k: v[0] for k, v in tot.items()}, H1={k: v[1] for k, v in tot.items()},
        faces=faces, forcedDofs=nForcedAll, forcedValueErr=forcedErr,
        forceStrongConditions=bool(model.forceStrongConditions),
        nullSpaceAtFirstJacobian=_firstNullSpace[0])), flush=True)

if os.environ.get("DIAG") and comm.rank() == 0:
    nodes = np.asarray(mesh.nodeArray)
    pd = np.asarray(model.u[0].dof)
    pe = p_f(*(nodes[:len(pd), i] for i in range(3)))
    print("DIAG p_h - p_exact: min %g max %g ; corr(p_h, p_exact) %g ; lstsq p_h = a p_exact + b: %s" % (
        (pd - pe).min(), (pd - pe).max(), np.corrcoef(pd, pe)[0, 1], np.linalg.lstsq(np.c_[pe, np.ones_like(pe)], pd, rcond=None)[0]))
    ud = np.asarray(model.u[1].dof)
    print("DIAG u dofs: max %g ; exact range at nodes %g" % (abs(ud).max(), abs(u_f(*(nodes[:, i] for i in range(3)))).max()))
