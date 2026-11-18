#!/usr/bin/env python
"""TADR (SUPG, backward Euler to steady state) with a different boundary kind on each wall (C12): strong on the left,
weak on the right, declared open ('none') on the top and bottom. Run by test_mixed_bc_mms.py, serially and under mpiexec.

    b u_x = D u_xx on the box (0,1) x (0,H),   b = (1, 0),   u(0) = 1,  u(1) = 0,

whose exact solution u = (exp(b x/D) - exp(b/D)) / (1 - exp(b/D)) depends on x only (a boundary layer of thickness D/b at the
outflow x = 1): the top and bottom walls carry no flux (b.n = 0, u_y = 0), which is what a face declared open gives.

Environment: KINDS (left=strong,right=weak,...: a wall left out has no condition), DECLARE (0: a plain setter, the kinds
are not read), ALL (strong: coefficients.forceStrongConditions, every Dirichlet face strong: the baseline the kernel's scalar
flag gave), NN (cells in x), D, PART_TYPE. Rank 0 prints one line ``RESULT {json}``.
"""
import json
import os

from proteus import runtime
comm = runtime.start()
from proteus import Profiling
Profiling.logLevel = int(os.environ.get('LOGLEVEL', '1'))

import numpy as np
from petsc4py import PETSc
from proteus import (BoundaryKinds as bk, Domain, FemTools, MeshTools, NumericalFlux, NumericalSolution, Quadrature,
                     default_s, defaults)
from proteus.LinearAlgebraTools import SparseMatrix
from proteus.LinearSolvers import KSP_petsc4py
from proteus.NonlinearSolvers import Newton
from proteus.TimeIntegration import BackwardEuler
from proteus.StepControl import Min_dt_controller
from proteus.mprans import TADR

nx = int(os.environ.get("NN", "41"))
D = float(os.environ.get("D", "0.05"))
H = 0.2
kinds = dict(kv.split("=") for kv in os.environ.get("KINDS", "left=strong,right=weak,top=none,bottom=none").split(",") if kv)
declare = os.environ.get("DECLARE", "1") == "1"
allStrong = os.environ.get("ALL") == "strong"
part = {"element": MeshTools.MeshParallelPartitioningTypes.element,
        "node": MeshTools.MeshParallelPartitioningTypes.node}[os.environ.get("PART_TYPE", "node")]


def u_exact(x):
    return (np.exp(x / D) - np.exp(1.0 / D)) / (1.0 - np.exp(1.0 / D))


domain = Domain.RectangularDomain(L=(1.0, H), x=(0.0, 0.0), name="box")
tag = {v: k for k, v in domain.boundaryTags.items()}
data = {"left": 1.0, "right": 0.0}
_kind = {"strong": bk.strong, "weak": bk.weak}


def dirichlet(x, flag):
    name = tag.get(flag)
    k = kinds.get(name)
    if k is None:
        return None
    if k == "none":
        return bk.OPEN if declare else None
    g = (lambda x, t, v=data[name]: v)
    if k == "bare" or not declare:
        return g
    return _kind[k](g)


o = domain.MeshOptions
o.nnx = nx + 1
o.nny = max(3, int(round(nx * H)) + 1)
o.nnz = 1
o.triangleFlag = 0
o.parallelPartitioningType = part
o.nLayersOfOverlapForParallel = 1


class Coefficients(TADR.Coefficients):
    def attachModels(self, modelList):
        self.model = modelList[0]
        self.rdModel = self.model
        self.q_v = np.zeros(self.model.q[("grad(u)", 0)].shape, "d")
        self.ebqe_v = np.zeros(self.model.ebqe[("grad(u)", 0)].shape, "d")
        self.ebqe_phi = np.zeros(self.model.ebqe[("u", 0)].shape, "d")


p = defaults.Physics_base(nd=2, name="tadr_mixed_bc")
p.domain = domain
p.LevelModelType = TADR.LevelModel
p.coefficients = Coefficients(alpha_L=0.0, alpha_T=0.0, Dm=D, checkMass=False, FCT=False, LUMPED_MASS_MATRIX=False,
                              STABILIZATION_TYPE=0, ENTROPY_TYPE=1, cE=0.1, cK=1.0, physicalDiffusion=0.0,
                              forceStrongConditions=allStrong)
p.coefficients.variableNames = ["u"]
p.initialConditions = {0: type("Zero", (), {"uOfXT": lambda self, x, t: 0.0})()}
p.analyticalSolution = {0: type("Exact", (), {"uOfXT": lambda self, x, t: float(u_exact(x[0]))})()}
p.dirichletConditions = {0: bk.declare(dirichlet) if declare else dirichlet}
p.advectiveFluxBoundaryConditions = {0: lambda x, flag: None}
p.diffusiveFluxBoundaryConditions = {0: {}}
p.fluxBoundaryConditions = {0: "outFlow"}

n = defaults.Numerics_base()
n.timeIntegration = BackwardEuler
n.stepController = Min_dt_controller
n.runCFL = 0.5
n.femSpaces = {0: FemTools.C0_AffineLinearOnSimplexWithNodalBasis}
n.elementQuadrature = Quadrature.SimplexGaussQuadrature(2, 3)
n.elementBoundaryQuadrature = Quadrature.SimplexGaussQuadrature(1, 3)
n.multilevelNonlinearSolver = n.levelNonlinearSolver = Newton
n.fullNewtonFlag = True
n.nl_atol_res = 1.0e-9
n.tolFac = 0.0
n.maxNonlinearIts = 150   # a weak Dirichlet face makes TADR's Newton converge only linearly (0.74 per iteration), see the commit
n.matrix = SparseMatrix
n.numericalFluxType = NumericalFlux.Advection_DiagonalUpwind_IIPG_exterior
n.multilevelLinearSolver = n.levelLinearSolver = KSP_petsc4py
n.linearSmoother = None
n.linearSolverConvergenceTest = "r-true"
n.l_atol_res = 1.0e-12
n.subgridError = TADR.SubgridError(p.coefficients, 2)
n.shockCapturing = TADR.ShockCapturing(p.coefficients, 2, shockCapturingFactor=0.0, lag=True)
n.velocityFieldAsFunction = {0: lambda X, t: 1.0, 1: lambda X, t: 0.0}
options = PETSc.Options()
options.setValue("ksp_type", "gmres")
options.setValue("ksp_rtol", 1.0e-14)
options.setValue("ksp_atol", 1.0e-15)
options.setValue("pc_type", "lu")
options.setValue("pc_factor_mat_solver_type", os.environ.get("LU", "superlu_dist" if comm.size() > 1 else "superlu"))
n.parallelPartitioningType = part
n.nLayersOfOverlapForParallel = 1

so = defaults.System_base(name="tadr_mixed_bc", pnList=[(p, n)], sList=[default_s])
so.tnList = [0.0] + [2.0 ** k for k in range(0, 13)]   # implicit steps of growing size to the steady state
ns = NumericalSolution.NS_base(so, [p], [n], so.sList, runtime.RunOptions())
ns.calculateSolution("tadr_mixed_bc")

model = ns.modelList[0].levelModelList[-1]
mesh = model.mesh
dof = np.asarray(model.u[0].dof)
X = np.asarray(mesh.nodeArray)
nOwned = mesh.nNodes_owned
err = dof[:nOwned] - u_exact(X[:nOwned, 0])
sq, mx = float(np.sum(err ** 2)), float(np.abs(err).max())
nf = model.numericalFlux
force = getattr(model, "dirichletConditionsForceDOF", None)
nForced = len(force.DOFBoundaryConditionsDict) if force is not None else 0
weak = int(nf.isDOFBoundary[0].sum())
strongPts = int(nf.isStrongBoundary[0].sum())
openPts = int(nf.isOpenBoundary[0].sum())
from mpi4py import MPI
mc = comm.comm.tompi4py()
sq, nn_ = mc.allreduce(sq, MPI.SUM), mc.allreduce(int(nOwned), MPI.SUM)
mx = mc.allreduce(mx, MPI.MAX)
weak, strongPts, openPts = (mc.allreduce(v, MPI.SUM) for v in (weak, strongPts, openPts))
if comm.rank() == 0:
    print("RESULT " + json.dumps(dict(nx=nx, D=D, kinds=kinds, declare=declare, all=os.environ.get("ALL"), ranks=comm.size(),
                                      rms=(sq / nn_) ** 0.5, linf=mx, weakPts=weak, strongPts=strongPts, openPts=openPts,
                                      forcedDofsOnRank0=nForced, forceStrongConditions=bool(model.forceStrongConditions),
                                      uMin=float(dof.min()), uMax=float(dof.max()))), flush=True)
