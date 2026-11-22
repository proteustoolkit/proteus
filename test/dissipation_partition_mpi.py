#!/usr/bin/env python
"""The Dissipation residual at one state, saved per rank; run under mpiexec by test_rank_interface_flux.py.

mprans Dissipation is a C++ kernel with its own exterior loop. Its open-boundary
branch (no Dirichlet value, no flux condition) is one-sided: outflow gets
u b.n, inflow gets none. The faces a rank shares with its neighbour are
such faces when the partition has no overlap, so the ranks' terms do not
cancel. FluxBoundaryConditions specifies a zero flux there. NOZERO=1 turns
that off, as a control.

The flow and dissipation are stand-ins (constant velocity (1, 1), constant
velocity gradients and dissipation), not a RANS2P solution: the exterior
flux sees only the velocity. Writes res_rank<r>_of<n>.npz in the working directory.
"""
import os
from proteus import Comm
comm = Comm.init()
import numpy as np
from proteus import (Domain, MeshTools, NumericalSolution, StepControl, TimeIntegration, default_s, defaults)
from proteus.FemTools import C0_AffineLinearOnSimplexWithNodalBasis
from proteus.iproteus import opts
from proteus.kernels import Dissipation as NCLS
from proteus.NonlinearSolvers import Newton
from proteus.LinearAlgebraTools import SparseMatrix
from proteus.LinearSolvers import KSP_petsc4py
from proteus.Quadrature import SimplexGaussQuadrature

nn = 21
domain = Domain.RectangularDomain(L=(1.0, 1.0), x=(0.0, 0.0), name="box")
o = domain.MeshOptions
o.nnx = o.nny = o.nn = nn
o.triangleFlag = 0
o.parallelPartitioningType = MeshTools.MeshParallelPartitioningTypes.element
o.nLayersOfOverlapForParallel = 0

RD = os.environ.get("RD", "same")

if os.environ.get("NOZERO"):
    from proteus import FemTools
    FemTools.FluxBoundaryConditions._noAdvectiveFluxOnRankInterfaces = lambda self, *a, **k: None

class Coef(NCLS.Coefficients):
    rho_s = 998.2
    def attachModels(self, modelList):
        self.model = modelList[self.modelIndex]
        self.u_old_dof = np.copy(self.model.u[0].dof)
        self.velocity_dof_u = np.ones(self.model.u[0].dof.shape, 'd')
        self.velocity_dof_v = np.ones(self.model.u[0].dof.shape, 'd')
        self.velocity_dof_w = self.velocity_dof_v.copy()
        self.rho_s = self.rho_0
    def initializeElementQuadrature(self, t, cq):
        sh = cq[('grad(u)', 0)].shape
        self.q_phi = -np.ones(cq[('u', 0)].shape, 'd')
        self.q_v = np.ones(sh, 'd'); self.vs = self.q_v
        self.q_grad_u = np.ones(sh, 'd'); self.q_grad_v = 0.5*np.ones(sh, 'd')
        self.q_grad_w = self.q_grad_v.copy()
        self.q_porosity = np.ones(cq[('u', 0)].shape, 'd')
        self.q_kappa = np.full(cq[('u', 0)].shape, self.default_kappa)
        self.q_grad_kappa = np.zeros(sh, 'd')
        self.q_vos = cq[('u', 0)]; self.grad_vos = cq[('u', 0)]
    def initializeElementBoundaryQuadrature(self, t, cebq, cebq_global):
        self.ebq_v = np.ones(cebq[('grad(u)', 0)].shape, 'd')
    def initializeGlobalExteriorElementBoundaryQuadrature(self, t, cebqe):
        sh = cebqe[('grad(u)', 0)].shape
        self.ebqe_phi = -np.ones(cebqe[('u', 0)].shape, 'd')
        self.ebqe_v = np.ones(sh, 'd'); self.ebqe_vs = self.ebqe_v
        self.ebqe_grad_u = np.ones(sh, 'd'); self.ebqe_grad_v = 0.5*np.ones(sh, 'd')
        self.ebqe_grad_w = self.ebqe_grad_v.copy()
        self.ebqe_porosity = np.ones(sh, 'd')
        self.ebqe_kappa = np.full(cebqe[('u', 0)].shape, self.default_kappa)
        self.ebqe_vos = cebqe[('u', 0)]; self.ebqe_grad_vos = cebqe[('u', 0)]

class Init(object):
    def uOfXT(self, x, t):
        return np.exp(-(((x[0]-0.4)/0.2)**2 + ((x[1]-0.4)/0.2)**2))

p = defaults.Physics_base(nd=2, name="dissipation_res")
p.domain = domain
p.LevelModelType = NCLS.LevelModel
p.coefficients = Coef(V_model=0, ME_model=0, nd=2, g=np.array([0.0, -9.8, 0.0]))
p.initialConditions = {0: Init()}
p.dirichletConditions = {0: lambda x, flag: None}
p.advectiveFluxBoundaryConditions = {0: lambda x, flag: None}
p.fluxBoundaryConditions = {0: 'outFlow'}
p.diffusiveFluxBoundaryConditions = {0: {}}
n = defaults.Numerics_base()
n.timeIntegration = TimeIntegration.BackwardEuler
n.stepController = StepControl.FixedStep
n.DT = 0.02
n.femSpaces = {0: C0_AffineLinearOnSimplexWithNodalBasis}
n.elementQuadrature = SimplexGaussQuadrature(2, 3)
n.elementBoundaryQuadrature = SimplexGaussQuadrature(1, 3)
n.multilevelNonlinearSolver = n.levelNonlinearSolver = Newton
n.fullNewtonFlag = True
n.numericalFluxType = NCLS.NumericalFlux
n.shockCapturing = NCLS.ShockCapturing(p.coefficients, 2, shockCapturingFactor=0.0, lag=False)
n.subgridError = NCLS.SubgridError(p.coefficients, 2)
n.matrix = SparseMatrix
n.multilevelLinearSolver = n.levelLinearSolver = KSP_petsc4py
n.linear_solver_options_prefix = 'dissipation_'
n.linearSolverConvergenceTest = 'r-true'
n.tolFac = 0.0
n.nl_atol_res = 1e-12
n.l_atol_res = 1e-13
n.maxNonlinearIts = 30
n.parallelPartitioningType = o.parallelPartitioningType
n.nLayersOfOverlapForParallel = o.nLayersOfOverlapForParallel
so = defaults.System_base(name="dissipation_res", pnList=[(p, n)], sList=[default_s])
so.tnList = [0.0, n.DT]
ns = NumericalSolution.NS_base(so, [p], [n], so.sList, opts)
m = ns.modelList[0].levelModelList[-1]
m.coefficients.attachModels([m])
xy = m.mesh.nodeArray[:, :2]
f = np.exp(-(((xy[:,0]-0.4)/0.2)**2 + ((xy[:,1]-0.4)/0.2)**2))
m.u[0].dof[:] = f
m.timeIntegration.m_last[0][:] = m.q[('m', 0)]
free = np.zeros(m.dim); m.setFreeDOF(free)
if False:
    X = m.ebqe['x']
    m.coefficients.ebqe_rd_u = 0.7*np.exp(-(((X[..., 0]-0.45)/0.2)**2 + ((X[..., 1]-0.4)/0.2)**2)) + 0.05
r = np.zeros(m.dim)
m.getResidual(free, r); m.getResidual(free, r)
rn = np.zeros(len(xy))
g = m.dirichletConditions[0].global2freeGlobal_global_dofs; fr = m.dirichletConditions[0].global2freeGlobal_free_dofs
rn[g] = r[fr]
np.savez("res_rank%d_of%d.npz" % (comm.rank(), comm.size()), xy=xy, r=rn)
mesh = m.mesh
print("MESH rank=%d owned=%d layers=%s" % (comm.rank(), mesh.nElements_owned, getattr(mesh, 'nLayersOfOverlap', None)), flush=True)
