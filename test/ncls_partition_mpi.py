#!/usr/bin/env python
"""The NCLS residual at one state, saved per rank; run under mpiexec by test_rank_interface_flux.py.

mprans NCLS is a C++ kernel with its own exterior loop. Where a face has no
Dirichlet value the flux is 0 for outflow and -flow*(u - rd_u) for inflow,
with rd_u the redistanced level set. That is the right treatment on a domain
boundary, but the faces a rank shares with its neighbour are exterior faces of
its subdomain when the partition has no overlap, and the neighbour's outflow
side contributes nothing to cancel the inflow term. RD=perturbed gives rd_u a
smooth field different from u, as a redistancing step does; RD=same leaves it
equal to u (no RDLS model).

Constant velocity (1, 1), a Gaussian on the unit square, P1, element
partition without overlap. Writes res_rank<r>_of<n>.npz in the working
directory.
"""
import os
from proteus import Comm
comm = Comm.init()
import numpy as np
from proteus import (Domain, MeshTools, NumericalSolution, StepControl, TimeIntegration, default_s, defaults)
from proteus.FemTools import C0_AffineLinearOnSimplexWithNodalBasis
from proteus.iproteus import opts
from proteus.kernels import NCLS
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

class Coef(NCLS.Coefficients):
    def attachModels(self, modelList):
        self.model = modelList[self.modelIndex]
        m = self.model
    def initializeElementQuadrature(self, t, cq):
        self.q_v = np.ones(cq[('grad(u)', 0)].shape, 'd')
    def initializeElementBoundaryQuadrature(self, t, cebq, cebq_global):
        self.ebq_v = np.ones(cebq[('grad(u)', 0)].shape, 'd')
    def initializeGlobalExteriorElementBoundaryQuadrature(self, t, cebqe):
        self.ebqe_v = np.ones(cebqe[('grad(u)', 0)].shape, 'd')
        self.ebqe_rd_u = cebqe[('u', 0)]

class Init(object):
    def uOfXT(self, x, t):
        return np.exp(-(((x[0]-0.4)/0.2)**2 + ((x[1]-0.4)/0.2)**2))

p = defaults.Physics_base(nd=2, name="ncls_res")
p.domain = domain
p.LevelModelType = NCLS.LevelModel
p.coefficients = Coef(V_model=None, RD_model=None, ME_model=0, checkMass=False, epsFact=0.0, STABILIZATION_TYPE=0)
p.initialConditions = {0: Init()}
p.dirichletConditions = {0: lambda x, flag: None}
p.advectiveFluxBoundaryConditions = {}
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
n.linear_solver_options_prefix = 'ncls_'
n.linearSolverConvergenceTest = 'r-true'
n.tolFac = 0.0
n.nl_atol_res = 1e-12
n.l_atol_res = 1e-13
n.maxNonlinearIts = 30
n.parallelPartitioningType = o.parallelPartitioningType
n.nLayersOfOverlapForParallel = o.nLayersOfOverlapForParallel
so = defaults.System_base(name="ncls_res", pnList=[(p, n)], sList=[default_s])
so.tnList = [0.0, n.DT]
ns = NumericalSolution.NS_base(so, [p], [n], so.sList, opts)
m = ns.modelList[0].levelModelList[-1]
xy = m.mesh.nodeArray[:, :2]
f = np.exp(-(((xy[:,0]-0.4)/0.2)**2 + ((xy[:,1]-0.4)/0.2)**2))
m.u[0].dof[:] = f
m.timeIntegration.m_last[0][:] = m.q[('m', 0)]
free = np.zeros(m.dim); m.setFreeDOF(free)
if RD == "perturbed":
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
