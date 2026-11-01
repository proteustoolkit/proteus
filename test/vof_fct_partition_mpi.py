#!/usr/bin/env python
"""VOF edge-based (STAB=3 smoothness indicator, the default, or STAB=2 entropy viscosity) with FCT, a blob advected across a rank cut.

Run by test_vof_partition_independence.py under mpiexec in an empty directory; leaves u.csv (owned nodes,
sorted by x, y). Same blob and velocity as vof_partition_mpi.py, explicit lumped-mass RKEV, node partition with one
overlap layer. NOFIX=1 turns off the owner-to-ghost scatters, as a control. GHOSTCFL=1 puts garbage in the edge based cfl of the ghost
dofs; OLDMAX=1 makes the time step read it again (a control).
"""
import os

from proteus import Comm

comm = Comm.init()

import numpy as np

from proteus import (Domain, MeshTools, NumericalSolution, StepControl, TimeIntegration,
                     default_s, default_so, defaults)
from proteus.FemTools import C0_AffineLinearOnSimplexWithNodalBasis
from proteus.iproteus import opts
import importlib
VOF = importlib.import_module("proteus.mprans." + os.environ.get("MODEL", "VOF"))
from proteus.NonlinearSolvers import Newton, ExplicitLumpedMassMatrix
from proteus.LinearAlgebraTools import SparseMatrix
from proteus.LinearSolvers import KSP_petsc4py
from proteus.Quadrature import SimplexGaussQuadrature

nn = int(os.environ.get("NN", "21"))
domain = Domain.RectangularDomain(L=(1.0, 1.0), x=(0.0, 0.0), name="box")
options = domain.MeshOptions
options.nnx = options.nny = options.nn = nn
options.triangleFlag = 0
options.parallelPartitioningType = getattr(MeshTools.MeshParallelPartitioningTypes,
                                           "node")
options.nLayersOfOverlapForParallel = int(os.environ.get("NLAYERS", "1"))


class _Flow(object):
    """What VOF reads from a flow model; there is none here and no solid."""


class Coefficients(VOF.Coefficients):
    def attachModels(self, modelList):
        VOF.Coefficients.attachModels(self, modelList)
        m = modelList[self.modelIndex]
        flow = _Flow()
        flow.ghost_penalty_constant = 0.0
        flow.phi_s = np.full(m.mesh.nodeArray.shape[0], 1.0e10)
        flow.ebqe_phi_s = np.full(m.ebqe[('u', 0)].shape, 1.0e10)
        flow.q_phi_solid = np.full(m.q[('u', 0)].shape, 1.0e10)
        self.flowCoefficients = flow


class Blob(object):
    def uOfXT(self, x, t):
        return np.exp(-(((x[0] - 0.4) / 0.12) ** 2 + ((x[1] - 0.4) / 0.12) ** 2))


def inflow(x, flag):
    if x[0] <= 1.0e-8 or x[1] <= 1.0e-8:
        return lambda x, t: 0.0


p = defaults.Physics_base(nd=2, name="vof_part")
p.domain = domain
p.LevelModelType = VOF.LevelModel
p.coefficients = Coefficients(LS_model=None, V_model=None, RD_model=None, ME_model=0,
                              checkMass=False, epsFact=0.0, STABILIZATION_TYPE=int(os.environ.get('STAB', '3')),
                              LUMPED_MASS_MATRIX=True, FCT=True)
p.initialConditions = {0: Blob()}
p.dirichletConditions = {0: inflow}
p.advectiveFluxBoundaryConditions = {0: lambda x, flag: None}
p.diffusiveFluxBoundaryConditions = {0: {}}
p.fluxBoundaryConditions = {0: 'outFlow'}

n = defaults.Numerics_base()
n.timeIntegration = VOF.RKEV
n.stepController = StepControl.Min_dt_controller
n.runCFL = 0.1
n.timeOrder = 1
n.nStagesTime = 1
n.DT = 0.0025
n.femSpaces = {0: C0_AffineLinearOnSimplexWithNodalBasis}
n.elementQuadrature = SimplexGaussQuadrature(2, 3)
n.elementBoundaryQuadrature = SimplexGaussQuadrature(1, 3)
n.multilevelNonlinearSolver = Newton
n.levelNonlinearSolver = ExplicitLumpedMassMatrix
n.fullNewtonFlag = False
n.updateJacobian = False
n.numericalFluxType = VOF.NumericalFlux
n.shockCapturing = VOF.ShockCapturing(p.coefficients, 2, shockCapturingFactor=0.0, lag=False)
n.matrix = SparseMatrix
n.multilevelLinearSolver = n.levelLinearSolver = KSP_petsc4py
n.linear_solver_options_prefix = 'vof_'
n.linearSolverConvergenceTest = 'r-true'
n.tolFac = 0.0
n.nl_atol_res = 1.0e-12
n.l_atol_res = 1.0e-13
n.linTolFac = 0.0
n.maxNonlinearIts = 30
n.parallelPartitioningType = options.parallelPartitioningType
n.nLayersOfOverlapForParallel = options.nLayersOfOverlapForParallel

so = defaults.System_base(name="vof_part", pnList=[(p, n)], sList=[default_s])
so.tnList = [0.0] + [n.DT * i for i in range(10, 81, 10)]
ns = NumericalSolution.NS_base(so, [p], [n], so.sList, opts)
if os.environ.get("NOFIX"):
    VOF.LevelModel.scatterDOFArray = lambda self, a: None
if os.environ.get("GHOSTCFL"):
    # garbage in the cfl of the ghost dofs, which the time step must not read
    _getResidual = VOF.LevelModel.getResidual
    def getResidual(self, u, r):
        out = _getResidual(self, u, r)
        self.edge_based_cfl[self.mesh.nNodes_owned:] = 200.0
        return out
    VOF.LevelModel.getResidual = getResidual
if os.environ.get("OLDMAX"):
    VOF.RKEV.ownedMaxCFL = lambda self: self.cfl.max()
ns.calculateSolution("vof_part")
model = ns.modelList[0].levelModelList[-1]
mesh = model.mesh
nOwned = mesh.nNodes_owned
c = comm.comm.tompi4py()
g = c.gather(np.column_stack([mesh.nodeArray[:nOwned, :2], model.u[0].dof[:nOwned]]), root=0)
if comm.rank() == 0:
    a = np.vstack(g)
    np.savetxt("u.csv", a[np.lexsort((a[:, 1], a[:, 0]))], delimiter=",", header="x,y,u")
