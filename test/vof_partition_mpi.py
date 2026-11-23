#!/usr/bin/env python
"""A VOF (or, MODEL=VOF3P or VOS3P, that model) blob advected across a rank cut; prints its mass and L2 norm.

Run by ``test_rank_interface_flux.py`` under ``mpiexec`` to compare a partitioned
run against a serial one. mprans VOF is a C++ kernel with its own exterior loop,
and the advective flux there is one-sided on a face with no boundary condition:
outflow gets ``u b.n``, inflow gets none. The faces a rank shares with its
neighbour are such faces when the partition has no overlap, so the two ranks'
terms do not cancel and mass leaves through the middle of the domain.

Constant velocity (1, 1), a Gaussian blob centred at (0.4, 0.4) on the unit
square, Dirichlet zero on the inflow sides, backward Euler with a fixed step,
P1, SUPG. PART_TYPE / NLAYERS pick the partition. The mesh is partitioned by
the domain's MeshOptions, which is also what decides how the residual is
assembled, so both are set there.
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
VOF = importlib.import_module("proteus.kernels." + os.environ.get("MODEL", "VOF"))
from proteus.NonlinearSolvers import Newton
from proteus.LinearAlgebraTools import SparseMatrix
from proteus.LinearSolvers import KSP_petsc4py
from proteus.Quadrature import SimplexGaussQuadrature

nn = int(os.environ.get("NN", "21"))
domain = Domain.RectangularDomain(L=(1.0, 1.0), x=(0.0, 0.0), name="box")
options = domain.MeshOptions
options.nnx = options.nny = options.nn = nn
options.triangleFlag = 0
options.parallelPartitioningType = getattr(MeshTools.MeshParallelPartitioningTypes,
                                           os.environ.get("PART_TYPE", "element"))
options.nLayersOfOverlapForParallel = int(os.environ.get("NLAYERS", "0"))


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
                              checkMass=False, epsFact=0.0, STABILIZATION_TYPE=0,
                              **({'FCT': False} if os.environ.get('MODEL') == 'VOS3P' else {}))
p.initialConditions = {0: Blob()}
p.dirichletConditions = {0: inflow}
p.advectiveFluxBoundaryConditions = {0: lambda x, flag: None}
p.diffusiveFluxBoundaryConditions = {0: {}}
p.fluxBoundaryConditions = {0: 'outFlow'}

n = defaults.Numerics_base()
n.timeIntegration = TimeIntegration.BackwardEuler
n.stepController = StepControl.FixedStep
n.DT = 0.02
n.femSpaces = {0: C0_AffineLinearOnSimplexWithNodalBasis}
n.elementQuadrature = SimplexGaussQuadrature(2, 3)
n.elementBoundaryQuadrature = SimplexGaussQuadrature(1, 3)
n.multilevelNonlinearSolver = n.levelNonlinearSolver = Newton
n.fullNewtonFlag = True
n.numericalFluxType = VOF.NumericalFlux
n.shockCapturing = VOF.ShockCapturing(p.coefficients, 2, shockCapturingFactor=0.0, lag=False)
n.subgridError = VOF.SubgridError(p.coefficients, 2)
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
so.tnList = [0.0] + [n.DT * i for i in range(1, 11)]
ns = NumericalSolution.NS_base(so, [p], [n], so.sList, opts)
ns.calculateSolution("vof_part")
model = ns.modelList[0].levelModelList[-1]
mesh = model.mesh
owned = mesh.nElements_owned
u = model.q[('u', 0)][:owned]
dV = model.q['dV'][:owned]
from mpi4py import MPI
c = comm.comm.tompi4py()
mass = c.allreduce(float((u * dV).sum()), op=MPI.SUM)
l2 = c.allreduce(float((u ** 2 * dV).sum()), op=MPI.SUM)
interfaces = int((mesh.elementBoundaryMaterialTypes[mesh.exteriorElementBoundariesArray] == -1).sum())
print("RANK rank=%d interfaces=%d" % (comm.rank(), interfaces), flush=True)
if comm.rank() == 0:
    print("RESULT ranks=%d mass=%.17e l2=%.17e" % (comm.size(), mass, l2), flush=True)
