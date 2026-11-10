#!/usr/bin/env python
"""Poisson on the unit square with strong Dirichlet dofs chosen by boundary FLAG (left face only).

Run by ``test_strong_dirichlet_authority.py`` under ``mpiexec``. u = x^2 + y^2 on the left face (tag 4 of
the rectangular domain's boundaryTags), natural elsewhere; every other face carries the exact flux. KSP
(LU, superlu_dist in parallel) on the kept-DOF system (StrongDirichletFactory(keepDOFs=True)). TRAP=1 puts the
left-face element and the other element of the corner (0, 1) and (0, 0) on different ranks.
"""
import os
import sys

from proteus import Comm

comm = Comm.init()

import numpy as np
from petsc4py import PETSc

from proteus import (Context, Domain, MeshTools, NumericalSolution, default_s, defaults, runtime, FemTools,
                     Quadrature, NumericalFlux, LinearSolvers, NonlinearSolvers, TimeIntegration, StepControl,
                     SplitOperator, TransportCoefficients, Transport, superluWrappers)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HE = float(os.environ.get("HE", "0.1"))
name = "sd_poisson"
rect = Domain.RectangularDomain(L=(1.0, 1.0), x=(0.0, 0.0), name=name)
rect.writePoly(name)
domain = Domain.PlanarStraightLineGraphDomain(name)
domain.boundaryTags = rect.boundaryTags
LEFT = rect.boundaryTags["left"]
if os.environ.get("TRAP"):
    from strong_dirichlet_trap import install
    install((LEFT,), ([0.0, 0.0], [0.0, 1.0]), report=lambda n: print("PROBE trapped corners", n, flush=True))
mo = MeshTools.getMeshOptions(domain)
mo.he = HE
mo.parallelPartitioningType = MeshTools.MeshParallelPartitioningTypes.element
mo.nLayersOfOverlapForParallel = 0
mo.genMesh = True


def uex(x):
    return x[0] ** 2 + x[1] ** 2


p = defaults.Physics_base()
p.name = name
p.nc = 1
p.nd = 2
p.domain = domain
p.L = (1.0, 1.0)
p.T = 1.0
p.coefficients = TransportCoefficients.PoissonEquationCoefficients(
    {0: lambda x: np.array([1.0, 0.0, 0.0, 1.0])}, {0: lambda x: -4.0}, nc=1, nd=2)
p.dirichletConditions = {0: lambda x, flag: (lambda x, t: uex(x)) if flag == LEFT else None}
p.advectiveFluxBoundaryConditions = {}
p.diffusiveFluxBoundaryConditions = {0: {0: lambda x, flag: None}}
p.initialConditions = {0: type("Z", (), {"uOfXT": lambda self, x, t: 0.0})()}
p.analyticalSolution = {}

n = defaults.Numerics_base()
n.timeIntegration = TimeIntegration.NoIntegration
n.stepController = StepControl.Min_dt_controller
n.femSpaces = {0: FemTools.C0_AffineLinearOnSimplexWithNodalBasis}
n.elementQuadrature = Quadrature.SimplexGaussQuadrature(2, 4)
n.elementBoundaryQuadrature = Quadrature.SimplexGaussQuadrature(1, 4)
n.numericalFluxType = NumericalFlux.StrongDirichletFactory({}, keepDOFs=True)
n.nnx = n.nny = None
n.multilevelNonlinearSolver = NonlinearSolvers.NLNI
n.levelNonlinearSolver = NonlinearSolvers.Newton
n.fullNewtonFlag = True
n.maxNonlinearIts = 5
n.tolFac = 0.0
n.nl_atol_res = 1.0e-12
n.linTolFac = 0.0
n.l_atol_res = 1.0e-14
n.matrix = superluWrappers.SparseMatrix
n.multilevelLinearSolver = LinearSolvers.KSP_petsc4py
n.levelLinearSolver = LinearSolvers.KSP_petsc4py
n.linearSmoother = None
n.linearSolverConvergenceTest = "r-true"
n.parallelPartitioningType = MeshTools.MeshParallelPartitioningTypes.element
n.nLayersOfOverlapForParallel = 0
n.nLevels = 1
o = PETSc.Options()
o.setValue("ksp_type", "gmres")
o.setValue("ksp_rtol", 1.0e-13)
o.setValue("ksp_atol", 1.0e-15)
o.setValue("pc_type", "lu")
o.setValue("pc_factor_mat_solver_type", os.environ.get("LU", "superlu_dist" if comm.size() > 1 else "superlu"))
so = defaults.System_base(name=name, pnList=[(p, n)], sList=[default_s])
so.tnList = [0.0, 1.0]
so.systemStepControllerType = SplitOperator.Sequential_FixedStep
so.dt_system_fixed = 1.0
ns = NumericalSolution.NS_base(so, [p], [n], so.sList, runtime.RunOptions())
ns.calculateSolution(name)
m = ns.modelList[0]
model = m.levelModelList[-1]
mesh = model.mesh
owned = mesh.nNodes_owned
from mpi4py import MPI
c = comm.comm.tompi4py()
dof = model.u[0].dof[:owned]
result = {"s": c.allreduce(float(dof.sum()), op=MPI.SUM), "s2": c.allreduce(float((dof ** 2).sum()), op=MPI.SUM),
          "newton": float(m.solver.solverList[-1].its)}
if comm.rank() == 0:
    print("RESULT ranks=%d " % comm.size() + " ".join("%s=%.17e" % kv for kv in result.items()), flush=True)
