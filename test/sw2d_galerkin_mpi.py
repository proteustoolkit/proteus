#!/usr/bin/env python
"""SW2D (the Galerkin shallow water kernel, nothing else in the tree builds it) on a small dam break, for test_sw2d_galerkin.py.

Flat bottom (the coefficients leave the bathymetry at zero), a 1.0 / 0.5 depth step across x = 0.5 of the unit
square, no flow initially, no flow through the walls (u = v = 0 on the whole boundary), BackwardEuler with Newton,
a few fixed steps. SUPG=1 sets ``options.try_supg_stabilization`` (calculateResidual_supg / calculateJacobian_supg),
otherwise the plain Galerkin residual and Jacobian. Run in an empty directory (serially or under mpiexec); rank 0 leaves
sw2d.npz: the depth h, hu, hv (the three dofs of the owned nodes, sorted by x, y), the nodes' x, y, and the names of the residual and Jacobian entry points the model bound.
"""
import os
from proteus import Comm
comm = Comm.init()
import numpy as np
from proteus import (Domain, MeshTools, NumericalSolution, StepControl, TimeIntegration, default_s, defaults)
from proteus.FemTools import C0_AffineLinearOnSimplexWithNodalBasis
from proteus.iproteus import opts
from proteus.kernels import SW2D
from proteus.NonlinearSolvers import Newton
from proteus.NumericalFlux import Advection_DiagonalUpwind_Diffusion_IIPG_exterior
from proteus.LinearAlgebraTools import SparseMatrix
from proteus.LinearSolvers import LU, KSP_petsc4py
from proteus.Quadrature import SimplexGaussQuadrature

import proteus.Transport
# SW2D.py uses the bare names below and imports none of them (the star import of cSW2D does not provide them), so
# neither its Coefficients nor its LevelModel can be built as written; the other kernels take them from
# proteus.Transport. They are put in the kernel's namespace here; the report is in HANDOFF_2026-11-23_c49-r3-done.md.
# Drop this loop when SW2D.py imports them.
for _name in ("TC_base", "NonlinearEquation", "Quadrature", "logEvent", "cfemIntegrals", "memory", "Comm",
              "DOFBoundaryConditions", "FluxBoundaryConditions"):
    if _name not in SW2D.__dict__:
        setattr(SW2D, _name, getattr(proteus.Transport, _name))
SUPG = bool(int(os.environ.get("SUPG", "0")))
NN = int(os.environ.get("NN", "11"))
NSTEPS = int(os.environ.get("NSTEPS", "3"))
DT = float(os.environ.get("DT", "0.005"))

domain = Domain.RectangularDomain(L=(1.0, 1.0), x=(0.0, 0.0), name="box")
o = domain.MeshOptions
o.nnx = o.nny = o.nn = NN
o.triangleFlag = 0
o.parallelPartitioningType = MeshTools.MeshParallelPartitioningTypes.node
o.nLayersOfOverlapForParallel = 1


class Init(object):
    def __init__(self, ci):
        self.ci = ci

    def uOfXT(self, x, t):
        return (1.0 if x[0] < 0.5 else 0.5) if self.ci == 0 else 0.0


p = defaults.Physics_base(nd=2, name="sw2d")
p.domain = domain
p.LevelModelType = SW2D.LevelModel
p.coefficients = SW2D.Coefficients(g=9.81, nu=1.0e-3, nd=2)
p.initialConditions = {0: Init(0), 1: Init(1), 2: Init(2)}
p.dirichletConditions = {0: lambda x, flag: None, 1: lambda x, flag: lambda x, t: 0.0, 2: lambda x, flag: lambda x, t: 0.0}
p.advectiveFluxBoundaryConditions = {}
p.diffusiveFluxBoundaryConditions = {0: {}, 1: {}, 2: {}}
p.fluxBoundaryConditions = {0: 'outFlow', 1: 'outFlow', 2: 'outFlow'}
n = defaults.Numerics_base()
n.timeIntegration = TimeIntegration.BackwardEuler
n.stepController = StepControl.FixedStep
n.DT = DT
n.femSpaces = {0: C0_AffineLinearOnSimplexWithNodalBasis, 1: C0_AffineLinearOnSimplexWithNodalBasis,
               2: C0_AffineLinearOnSimplexWithNodalBasis}
n.elementQuadrature = SimplexGaussQuadrature(2, 3)
n.elementBoundaryQuadrature = SimplexGaussQuadrature(1, 3)
n.multilevelNonlinearSolver = n.levelNonlinearSolver = Newton
n.fullNewtonFlag = True
n.numericalFluxType = SW2D.NumericalFlux
n.matrix = SparseMatrix
n.multilevelLinearSolver = n.levelLinearSolver = LU if comm.size() == 1 else KSP_petsc4py
n.linear_solver_options_prefix = 'sw2d_'
n.linearSolverConvergenceTest = 'r-true'
n.tolFac = 0.0
n.nl_atol_res = 1e-10
n.l_atol_res = 1e-12
n.maxNonlinearIts = 30
n.try_supg_stabilization = SUPG
n.subgridError = SW2D.SubgridError(p.coefficients, 2)
n.shockCapturing = SW2D.ShockCapturing(p.coefficients, 2, shockCapturingFactor=0.25, lag=False)
n.parallelPartitioningType = o.parallelPartitioningType
n.nLayersOfOverlapForParallel = o.nLayersOfOverlapForParallel
so = defaults.System_base(name="sw2d", pnList=[(p, n)], sList=[default_s])
so.tnList = [0.0] + [DT * (i + 1) for i in range(NSTEPS)]
ns = NumericalSolution.NS_base(so, [p], [n], so.sList, opts)
m = ns.modelList[0].levelModelList[-1]
nOwned = m.mesh.nNodes_owned


ns.calculateSolution("sw2d")
cols = [m.mesh.nodeArray[:nOwned, :2]] + [m.u[i].dof[:nOwned, None] for i in range(3)]
mpi = comm.comm.tompi4py()
gathered = mpi.gather(np.hstack(cols), root=0)
if mpi.rank == 0:
    a = np.vstack(gathered)
    a = a[np.lexsort((a[:, 1], a[:, 0]))]
    np.savez("sw2d.npz", x=a[:, 0], y=a[:, 1], h=a[:, 2], hu=a[:, 3], hv=a[:, 4], residual=m.calculateResidual.__name__,
             jacobian=m.calculateJacobian.__name__, residual_calls=m.nonlinear_function_evaluations)
