#!/usr/bin/env python
"""RANS2P single-phase step flow on an element partition without overlap.

Run by ``test_rank_interface_flux.py`` under ``mpiexec``. Prints the sums over
owned nodes of p, u and v squared and unsummed, which a serial run must match.
The system is test/solver_tests/import_modules/step2d, solved with a direct
factorization wrapped in a tightly converged GMRES, so the comparison is not limited by a solver tolerance. The mesh is
partitioned by the domain's MeshOptions, which also decides how the residual is
assembled, so both are set there.
"""
import os
import sys

from proteus import Comm

comm = Comm.init()

import numpy as np
from petsc4py import PETSc

from proteus import Context, MeshTools, NumericalSolution, default_s, defaults, runtime

MODULES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solver_tests", "import_modules")
sys.path.insert(0, MODULES)
Context.contextOptionsString = "he=0.1 schur_solver='petsc_ASM'"
so = defaults.load_system("step2d_so", MODULES)
p = defaults.load_physics("twp_navier_stokes_step2d_p", MODULES)
n = defaults.load_numerics("twp_navier_stokes_step2d_n", MODULES)
n.linearSmoother = None
# the deck stops Newton at max(1e-10, 0.001 he^2) = 1e-5 residual for he=0.1: a serial and a parallel run each stop
# there, and their answers can differ by about that much (fleet, 2026-11-09: p2 10.396641 vs 10.396764, 1.2e-5),
# which the test then compares to 1e-6. Converge Newton instead, so the comparison is about the discretization.
n.nl_atol_res = 1.0e-12
n.l_atol_res = 1.0e-14
n.tolFac = 0.0
options = PETSc.Options()
# LU as a preconditioner of a converged Krylov solve, not as the solve (preonly): SuperLU_DIST pivots statically, so on
# this saddle-point Jacobian a parallel LU is not always an exact solve, and on spark2 (PETSc 3.26.0, 2 ranks) Newton
# stalled at 6e-5 (-0.4% per iteration) with every linear solve preonly, while one rank converged quadratically to 4e-15.
# The Krylov method corrects an inexact factorization, so the comparison is about the discretization, not the solver.
options.setValue("rans2p_ksp_type", "gmres")
options.setValue("rans2p_ksp_rtol", 1.0e-13)
options.setValue("rans2p_ksp_atol", 1.0e-15)
options.setValue("rans2p_ksp_max_it", 50)
options.setValue("rans2p_pc_type", "lu")
options.setValue("rans2p_pc_factor_mat_solver_type", "superlu_dist" if comm.size() > 1 else "superlu")
p.name = so.name = "step2d_part"
so.sList = [default_s]
mesh_options = p.domain.MeshOptions
mesh_options.parallelPartitioningType = MeshTools.MeshParallelPartitioningTypes.element
mesh_options.nLayersOfOverlapForParallel = int(os.environ.get("NLAYERS", "0"))
if os.environ.get("DIAG"):
    from proteus import Profiling
    Profiling.logLevel = 7
    Profiling.verbose = True
ns = NumericalSolution.NS_base(so, [p], [n], so.sList, runtime.RunOptions())
ns.calculateSolution("step2d_part")
model = ns.modelList[0].levelModelList[-1]
mesh = model.mesh
owned = mesh.nNodes_owned
from mpi4py import MPI
c = comm.comm.tompi4py()
result = {}
for i, name in enumerate(("p", "u", "v")):
    dof = model.u[i].dof[:owned]
    result[name + "2"] = c.allreduce(float((dof ** 2).sum()), op=MPI.SUM)
    result[name + "s"] = c.allreduce(float(dof.sum()), op=MPI.SUM)
interfaces = int((mesh.elementBoundaryMaterialTypes[mesh.exteriorElementBoundariesArray] == -1).sum())
print("RANK rank=%d interfaces=%d" % (comm.rank(), interfaces), flush=True)
if comm.rank() == 0:
    print("RESULT ranks=%d " % comm.size() + " ".join("%s=%.17e" % kv for kv in result.items()), flush=True)
