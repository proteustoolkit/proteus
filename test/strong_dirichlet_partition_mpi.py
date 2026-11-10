#!/usr/bin/env python
"""RANS2P step flow (strong Dirichlet) on an element partition whose cut runs through boundary nodes.

Run by ``test_strong_dirichlet_authority.py`` under ``mpiexec``. TRAP=1 gives rank k >= 1 one element
that touches a wall node only through its interior (no boundary face of that element contains the node)
and everything else to rank 0: the wall faces that make the node a strong-Dirichlet dof are all on rank 0,
so rank k holds the node as a ghost without the Dirichlet condition. Prints sums over owned nodes of p, u, v
squared and unsummed, and the number of Newton iterations, which any rank count must reproduce.
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


if os.environ.get("TRAP"):
    # step2d: v = 0 on the bottom (3) and top (4) wall faces; the corners (5, +-1) are tagged as outflow nodes
    from strong_dirichlet_trap import install
    install((3, 4), ([5.0, -1.0], [5.0, 1.0]),
            report=lambda n: print("PROBE trapped corners", n, flush=True))

Context.contextOptionsString = "he=%s schur_solver='petsc_ASM'" % os.environ.get("HE", "0.2")
so = defaults.load_system("step2d_so", MODULES)
p = defaults.load_physics("twp_navier_stokes_step2d_p", MODULES)
n = defaults.load_numerics("twp_navier_stokes_step2d_n", MODULES)
n.linearSmoother = None
n.nl_atol_res = 1.0e-12
n.l_atol_res = 1.0e-14
n.tolFac = 0.0
options = PETSc.Options()
options.setValue("rans2p_ksp_type", "gmres")
options.setValue("rans2p_ksp_rtol", 1.0e-13)
options.setValue("rans2p_ksp_atol", 1.0e-15)
options.setValue("rans2p_ksp_max_it", 50)
options.setValue("rans2p_pc_type", "lu")
options.setValue("rans2p_pc_factor_mat_solver_type", os.environ.get("LU", "superlu_dist" if comm.size() > 1 else "superlu"))
p.name = so.name = "step2d_auth"
so.sList = [default_s]
mesh_options = p.domain.MeshOptions
mesh_options.parallelPartitioningType = MeshTools.MeshParallelPartitioningTypes.element
mesh_options.nLayersOfOverlapForParallel = 0
ns = NumericalSolution.NS_base(so, [p], [n], so.sList, runtime.RunOptions())
ns.calculateSolution("step2d_auth")
m = ns.modelList[0]
model = m.levelModelList[-1]
mesh = model.mesh
owned = mesh.nNodes_owned
from mpi4py import MPI
c = comm.comm.tompi4py()
result = {}
for i, name in enumerate(("p", "u", "v")):
    dof = model.u[i].dof[:owned]
    result[name + "2"] = c.allreduce(float((dof ** 2).sum()), op=MPI.SUM)
    result[name + "s"] = c.allreduce(float(dof.sum()), op=MPI.SUM)
result["newton"] = float(m.solver.solverList[-1].its)
interfaces = int((mesh.elementBoundaryMaterialTypes[mesh.exteriorElementBoundariesArray] == -1).sum())
print("RANK rank=%d interfaces=%d owned=%d" % (comm.rank(), interfaces, owned), flush=True)
if comm.rank() == 0:
    print("RESULT ranks=%d " % comm.size() + " ".join("%s=%.17e" % kv for kv in result.items()), flush=True)
