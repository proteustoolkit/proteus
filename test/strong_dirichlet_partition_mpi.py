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


def trap_owners(mesh, nranks):
    """Rank 1 gets the outflow-side element of a corner, rank 0 the element with the wall face.

    The step2d corners (5, 1) and (5, -1) are tagged as outflow nodes (p = 0 by the node tag) and v = 0 comes
    from the wall face only. Where a corner has two elements, one with the outflow face and one with the wall
    face, putting them on different ranks makes the corner a strong-Dirichlet dof for v on the rank with the
    wall face and a free one on the other. The other elements are dealt out in contiguous blocks to the ranks
    except 1 (rank 1 has nothing else, so it is a ghost-only copy of the corner).
    """
    others = [r for r in range(nranks) if r != 1] or [0]
    owners = np.array([others[min(len(others) - 1, e * len(others) // mesh.nElements_global)]
                       for e in range(mesh.nElements_global)], dtype=np.intc)
    exterior = set(mesh.exteriorElementBoundariesArray.tolist())
    found = 0
    for corner in ([5.0, -1.0], [5.0, 1.0]):
        i = np.where(np.abs(mesh.nodeArray[:, :2] - corner).max(1) < 1e-9)[0]
        if len(i) == 0:
            continue
        i = int(i[0])
        wall_el, out_el = None, None
        for e in range(mesh.nElements_global):
            if i not in mesh.elementNodesArray[e]:
                continue
            for f in mesh.elementBoundariesArray[e]:
                if f in exterior and i in mesh.elementBoundaryNodesArray[f]:
                    if abs(mesh.elementBoundaryBarycentersArray[f][1]) > 0.999:   # top or bottom wall face
                        wall_el = e
                    else:
                        out_el = e
        if wall_el is not None and out_el is not None and wall_el != out_el:
            owners[wall_el] = 0
            owners[out_el] = 1
            found += 1
    if os.environ.get("PROBE"):
        print("PROBE trapped corners", found, flush=True)
    return owners


if os.environ.get("TRAP"):
    _part = MeshTools.Mesh.partitionMesh

    def partitionMesh(self, nLayersOfOverlap=1, parallelPartitioningType=MeshTools.MeshParallelPartitioningTypes.node,
                      nodeOwners=None, elementOwners=None, nodeWeights=None):
        if elementOwners is None and Comm.get().size() > 1:
            elementOwners = trap_owners(self, Comm.get().size())
        return _part(self, nLayersOfOverlap=nLayersOfOverlap, parallelPartitioningType=parallelPartitioningType,
                     nodeOwners=nodeOwners, elementOwners=elementOwners, nodeWeights=nodeWeights)
    MeshTools.Mesh.partitionMesh = partitionMesh

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
