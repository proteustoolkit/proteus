#!/usr/bin/env python
"""Run the RANS2P_IB beam channel and print what a serial run must match.

Run by test_rans2p_ib.py, under mpiexec for several ranks. Prints the sums over owned nodes of p, u, v, w (and of
their squares) at T, the beams' drag and the beam loads q1, q2, q3, summed. Direct solves (SuperLU / SuperLU_DIST) and
a 1e-10 Newton tolerance, so the comparison is not limited by a Krylov tolerance. Environment: PARTITION (node or
element, default node), NLAYERS (default 0), OPTS (the deck's context options, e.g. beamRigid=False).
"""
import os
import sys

from proteus import Comm

comm = Comm.init()

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from proteus import Context, MeshTools, NumericalSolution, default_s, default_so, defaults, runtime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
Context.contextOptionsString = os.environ.get("OPTS", "")
p = defaults.load_physics("beam_channel_p", HERE)
n = defaults.load_numerics("beam_channel_n", HERE)
n.parallelPartitioningType = getattr(MeshTools.MeshParallelPartitioningTypes, os.environ.get("PARTITION", "node"))
n.nLayersOfOverlapForParallel = int(os.environ.get("NLAYERS", "0"))
options = PETSc.Options()
options.setValue("rans2p_ksp_type", "preonly")
options.setValue("rans2p_pc_type", "lu")
options.setValue("rans2p_pc_factor_mat_solver_type", "superlu_dist" if comm.size() > 1 else "superlu")
so = default_so
so.name = p.name
so.pnList = [(p, n)]
so.sList = [default_s]
so.tnList = [0.0] + [n.DT * (i + 1) for i in range(n.nDTout)]
ns = NumericalSolution.NS_base(so, [p], [n], so.sList, runtime.RunOptions())
ns.calculateSolution(p.name)
model = ns.modelList[0].levelModelList[-1]
c = comm.comm.tompi4py()
owned = model.mesh.nNodes_owned
result = {}
for i, name in enumerate("puvw"):
    dof = model.u[i].dof[:owned]
    result[name + "2"] = c.allreduce(float((dof ** 2).sum()), op=MPI.SUM)
    result[name + "s"] = c.allreduce(float(dof.sum()), op=MPI.SUM)
coefficients = model.coefficients
for i in range(3):
    result["drag%d" % i] = float(coefficients.beamDrag[i])
for name in ("q1", "q2", "q3", "xv", "yv", "zv"):
    result[name] = float(getattr(coefficients, name).sum())
if comm.rank() == 0:
    print("RESULT ranks=%d " % comm.size() + " ".join("%s=%.17e" % kv for kv in result.items()), flush=True)
