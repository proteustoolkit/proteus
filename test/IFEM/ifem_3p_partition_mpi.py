#!/usr/bin/env python
"""One three-phase case of test/ci/ladr_3p_2d_p.py; prints its L2 and Linfty errors.

Run by test_ifem_3p.py under mpiexec to compare a partitioned run against the serial pins.
The case's context options are argv[1]. NLAYERS, when set, overrides the two layers of mesh
overlap the drivers ask for (the guard test sets 1). The linear solver comes from
PETSC_OPTIONS (a parallel direct solve, set by the test).
"""
import os
import sys

from proteus import Comm

comm = Comm.init()

from proteus import Context, NumericalSolution, default_s, defaults
from proteus.iproteus import opts

CI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "ci")

Context.contextOptionsString = sys.argv[1] + " usePetsc=True unstructured=False"
try:
    p = defaults.load_physics("ladr_3p_2d_p", CI_DIR)
    n = defaults.load_numerics("ladr_3p_2d_c0pk_n", CI_DIR)
finally:
    Context.contextOptionsString = None
if os.environ.get("NLAYERS"):
    p.domain.MeshOptions.nLayersOfOverlapForParallel = int(os.environ["NLAYERS"])
    n.nLayersOfOverlapForParallel = int(os.environ["NLAYERS"])

so = defaults.System_base()
so.name = p.name
so.pnList = [("ladr_3p_2d_p", "ladr_3p_2d_c0pk_n")]
so.sList = [default_s]
for attr in ("tnList", "systemStepControllerType", "systemStepExact", "archiveFlag"):
    if hasattr(n, attr):
        setattr(so, attr, getattr(n, attr))
ns = NumericalSolution.NS_base(so, [p], [n], so.sList, opts)
ns.calculateSolution("ifem3p_mpi")
lm = ns.modelList[0].levelModelList[-1]
if comm.rank() == 0:
    print("RESULT ranks=%d L2=%.17e Linfty=%.17e"
          % (comm.size(), float(lm.L2_error[0] ** 0.5), float(lm.Linfty_error[0])), flush=True)
