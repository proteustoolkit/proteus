#!/usr/bin/env python
"""One case of test/ci/ladr_ss_3d_p.py; prints its L2 and Linfty errors.

Run by test_adr_3d.py, alone or under mpiexec. The context options are argv[1]. In parallel the linear
solver comes from PETSC_OPTIONS (a parallel direct solve, set by the test).
"""
import os
import sys

from proteus import Comm

comm = Comm.init()

from proteus import Context, NumericalSolution, default_s, defaults
from proteus.iproteus import opts

CI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "ci")

Context.contextOptionsString = sys.argv[1]
try:
    p = defaults.load_physics("ladr_ss_3d_p", CI_DIR)
    n = defaults.load_numerics("ladr_ss_3d_n", CI_DIR)
finally:
    Context.contextOptionsString = None

so = defaults.System_base()
so.name = p.name
so.pnList = [("ladr_ss_3d_p", "ladr_ss_3d_n")]
so.sList = [default_s]
for attr in ("tnList", "systemStepControllerType", "systemStepExact", "archiveFlag"):
    if hasattr(n, attr):
        setattr(so, attr, getattr(n, attr))
ns = NumericalSolution.NS_base(so, [p], [n], so.sList, opts)
ns.calculateSolution("adr3d")
lm = ns.modelList[0].levelModelList[-1]
if comm.rank() == 0:
    print("RESULT ranks=%d L2=%.17e Linfty=%.17e"
          % (comm.size(), float(lm.L2_error[0] ** 0.5), float(lm.Linfty_error[0])), flush=True)
