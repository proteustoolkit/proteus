"""test/ci's ladr_2d_n, with an exception raised at the first archive write (NS_base.archiveInitialSolution).

RAISERANK=r raises a ValueError on rank r only, as one rank's Archiver did in a 4096-rank run whose other ranks
then waited in a collective until the wall time ran out; RAISERANK=all raises a NumericalSolution.StepTimeError on
every rank together, as the stepping loop's collective time check does. Without RAISERANK nothing is raised.
Copy it beside adr.py, ladr_2d_p.py and ladr_2d_n.py from test/ci.
"""
import os
from ladr_2d_n import *
from proteus import Comm
from proteus import NumericalSolution as _NS

# LU is serial only
multilevelLinearSolver = levelLinearSolver = KSP_petsc4py

if os.environ.get("RAISERANK") and not getattr(_NS.NS_base.archiveInitialSolution, "raises", False):
    _archiveInitialSolution = _NS.NS_base.archiveInitialSolution

    def _raise(self, model, index):
        rank = Comm.get().rank()
        if os.environ["RAISERANK"] == "all":
            raise _NS.StepTimeError("RAISERANK: every rank stops at its first archive write")
        if rank == int(os.environ["RAISERANK"]):
            raise ValueError("RAISERANK: rank %d fails its first archive write" % rank)
        return _archiveInitialSolution(self, model, index)
    _raise.raises = True
    _NS.NS_base.archiveInitialSolution = _raise
