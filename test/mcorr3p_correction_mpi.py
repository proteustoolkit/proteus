#!/usr/bin/env python
"""twophaseflow_partition_mpi.py on the 2D rising bubble (RANS3PF, VOF3P, NCLS3P, RDLS, MCorr3P) with the level solver of MCorr3P named by CORRECTION, for test_mcorr3p_correction.py.

CORRECTION = newton (the deck's own), dg (MCorr3P.ElementNewton), dgp0 (ElementConstantNewton), global (GlobalConstantNewton)
or none (DummyNewton), the same names as the ``correctionType`` of the vortex2D deck. After the run rank 0 writes mcorr3p.json: the solver class the model
used, the model's mass conservation error and the number of distinct values of its dofs.
"""
import json
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.update(DIR="surface_tension/rising_bubble_rans3p", SO="risingBubble_so.py",
                  OPTS="nd=2 USE_SUPG_NS=1 ARTIFICIAL_VISCOSITY_NS=1", DIRECT="1", ATOL="1e-10", MAXITS="200",
                  FT=os.environ.get("FT", "0.004"))
sys.path.insert(0, HERE)
import numpy as np
from proteus import NumericalSolution as NS
from proteus import Comm
from proteus.kernels import MCorr3P
from proteus.NonlinearSolvers import NLNI

CLASSES = {"dg": MCorr3P.ElementNewton, "dgp0": MCorr3P.ElementConstantNewton, "global": MCorr3P.GlobalConstantNewton,
           "none": MCorr3P.DummyNewton}
CORRECTION = os.environ.get("CORRECTION", "newton")
_init = NS.NS_base.__init__


def init(self, so, pList, nList, *args, **kwargs):
    if CORRECTION in CLASSES:
        for p, n in zip(pList, nList):
            if getattr(p.LevelModelType, "__module__", "").endswith("MCorr3P"):
                n.multilevelNonlinearSolver = NLNI
                n.levelNonlinearSolver = CLASSES[CORRECTION]
    return _init(self, so, pList, nList, *args, **kwargs)


NS.NS_base.__init__ = init
_calc = NS.NS_base.calculateSolution


def calc(self, *a, **k):
    r = _calc(self, *a, **k)
    m = [m for m in self.modelList if type(m.levelModelList[-1]).__module__.endswith("MCorr3P")][0]
    mc = m.levelModelList[-1]
    if Comm.get().rank() == 0:
        with open("mcorr3p.json", "w") as f:
            json.dump(dict(solver=type(m.solver.solverList[-1]).__name__,
                           mass_error=float(mc.coefficients.massConservationError),
                           finite=bool(np.isfinite(mc.u[0].dof).all()),
                           distinct=int(len(np.unique(mc.u[0].dof[:mc.mesh.nNodes_owned])))), f)
    return r


NS.NS_base.calculateSolution = calc
runpy.run_path(os.path.join(HERE, "twophaseflow_partition_mpi.py"), run_name="__main__")
