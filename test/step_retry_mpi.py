#!/usr/bin/env python
"""A test/TwoPhaseFlow deck (DECK: fillingTank.py, CLSVOF + RANS3PF projection scheme, or damBreakLevelSet.py,
RANS2P + VOF + NCLS + RDLS + MCorr) at he=0.1 to t=0.05, for test_step_retry.py.

Run serially in an empty directory; leaves u.csv (every model's dofs, sorted by (100 * model + component, x, y): key, x,
y, value), accepted.txt (the accepted system times) and the deck's own history files. ATOL=a sets every model's
nonlinear atol to a (linear a/100); without it the deck's tolerances apply.
FAILMODEL=<name prefix> and FAILAT=n report the n-th solve of that model as failed (after solving it), so the system
step is cut and retried. REPLAY=<file> makes the system step land on the times in the file instead of choosing them,
which gives the run a failure-free run with the same steps. NORESTORE=1 turns off the restore before a retry (control).
"""
import glob, os, runpy, shutil, sys
import numpy as np

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TwoPhaseFlow")
DECK = os.environ.get("DECK", "fillingTank.py")
for f in [os.path.join(SRC, DECK)] + glob.glob(os.path.join(SRC, "mesh*")):
    shutil.copy(f, ".")

from proteus import NonlinearSolvers, SplitOperator
from proteus import NumericalSolution as NS

_init = NS.NS_base.__init__
def init(self, so, pList, nList, *args, **kwargs):
    if os.environ.get("ATOL"):
        atol = float(os.environ["ATOL"])
        for n in nList:
            n.nl_atol_res, n.l_atol_res, n.l_rtol_res, n.linTolFac, n.tolFac = atol, 0.01 * atol, 0.0, 0.0, 0.0
    return _init(self, so, pList, nList, *args, **kwargs)
NS.NS_base.__init__ = init

if os.environ.get("NORESTORE"):
    NS.NS_base.restoreSystemStepState = lambda self: None

if os.environ.get("FAILMODEL"):
    _solve = NonlinearSolvers.Newton.solve
    calls = [0]
    def solve(self, *args, **kwargs):
        failed = _solve(self, *args, **kwargs)
        # CLSVOFNewton.solve calls Newton.solve for its stages: count the model's solve, not the stages
        if type(self).__name__ != "CLSVOFNewton" and self.F.name.startswith(os.environ["FAILMODEL"]):
            calls[0] += 1
            if calls[0] == int(os.environ["FAILAT"]):
                self.failedFlag = True
        return self.failedFlag
    NonlinearSolvers.Newton.solve = solve
    if os.environ["FAILMODEL"].startswith("clsvof"):
        _csolve = NonlinearSolvers.CLSVOFNewton.solve
        ccalls = [0]
        def csolve(self, *args, **kwargs):
            _csolve(self, *args, **kwargs)
            ccalls[0] += 1
            if ccalls[0] == int(os.environ["FAILAT"]):
                self.failedFlag = True
            return self.failedFlag
        NonlinearSolvers.CLSVOFNewton.solve = csolve

if os.environ.get("REPLAY"):
    times = np.loadtxt(os.environ["REPLAY"])
    _choose = SplitOperator.Sequential_MinAdaptiveModelStep.choose_dt_system
    def choose_dt_system(self):
        _choose(self)
        later = times[times > self.t_system_last * (1 + 1e-12) + 1e-15]
        if len(later):
            self.dt_system = later[0] - self.t_system_last
            self.t_system = later[0]
            self.stepSequence = [(self.t_system, model) for model in self.modelList]
            for model in self.modelList:
                model.stepController.dt_model = self.dt_system
                model.stepController.set_dt_allLevels()
    SplitOperator.Sequential_MinAdaptiveModelStep.choose_dt_system = choose_dt_system

accepted = []
_taken = SplitOperator.Sequential_MinAdaptiveModelStep.sequenceTaken
def sequenceTaken(self):
    accepted.append(self.t_system)
    return _taken(self)
SplitOperator.Sequential_MinAdaptiveModelStep.sequenceTaken = sequenceTaken

_calc = NS.NS_base.calculateSolution
def calc(self, *a, **k):
    r = _calc(self, *a, **k)
    rows = []
    for mi, mod in enumerate(self.modelList):
        lm = mod.levelModelList[-1]
        for ci in range(lm.nc):
            x = getattr(lm.u[ci].femSpace.dofMap, "lagrangeNodesArray", lm.mesh.nodeArray)[:, :2]
            rows.append(np.hstack([np.full((x.shape[0], 1), 100 * mi + ci), x, lm.u[ci].dof[:, None]]))
    a = np.vstack(rows)
    np.savetxt("u.csv", a[np.lexsort((a[:, 2], a[:, 1], a[:, 0]))], delimiter=",", header="key,x,y,u")
    np.savetxt("accepted.txt", np.array(accepted))
    return r
NS.NS_base.calculateSolution = calc

sys.argv = ['parun', '--TwoPhaseFlow', DECK, '-l1', '-C', 'final_time=0.05 dt_output=0.05 he=0.1']
runpy.run_path(shutil.which('parun'), run_name='__main__')
