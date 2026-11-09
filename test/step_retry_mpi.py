#!/usr/bin/env python
"""A deck (by default a test/TwoPhaseFlow one: DECK=fillingTank.py, CLSVOF + RANS3PF projection scheme, or
damBreakLevelSet.py, RANS2P + VOF + NCLS + RDLS + MCorr, at he=0.1 to t=0.05), for test_step_retry.py.

Run serially in an empty directory; leaves u.csv (every model's dofs, sorted by (100 * model + component, x, y, z):
key, x, y, z, value), accepted.txt (the accepted system times), retries.txt (how many system steps were retried) and
the deck's own history files. ATOL=a sets every model's nonlinear atol to a (linear a/100); without it the deck's
tolerances apply; MAXITS=n raises every model's maxNonlinearIts to n.
FAILMODEL=<name prefix> and FAILAT=n report the n-th solve of that model as failed (after solving it), so the system
step is cut and retried; FAILSTEP=n instead counts the model's steps, whatever its nonlinear solver (FAILAT counts a
TwoStageNewton's stages: an odd n fails a first stage, which skips the second). REPLAY=<file> makes the system step
land on the times in the file instead of choosing them (so.systemStepSchedule, the first step included), which gives
the run a failure-free run with the same steps. NORESTORE=1 turns off the restore before a retry, NOTMODELRESTORE=1
only its t_model_last part. NANDT=n makes the n-th system step the controller chooses nan (NANDTRANK=r: on rank r
only); NOTIMECHECK=1 turns off the stepping loop's time check (NS_base._checkStepTimes). accepted.txt and
retries.txt are written also when the run stops on a StepTimeError (u.csv is not). SYSTEMCONTROLLER=<class of proteus.SplitOperator> runs the deck under that
system step controller (Sequential_MinAdaptiveModelStep_SS has no deck of its own).
FT and OPTS set the final time and the deck's other options, DTOUT the archive interval (default FT);
HOTSTART=t hot-starts from the archive the directory holds (-H -t t, HOTSTARTMODE continue or restart); DIR and SO run another deck under DIR instead: SO is
parun's file arguments ("x_so.py", "x_p.py x_n.py" or "--SWEs x.py") and OPTS the whole -C string.
"""
import glob, os, runpy, shutil, sys
import numpy as np

SO = os.environ.get("SO")
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("DIR", "TwoPhaseFlow"))
DECK = os.environ.get("DECK", "fillingTank.py")
for f in (glob.glob(os.path.join(SRC, "*")) if SO else [os.path.join(SRC, DECK)] + glob.glob(os.path.join(SRC, "mesh*"))):
    if os.path.isfile(f):
        shutil.copy(f, ".")

from proteus import NonlinearSolvers, SplitOperator
from proteus import NumericalSolution as NS

def boundary_integral_gauges(pList, nList):
    """BIGAUGES=flags (comma list): a BoundaryIntegralGauges of every quantity on those flags, with a moment point on
    the first, attached to the RANS2P/RANS3PF model; it writes boundary_integrals.csv."""
    from proteus.Gauges import BoundaryIntegralGauges
    for p, n in zip(pList, nList):
        if type(p.coefficients).__module__ in ("proteus.mprans.RANS2P", "proteus.mprans.RANS3PF"):
            flags = [int(f) for f in os.environ["BIGAUGES"].split(",")]
            gauge = BoundaryIntegralGauges(tags=flags, moment_points={str(flags[0]): (0.1, 0.2, 0.0)},
                                           fileName="boundary_integrals.csv")
            n.auxiliaryVariables = list(getattr(n, "auxiliaryVariables", [])) + [gauge]


_init = NS.NS_base.__init__
def init(self, so, pList, nList, *args, **kwargs):
    if os.environ.get("NOHOTSTARTMODE"):
        # options from a driver older than --hotStartMode (sList, opts follow nList)
        delattr(args[1], "hotStartMode")
    if os.environ.get("SYSTEMCONTROLLER"):
        # SYSTEMCONTROLLER=<class in proteus.SplitOperator>: run the deck under that system step controller
        so.systemStepControllerType = getattr(SplitOperator, os.environ["SYSTEMCONTROLLER"])
    if os.environ.get("REPLAY"):
        # the system steps land on the replayed run's accepted times, the first step included
        so.systemStepSchedule = np.loadtxt(os.environ["REPLAY"])
    if os.environ.get("BIGAUGES"):
        boundary_integral_gauges(pList, nList)
    if os.environ.get("ATOL"):
        atol = float(os.environ["ATOL"])
        for n in nList:
            n.nl_atol_res, n.l_atol_res, n.l_rtol_res, n.linTolFac, n.tolFac = atol, 0.01 * atol, 0.0, 0.0, 0.0
    if os.environ.get("MAXITS"):
        for n in nList:
            n.maxNonlinearIts = max(getattr(n, "maxNonlinearIts", 0), int(os.environ["MAXITS"]))
    return _init(self, so, pList, nList, *args, **kwargs)
NS.NS_base.__init__ = init

if os.environ.get("OTHERPARTITION"):
    # the archived solver state looks as if another partition wrote it (its fingerprint differs)
    _fingerprint = NS.NS_base._solver_state_fingerprint
    def _other_partition(self, model, index):
        f = _fingerprint(self, model, index)
        f[3:] += 1  # same ranks and mesh, other pieces of it
        return f
    NS.NS_base._solver_state_fingerprint = _other_partition

if os.environ.get("NORESTORE"):
    NS.NS_base.restoreSystemStepState = lambda self: None

if os.environ.get("NOTMODELRESTORE"):
    # the retry leaves each model's t_model_last where the failed attempt put it (before 83ac62c9): under
    # Sequential_MinModelStep the system time then walks backwards to -inf and nan
    _restore_tmodel = NS.NS_base.restoreSystemStepState
    def restore_without_t_model_last(self):
        kept = [m.stepController.t_model_last for m in self.modelList]
        _restore_tmodel(self)
        for m, t in zip(self.modelList, kept):
            m.stepController.t_model_last = t
    NS.NS_base.restoreSystemStepState = restore_without_t_model_last

if os.environ.get("NOTIMECHECK"):
    # the stepping loop as it was before the time check: a non-finite or backward time goes on
    NS.NS_base._checkStepTimes = lambda self, *args, **kwargs: True

if os.environ.get("FAILMODEL") and os.environ.get("FAILSTEP"):
    # FAILSTEP=n: report the model's n-th model step as failed (after solving it), whatever its nonlinear solver
    # (the explicit shallow-water solvers do not go through Newton.solve)
    steps = [0]
    def failing(_ml):
        def solveMultilevel(self, *args, **kwargs):
            failed = _ml(self, *args, **kwargs)
            if self.solverList[-1].F.name.startswith(os.environ["FAILMODEL"]):
                steps[0] += 1
                if steps[0] == int(os.environ["FAILSTEP"]):
                    return True
            return failed
        return solveMultilevel
    for cls in (NonlinearSolvers.MultilevelNonlinearSolver, NonlinearSolvers.NLNI):
        cls.solveMultilevel = failing(cls.solveMultilevel)
elif os.environ.get("FAILMODEL"):
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

# the system step controllers the decks use: the adaptive one (TwoPhaseFlow, so files) and the default (SWFlow)
CONTROLLERS = (SplitOperator.Sequential_MinAdaptiveModelStep, SplitOperator.Sequential_MinModelStep,
               SplitOperator.Sequential_MinAdaptiveModelStep_SS)


accepted = []
def recording(_taken):
    def sequenceTaken(self):
        accepted.append(self.t_system)
        return _taken(self)
    return sequenceTaken
for cls in CONTROLLERS:
    cls.sequenceTaken = recording(cls.sequenceTaken)

if os.environ.get("NANDT"):
    # NANDT=n: the n-th system step the controller chooses (choose_dt_system) comes out nan, as a nan CFL or error
    # estimate would make it; NANDTRANK=r: on rank r only
    chosen = [0]
    def nan_dt(_choose):
        def choose_dt_system(self):
            _choose(self)
            chosen[0] += 1
            from proteus import Comm
            if (chosen[0] == int(os.environ["NANDT"]) and
                    int(os.environ.get("NANDTRANK", Comm.get().rank())) == Comm.get().rank()):
                self.dt_system = float("nan")
                self.t_system = self.t_system_last + self.dt_system
                self.stepSequence = [(self.t_system, m) for (_, m) in self.stepSequence]
                for model in self.modelList:
                    model.stepController.dt_model = self.dt_system
                    model.stepController.set_dt_allLevels()
        return choose_dt_system
    for cls in CONTROLLERS:
        cls.choose_dt_system = nan_dt(cls.choose_dt_system)

retries = [0]
_restore = NS.NS_base.restoreSystemStepState
def restoreSystemStepState(self):
    retries[0] += 1
    return _restore(self)
NS.NS_base.restoreSystemStepState = restoreSystemStepState

_calc = NS.NS_base.calculateSolution
def calc(self, *a, **k):
    try:
        r = _calc(self, *a, **k)
    finally:
        # also when the run stops (NumericalSolution.StepTimeError)
        np.savetxt("accepted.txt", np.array(accepted))
        with open("retries.txt", "w") as f:
            f.write("%d\n" % retries[0])
    rows = []
    for mi, mod in enumerate(self.modelList):
        lm = mod.levelModelList[-1]
        for ci in range(lm.nc):
            x = getattr(lm.u[ci].femSpace.dofMap, "lagrangeNodesArray", lm.mesh.nodeArray)[:, :3]
            rows.append(np.hstack([np.full((x.shape[0], 1), 100 * mi + ci), x, lm.u[ci].dof[:, None]]))
    a = np.vstack(rows)
    # in parallel each rank writes its own dofs (owned and ghost), u<rank>.csv
    from proteus import Comm
    comm = Comm.get()
    np.savetxt("u.csv" if comm.size() == 1 else "u%d.csv" % comm.rank(), a[np.lexsort((a[:, 3], a[:, 2], a[:, 1], a[:, 0]))], delimiter=",", header="key,x,y,z,u")
    return r
NS.NS_base.calculateSolution = calc

if SO:
    # SO may be several words: "vos_p.py vos_n.py", or "--SWEs circular_damBreak.py"
    sys.argv = ['parun'] + SO.split() + ['-l' + os.environ.get('LOGL', '1'), '-C', os.environ.get("OPTS", "")]
else:
    sys.argv = ['parun', '--TwoPhaseFlow', DECK, '-l' + os.environ.get('LOGL', '1'), '-C',
                'final_time=%s dt_output=%s %s' % (os.environ.get("FT", "0.05"),
                                                   os.environ.get("DTOUT", os.environ.get("FT", "0.05")),
                                                   os.environ.get("OPTS", "he=0.1"))]
if os.environ.get("HOTSTART"):
    # continue the archive in this directory from its latest step at or before HOTSTART
    sys.argv += ['-H', '-t', os.environ["HOTSTART"], '--hotStartMode', os.environ.get("HOTSTARTMODE", "continue")]
# the source tree's driver when there is one: an installed parun is a copy made at install time
PARUN = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "scripts", "parun")
runpy.run_path(PARUN if os.path.isfile(PARUN) else shutil.which('parun'), run_name='__main__')
