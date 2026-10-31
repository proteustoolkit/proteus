"""Run the test/HotStart_3P cylinder case (mprans.RANS3PF, single phase, P1 velocity) in one of
three modes, for the hotstart measurement of the lagged RANS3PF shock capturing and subgrid error
(campaign C17). The deck sets ns_lag_shockCapturing = ns_lag_subgridError = True.

    python run_rans3p.py OUTDIR MODE [N]

MODE    full   run 0 -> T (T = HS_NSTEPS steps, default 40, of dt_fixed = 0.005)
        stop   run 0 -> t_N and exit; archive left in OUTDIR
        hot    hotstart from OUTDIR's archive at t_N, then run on to T

Environment: HS_STAB sets STABILIZATION_TYPE (0 SUPG, 1 EV weak residual, 2 EV strong residual).
"""
import os
import sys
import importlib
from importlib import reload

here = os.path.dirname(os.path.abspath(__file__))
testdir = os.path.join(here, "..")
case = os.path.join(testdir, "HotStart_3P")

outdir, mode = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 20
NSTEPS, DT = int(os.environ.get("HS_NSTEPS", 40)), 0.005
os.makedirs(outdir, exist_ok=True)
for f in os.listdir(case):
    if f.startswith("mesh."):
        dst = os.path.join(outdir, f)
        if not os.path.exists(dst):
            open(dst, "wb").write(open(os.path.join(case, f), "rb").read())
os.chdir(outdir)
sys.path.insert(0, testdir)
sys.path.insert(0, case)

from proteus import Context, default_s  # noqa: E402
import proteus  # noqa: E402
from proteus.iproteus import opts  # noqa: E402

T = (NSTEPS if mode in ("full", "hot") else N) * DT
Context.contextOptionsString = "T=%g vspaceOrder=1 isHotStart=%s" % (T, mode == "hot")
if os.environ.get("HS_STAB"):
    import re
    _p = os.path.join(case, "NS_hotstart.py")
    # STABILIZATION_TYPE is a module constant of the deck; override it after import below
so = importlib.import_module("HotStart_3P.NS_hotstart_so")
if os.environ.get("HS_STAB"):
    sys.modules["NS_hotstart"].STABILIZATION_TYPE = int(os.environ["HS_STAB"])
opts.gatherArchive = True
opts.hotStart = mode == "hot"
opts.hotStartTime = N * DT if mode == "hot" else 0.0
pList, nList = [], []
for (pm, nm) in so.pnList:
    pList.append(importlib.import_module(pm))
    nList.append(importlib.import_module(nm))
    if pList[-1].name is None:
        pList[-1].name = pm
    reload(pList[-1])
    reload(nList[-1])
sList = [default_s for _ in so.pnList] if so.sList == [] else so.sList
so.name = "hs_r3pf"

# Oracle (as oracle.sh for vortex2D): HS_DUMP=file in a stop run writes the state of every model's
# owners as the archived step ended; HS_INJECT=groups with HS_TRUE=file in a hot run puts those
# values back after the spin-up. Groups are name prefixes: sc (ShockCapturing), sge (SubgridError),
# ti (TimeIntegration), lm (the level model's own *_old/*Star/lagged/entropy/dt_last/dV_last state),
# ctl (model step controller), sys (system step controller), or "all"; a digit narrows to one model.
import copy  # noqa: E402
import pickle  # noqa: E402
import numpy as np  # noqa: E402

_plain = (np.ndarray, int, float, bool, type(None), str)


def _owners(ns):
    out = {}
    for i, m in enumerate(ns.modelList):
        lm = m.levelModelList[-1]
        out.update({"sc%d" % i: lm.shockCapturing, "sge%d" % i: lm.stabilization, "ti%d" % i: lm.timeIntegration,
                    "lm%d" % i: lm, "ctl%d" % i: m.stepController, "q%d" % i: lm.q, "ebqe%d" % i: lm.ebqe,
                    "nf%d" % i: lm.numericalFlux, "coef%d" % i: lm.coefficients})
    out["sys"] = ns.systemStepController
    return {g: o for g, o in out.items() if hasattr(o, "__dict__") or isinstance(o, dict)}


def _grab(o, group):
    out = {}
    if isinstance(o, dict):  # q and ebqe: the model's arrays at quadrature points, kept in place
        return {k: v.copy() for k, v in o.items() if isinstance(v, np.ndarray)}
    for k, v in vars(o).items():
        if k in ("cq", "u", "work", "numDiff"):  # aliases of the model's own arrays
            continue
        if group.startswith("lm") and not os.environ.get("HS_LMALL") and not any(w in k for w in ("_old", "Star", "lagged", "ntropy", "dt_last", "dV_last", "firstStep", "sharp")):
            continue
        if isinstance(v, _plain) or (isinstance(v, dict) and all(isinstance(x, np.ndarray) for x in v.values())):
            out[k] = copy.deepcopy(v)
    return out


def _put(o, saved, group=None):
    if isinstance(o, dict):
        for k, v in saved.items():
            if os.environ.get("HS_QKEYS") and not any(w in str(k) for w in os.environ["HS_QKEYS"].split(",")):
                continue
            if os.environ.get("HS_QEXACT") is not None and str(k) not in os.environ["HS_QEXACT"].split("|"):
                continue
            if k in o and o[k].shape == v.shape:
                o[k][:] = v
        return
    for k, v in saved.items():
        if os.environ.get("HS_OWNEREXACT") is not None and group == os.environ.get("HS_OWNEREXACT_GROUP") and k not in os.environ["HS_OWNEREXACT"].split("|"):
            continue
        cur = getattr(o, k, None)
        if k.endswith("_last") and k != "dt_last":  # may alias another array until the lag switches on
            setattr(o, k, copy.deepcopy(v))
        elif isinstance(v, np.ndarray) and isinstance(cur, np.ndarray) and cur.shape == v.shape:
            cur[:] = v
        else:
            setattr(o, k, copy.deepcopy(v))


if os.environ.get("HS_INJECT"):
    from proteus.NumericalSolution import NS_base
    _truth = pickle.load(open(os.environ["HS_TRUE"], "rb"))
    _groups = os.environ["HS_INJECT"].split(",")
    _orig = NS_base.restoreSolverState

    def _restore(self, model, index, tCount):
        _orig(self, model, index, tCount)
        if index == len(self.modelList) - 1:
            if os.environ.get("HS_FIRST_FALSE"):  # the stepping loop's own flag, which preStep reads
                self.firstStep = False
            own = _owners(self)
            for g in own:
                if "all" in _groups or any(g.startswith(w) for w in _groups):
                    _put(own[g], _truth[g], g)
    NS_base.restoreSolverState = _restore

if os.environ.get("HS_NO_COEF"):  # the archive as it was before the coefficients owner
    proteus.NumericalSolution.NS_base.SOLVER_STATE_OWNERS = ("shockCapturing", "stabilization", "timeIntegration")
if os.environ.get("HS_DUMP_START"):  # the state a hot run starts stepping from, to diff against a stop run's HS_DUMP
    from proteus.NumericalSolution import NS_base as _NS
    _prev = _NS.restoreSolverState

    def _restore_and_dump(self, model, index, tCount):
        _prev(self, model, index, tCount)
        if index == len(self.modelList) - 1:
            pickle.dump({g: _grab(o, g) for g, o in _owners(self).items()}, open(os.environ["HS_DUMP_START"], "wb"))
    _NS.restoreSolverState = _restore_and_dump

ns = proteus.NumericalSolution.NS_base(so, pList, nList, sList, opts)
ns.calculateSolution(so.name)
if os.environ.get("HS_DUMP"):
    pickle.dump({g: _grab(o, g) for g, o in _owners(ns).items()}, open(os.environ["HS_DUMP"], "wb"))
