"""Run the vortex2D level-set/VOF case in one of three modes, for the
hotstart measurement (C17).

    python run_case.py OUTDIR TI TORDER MODE [N]

TI      be | vbdf | flcbdf        time integration for ls and vof
TORDER  timeOrder (vbdf)
MODE    full   run 0 -> T
        stop   run 0 -> t_N (N user steps) and exit; archive left in OUTDIR
        hot    hotstart from OUTDIR's archive at t_N, then run on to T

The case is test/levelset/vortex2D: ls + redistance + vof + mass
correction, a lagged shock capturing term on ls and vof (the `*_last`
history), and a choice of time integrator.
"""
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
vortex = os.path.join(here, "..", "levelset", "vortex2D")

outdir, ti, torder, mode = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
N = int(sys.argv[5]) if len(sys.argv) > 5 else 40
os.makedirs(outdir, exist_ok=True)
os.chdir(outdir)
sys.path.insert(0, vortex)

sys.path.insert(0, here)
import vortex2D  # noqa: E402
# HS_PARALLEL=1 selects the PETSc linear solvers the n files use in parallel
# (vortex2D.parallel is False, which means LU, which parallel runs reject)
if os.environ.get("HS_PARALLEL"):
    vortex2D.parallel = True
    # the decks' parallel branch leaves the PETSc solvers at their defaults (a KSP atol of 1 for
    # the redistancing model): direct solves on the subdomain blocks, tight tolerance
    from petsc4py import PETSc
    _opts = PETSc.Options()
    for _pre in ("ncls_", "rdls_", "vof_", "mcorr_"):
        _opts[_pre + "ksp_type"] = "gmres"
        _opts[_pre + "ksp_rtol"] = "1e-12"
        _opts[_pre + "ksp_atol"] = "1e-14"
        _opts[_pre + "pc_type"] = "asm"
        _opts[_pre + "pc_asm_overlap"] = "1"
        _opts[_pre + "sub_pc_type"] = "lu"
# HS_TI_LS / HS_TI_VOF override the integrator of one model (ls or vof)
vortex2D.timeIntegration_vof = os.environ.get("HS_TI_VOF", ti)
vortex2D.timeIntegration_ls = os.environ.get("HS_TI_LS", ti)
# HS_FLCBDF_TOL sets atol_u and rtol_u (the FLCBDF error control) for the models
# that use it; the n files do not set them, so they are overridden where the
# controller reads them
if os.environ.get("HS_FLCBDF_TOL"):
    from proteus import StepControl as _SC
    _init = _SC.FLCBDF_controller.__init__

    def _flcbdf_init(self, model, nOptions, _init=_init):
        for _d in (nOptions.atol_u, nOptions.rtol_u):
            for _ci in _d:
                _d[_ci] = float(os.environ["HS_FLCBDF_TOL"])
        if os.environ.get("HS_DTMIN"):
            nOptions.dt_min = float(os.environ["HS_DTMIN"])
        _init(self, model, nOptions)
    _SC.FLCBDF_controller.__init__ = _flcbdf_init
vortex2D.timeOrder = torder
vortex2D.soname = "hs_%s%d" % (ti, torder)
# decomposition switches: HS_LAG=0 turns the lagged shock capturing off,
# HS_RD=0 drops redistancing and mass correction (ls + vof only)
if os.environ.get("HS_LAG") == "0":
    vortex2D.lag_shockCapturing_vof = False
    vortex2D.lag_shockCapturing_ls = False
# HS_RD_LAG=1 turns the lagged shock capturing on for the redistancing model (mprans.RDLS)
if os.environ.get("HS_RD_LAG") == "1":
    vortex2D.lag_shockCapturing_rd = True
if os.environ.get("HS_RD") == "0":
    vortex2D.applyRedistancing = False
    vortex2D.applyCorrection = False
tag = os.environ.get("HS_TAG", "")
vortex2D.soname += tag
# HS_SONAME names the archive explicitly (a hot start with another integrator order
# reads the archive the stop run wrote)
vortex2D.soname = os.environ.get("HS_SONAME", vortex2D.soname)

import ls_vortex_2d_p, redist_vortex_2d_p, vof_vortex_2d_p, ls_consrv_vortex_2d_p  # noqa
import ls_vortex_2d_n, redist_vortex_2d_n, vof_vortex_2d_n, ls_consrv_vortex_2d_n  # noqa
if os.environ.get("HS_PARALLEL"):
    for _m in (ls_vortex_2d_n, redist_vortex_2d_n, vof_vortex_2d_n, ls_consrv_vortex_2d_n):
        _m.l_atol_res = float(os.environ.get("HS_PARALLEL_ATOL", "1e-12"))   # the redistancing deck leaves this at the default of 1
        _m.linTolFac = 0.0
        _m.l_rtol_res = 0.0
import ls_vortex_2d_so as so  # noqa
from proteus import NumericalSolution, default_s  # noqa
from proteus.runtime import RunOptions  # noqa

so.name = vortex2D.soname
DT = so.DT
tN = N * DT
opts = RunOptions(gatherArchive=True)
if mode == "stop":
    so.tnList = [i * DT for i in range(N + 1)]
elif mode in ("hot", "full"):
    # HS_ENDN: stop after this output (a short run to compare, hot or not)
    if os.environ.get("HS_ENDN"):
        so.tnList = [i * DT for i in range(int(os.environ["HS_ENDN"]) + 1)]
if mode == "hot":
    opts.hotStart = True
    opts.hotStartMode = os.environ.get("HS_HOTMODE", "continue")
    opts.hotStartTime = tN
sList = [default_s for _ in so.pnList]
pm = {"ls": (ls_vortex_2d_p, ls_vortex_2d_n),
      "redist": (redist_vortex_2d_p, redist_vortex_2d_n),
      "vof": (vof_vortex_2d_p, vof_vortex_2d_n),
      "ls_consrv": (ls_consrv_vortex_2d_p, ls_consrv_vortex_2d_n)}
order = [n.replace("_vortex_2d_p", "") for n, _ in so.pnList]
ns = NumericalSolution.NS_base(
    so, [pm[o][0] for o in order], [pm[o][1] for o in order], sList, opts)
snap = os.environ.get("HS_SNAP")
if (snap or os.environ.get("HS_ORACLE")) and mode == "hot":
    # state the hotstart rebuilds: after setup and spin-up, before step N+1
    from snapshot import snapshot
    sc = ns.systemStepController
    orig = sc.stepExact_system
    def once(t, orig=orig):
        sc.stepExact_system = orig
        if snap:
            snapshot(ns, snap)
        oracle = os.environ.get("HS_ORACLE")
        if oracle:
            from snapshot import restore
            if "first" in oracle.split(","):
                ns.firstStep = False
            print("ORACLE restored", restore(ns, os.environ["HS_TRUE"], oracle.split(",")), "values")
            if os.environ.get("HS_SNAP2"):
                snapshot(ns, os.environ["HS_SNAP2"])
        return orig(t)
    sc.stepExact_system = once
if os.environ.get("HS_SNAP_MID") and mode == "full":
    # the uninterrupted run's state at the start of interval N+1, taken
    # mid-run (a stop run's state is taken after calculateSolution returns)
    from snapshot import snapshot
    sc0 = ns.systemStepController
    orig0 = sc0.stepExact_system
    done = [False]
    def count(t, orig0=orig0):
        if not done[0] and abs(sc0.t_system_last - tN) < 1e-9:
            done[0] = True
            snapshot(ns, os.environ["HS_SNAP_MID"])
        return orig0(t)
    sc0.stepExact_system = count
if os.environ.get("HS_SPIN") and mode == "full":
    # emulate, on the live uninterrupted state at step N, the calls the
    # hotstart's spin-up makes, to find which of them perturbs the run
    ops = os.environ["HS_SPIN"].split(",")
    scs = ns.systemStepController
    origs = scs.stepExact_system
    didspin = [False]
    def spin(t, origs=origs):
        if not didspin[0] and abs(scs.t_system_last - tN) < 1e-9:
            didspin[0] = True
            tn0, tn1 = ns.tnList[N], ns.tnList[N + 1]
            if any(o.startswith("restore") for o in ops):
                from snapshot import snapshot as _snap
                _snap(ns, "spin_before.pkl")
            if "res0" in ops:
                for m in ns.modelList:
                    for lm, lu, lr in zip(m.levelModelList, m.uList, m.rList):
                        lm.getResidual(lu, lr)
            for m in ns.modelList:
                for lm, lu, lr in zip(m.levelModelList, m.uList, m.rList):
                    if "ith" in ops:
                        lm.getResidual(lu, lr)
                        lm.initializeTimeHistory()
                        lm.timeIntegration.initializeSpaceHistory()
                    if "ith_t" in ops:
                        lm.initializeTimeHistory()
                    if "ith_s" in ops:
                        lm.timeIntegration.initializeSpaceHistory()
                    if "res1" in ops:
                        lm.getResidual(lu, lr)
                    if "emt" in ops:
                        lm.estimate_mt()
                    if "aux" in ops:
                        lm.calculateAuxiliaryQuantitiesAfterStep()
                if "dt0" in ops:
                    m.stepController.initialize_dt_model(tn0, tn1)
                if "sith" in ops:
                    m.stepController.initializeTimeHistory()
                if "sig" in ops:
                    m.stepController.setInitialGuess(m.uList, m.rList)
            for o in ops:
                if o.startswith("restore"):
                    from snapshot import restore as _rest
                    g = o.split(":")[1].split("+") if ":" in o else ["all"]
                    print("SPIN restored", _rest(ns, "spin_before.pkl", g))
        return origs(t)
    scs.stepExact_system = spin
rec_path = os.environ.get("HS_REC")
if rec_path:
    # the dofs of every model as each model's step is taken, before the
    # mass correction's postStep changes the level set
    import pickle
    rec = []
    orig_post = ns.postStep
    def post(model, orig_post=orig_post):
        rec.append((model.name, {ci: u.dof.copy() for ci, u in model.levelModelList[-1].u.items()}))
        return orig_post(model)
    ns.postStep = post
ns.calculateSolution(so.name)
if rec_path:
    pickle.dump(rec, open(rec_path, "wb"))
if os.environ.get("HS_SNAP_END") and mode == "hot":
    from snapshot import snapshot
    snapshot(ns, os.environ["HS_SNAP_END"])
if snap and mode == "stop":
    # the uninterrupted run's state at step N
    from snapshot import snapshot
    snapshot(ns, snap)
print("DONE", mode, so.name, "tN", tN)
