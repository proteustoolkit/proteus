"""The on-node baseline of a production model: a TwoPhaseFlow case, kernel by kernel.

Companion to ``kernel_baseline_mpi.py``, which measures the generic Transport path on
Poisson. The two-phase models (RANS2P, RANS3PF, CLSVOF, the pressure models, ...) assemble
in one C++ call each -- ``self.rans3pf.calculateResidual(argsDict)`` with a hundred arrays --
so what is measured here is one pass per residual and per Jacobian, per model, per Newton
iteration, through a run of the case under ``mpiexec``:

* **ns per element** per kernel call, and the kernel's achieved **GB/s** against the node's
  STREAM (``machine_ceilings.py``), the bytes being those of the arrays the ``ArgumentsDict``
  carried in;
* each model's **Python remainder** around its kernel (boundary conditions, the
  ``ArgumentsDict`` itself, what the Transport base class still does per call);
* where a time step goes: assembly, the linear solves, the rest;
* **per Newton iteration, per model**: residual and Jacobian (each split into kernel, other C
  and Python), linear prepare (preconditioner setup), linear solve, and the Newton loop's own
  Python (norms, updates, ghost exchanges through petsc4py); and per system step the driver
  outside every Newton solve. Each model's Newton solver is logged call by call and the first
  ``warmup`` calls per model are left out, so the per-iteration figures are steady-state
  means (sum over calls / sum of iterations) with the median of the per-call ratio beside them.

Usage: model_baseline_mpi.py CASE.py [context='k=v ...'] [ceilings=FILE] [out=FILE] [datadir=DIR]
                             [warmup=3] [superlu=0|1] [petsc='-k v -k2 v2 !-k3'] [ctest=r|r-true]
                             [solution=PREFIX] [n='model.attr=value ...'] [params='model.attr=value ...']

``superlu=0`` overrides the case's ``SystemNumerics.useSuperlu`` before the models are set up, so
every model gets TwoPhaseFlow's iterative PETSc options (RANS2P: GMRES + ASM with LU blocks; VOF,
NCLS, MCorr: Krylov + BoomerAMG; RDLS: GMRES + ASM) instead of a ``superlu_dist`` direct solve
each; the test decks set ``useSuperlu=True``, production runs do not.

``petsc`` sets PETSc options after the case has set its own (TwoPhaseFlow writes the solver options
into the options database when the case loads, which overrides the command line); ``!-key`` deletes
one. ``ctest`` replaces RANS2P's linear convergence test (TwoPhaseFlow: 'r-true', a Python callback
that builds the true residual every Krylov iteration). ``solution`` writes every model's final
solution, per rank, to PREFIX.<rank>.npz, to compare two solver choices on the same partition, and
prints each model's integral of u over the domain (VOF: the water volume). ``n`` sets numerics-module
attributes of a model (``n='rans2p.nl_atol_res=1e-12'``: a tightly converged reference run).
``params`` sets attributes of a TwoPhaseFlow model's Parameters object before the problem initializes
(``params='rans2p.preconditioner=schur_boomeramg'``).

CASE.py is a TwoPhaseFlow case module (test/TwoPhaseFlow/damBreak.py, marin.py); the
context string is the case's ``-C`` options (``'final_time=0.05 he=0.05'``). Run from a
scratch directory: the case writes its archive there.
"""
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import accounting as acc  # noqa: E402

from proteus import Comm  # noqa: E402
comm = Comm.init()
from proteus import (cfemIntegrals, cnumericalFlux, csubgridError, cshockCapturing,  # noqa: E402
                     csparsity, csmoothers, ctimeIntegration, cpostprocessing, cmeshTools,
                     cpartitioning, LinearSolvers, NonlinearSolvers, Transport)
from proteus.mprans import cArgumentsDict  # noqa: E402

rank, size = comm.rank(), comm.size()
mpi = comm.comm.tompi4py()


def parse(argv):
    if not argv or not argv[0].endswith('.py'):
        raise SystemExit(__doc__)
    opts = {'case': argv[0], 'context': None, 'ceilings': None, 'out': None, 'datadir': '.',
            'warmup': '3', 'superlu': None, 'petsc': '', 'ctest': None, 'solution': None, 'n': '',
            'params': ''}
    for arg in argv[1:]:
        key, _, value = arg.partition('=')
        if key not in opts:
            raise SystemExit(__doc__)
        opts[key] = value
    return opts


def instrument_classes():
    for mod, label in ((cfemIntegrals, 'femIntegrals'), (cnumericalFlux, 'numericalFlux'),
                       (csubgridError, 'subgridError'), (cshockCapturing, 'shockCapturing'),
                       (csparsity, 'sparsity'), (csmoothers, 'smoothers'),
                       (ctimeIntegration, 'timeIntegration'), (cpostprocessing, 'postprocessing'),
                       (cmeshTools, 'meshTools'), (cpartitioning, 'partitioning')):
        acc.wrap_module(mod, label)
    #every model's residual and Jacobian, labelled by the model
    import importlib
    seen = set()
    for modname in ('RANS2P', 'RANS3PF', 'CLSVOF', 'VOF', 'NCLS', 'RDLS', 'MCorr',
                    'PresInc', 'Pres', 'PresInit'):
        try:
            cls = importlib.import_module('proteus.mprans.' + modname).LevelModel
        except (ImportError, AttributeError):
            continue
        if cls in seen:
            continue
        seen.add(cls)
        for method, tag in (('getResidual', 'residual'), ('getJacobian', 'jacobian'),
                            ('calculateCoefficients', 'coefficients')):
            if method in vars(cls):
                acc.wrap_phase(cls, method, lambda self, tag=tag: 'model %s/%s' % (self.name, tag))
    for cls in (LinearSolvers.KSP_petsc4py, LinearSolvers.LU, LinearSolvers.RedundantLU):
        for method, tag in (('prepare', 'linear prepare'), ('solve', 'linear solve')):
            if method in vars(cls):
                acc.wrap_phase(cls, method, tag)
    #the arrays each kernel is handed travel in an ArgumentsDict: count them
    cArgumentsDict.ArgumentsDict = acc.tracking_arguments_dict(cArgumentsDict.ArgumentsDict)


def instrument_kernels(ns):
    """Replace every level model's compiled kernel object with a timing proxy."""
    for model in ns.modelList:
        for lm in model.levelModelList:
            for attr, value in list(vars(lm).items()):
                module = type(value).__module__ or ''
                if module.startswith('proteus.mprans.c') or module.startswith('proteus.c'):
                    setattr(lm, attr, acc.KernelProxy(value, '%s.%s' % (lm.name, attr)))
                    continue
                #a kernel method bound at setup (VOF: self.calculateResidual = self.vof.calculate...)
                owner = getattr(value, '__self__', None)
                owner_module = type(owner).__module__ or '' if owner is not None else ''
                if (attr.startswith('calculate') and callable(value)
                        and (owner_module.startswith('proteus.mprans.c') or owner_module.startswith('proteus.c'))):
                    setattr(lm, attr, acc.wrap_pass(value, '%s.%s' % (lm.name, attr)))


#one record per Newton solve, per model: [iterations, seconds, {category: seconds}, {extra: value}]
newton_log = {}
NEWTON_CATEGORIES = ('residual kernel', 'residual other C', 'residual Python',
                     'jacobian kernel', 'jacobian other C', 'jacobian Python',
                     'linear prepare', 'linear solve', 'newton Python')
#inside the linear prepare and solve, not added to the categories: the Krylov iterations of the
#Newton solve's linear solves, and PETSc's own event times (preconditioner setup and application,
#matrix-vector products, GMRES orthogonalization) plus proteus's Python convergence test
#PCSetUpOnBlocks is where ASM factors its subdomain matrices: lazily, at the start of KSPSolve
NEWTON_EXTRAS = ('ksp its', 'PCSetUp', 'PCSetUpOnBlocks', 'PCApply', 'MatSolve', 'MatMult', 'KSPGMRESOrthog',
                 'VecScatterEnd', 'convergence test')
PETSC_EVENTS = NEWTON_EXTRAS[1:-1]
ksp_counter = {'its': 0, 'ctest': 0.0}
petsc_events = {}


def petsc_event_seconds():
    """Seconds this rank has spent in each of PETSC_EVENTS (PETSc logging on)."""
    return {name: ev.getPerfInfo()['time'] for name, ev in petsc_events.items()}


def instrument_linear_solvers():
    """Count the Krylov iterations of every KSP_petsc4py solve and time the r-true test."""
    from petsc4py import PETSc
    PETSc.Log.begin()
    for name in PETSC_EVENTS:
        petsc_events[name] = PETSc.Log.Event(name)
    cls = LinearSolvers.KSP_petsc4py
    solve, ctest = cls.solve, cls._converged_trueRes

    def counted_solve(self, *args, **kwargs):
        try:
            return solve(self, *args, **kwargs)
        finally:
            ksp_counter['its'] += self.ksp.its

    def timed_ctest(self, *args):
        t = time.perf_counter()
        try:
            return ctest(self, *args)
        finally:
            ksp_counter['ctest'] += time.perf_counter() - t
    cls.solve = counted_solve
    cls._converged_trueRes = timed_ctest


def newton_breakdown(path, level, ph0, ps0):
    """Seconds per category spent under the phase ``path`` since the snapshots ph0/ps0."""
    def ph_delta(suffix):
        return sum(v[1] - ph0.get(k, 0.0) for k, v in acc.phases.items()
                   if k.startswith(path + '/') and k.endswith(suffix))

    def ps_delta(phase_suffix, kernel):
        return sum(v[1] - ps0.get(k, 0.0) for k, v in acc.passes.items()
                   if k[0].startswith(path + '/') and phase_suffix in k[0]
                   and k[1].startswith(level + '.') == kernel)
    out = {}
    for tag in ('residual', 'jacobian'):
        inc = ph_delta('/' + tag)
        kern = ps_delta('/' + tag, True)
        other = ps_delta('/' + tag, False)
        out[tag + ' kernel'], out[tag + ' other C'], out[tag + ' Python'] = kern, other, inc - kern - other
    out['linear prepare'] = ph_delta('/linear prepare')
    out['linear solve'] = ph_delta('/linear solve')
    return out


def instrument_newton(ns):
    """Each model's level Newton solver: a phase ``newton <model>`` and a per-call log.

    Wrapped per instance, so the subclasses (CLSVOFNewton, TwoStageNewton, ...) are
    covered; the linear solver phases nest under it and are charged to its model.
    """
    for model, mls in zip(ns.modelList, ns.nlsList):
        for solver in mls.solverList:
            name, level = model.name, solver.F.name
            f = solver.solve
            log = newton_log.setdefault(name, [])

            def wrapper(*args, f=f, solver=solver, name=name, level=level, log=log, **kwargs):
                path = (acc.phase_stack[-1] + '/' if acc.phase_stack[-1] != '(outside)' else '') \
                    + 'newton ' + name
                ph0 = {k: v[1] for k, v in acc.phases.items() if k.startswith(path + '/')}
                ps0 = {k: v[1] for k, v in acc.passes.items() if k[0].startswith(path + '/')}
                ev0, its0, ct0 = petsc_event_seconds(), ksp_counter['its'], ksp_counter['ctest']
                acc.phase_stack.append(path)
                t = acc.perf_counter()
                try:
                    return f(*args, **kwargs)
                finally:
                    dt = acc.perf_counter() - t
                    acc.phase_stack.pop()
                    s = acc.phases.setdefault(path, [0, 0.0])
                    s[0] += 1
                    s[1] += dt
                    ev = petsc_event_seconds()
                    extra = {e: ev[e] - ev0[e] for e in PETSC_EVENTS}
                    extra['ksp its'] = ksp_counter['its'] - its0
                    extra['convergence test'] = ksp_counter['ctest'] - ct0
                    log.append([solver.its, dt, newton_breakdown(path, level, ph0, ps0), extra])
            solver.solve = wrapper


def newton_table(warmup):
    """Per model: per-call rows from the slowest rank, then per-iteration figures."""
    names = sorted(set(sum(mpi.allgather(list(newton_log)), [])))
    table = {}
    for name in names:
        rows = newton_log.get(name, [])
        n = mpi.allreduce(len(rows), op=__import__('mpi4py').MPI.MAX)
        nc = len(NEWTON_CATEGORIES)
        a = np.zeros((n, 2 + nc + len(NEWTON_EXTRAS)))
        for i, (its, dt, cat, extra) in enumerate(rows):
            a[i, 0], a[i, 1] = its, dt
            a[i, 2:1 + nc] = [cat[c] for c in NEWTON_CATEGORIES[:-1]]
            a[i, 2 + nc:] = [extra[e] for e in NEWTON_EXTRAS]
        #every rank makes the same calls; take each call's row from the rank that took longest,
        #so the categories of a row add up to its total
        a_all = np.array(mpi.allgather(a))
        slowest = a_all[:, :, 1].argmax(axis=0) if n else []
        a = a_all[slowest, np.arange(n)] if n else a
        steady = a[warmup:]
        its = steady[:, 0].sum()
        per_it = {c: (steady[:, 2 + j].sum() / its if its else 0.0)
                  for j, c in enumerate(NEWTON_CATEGORIES)}
        per_it['newton Python'] = (steady[:, 1].sum() - steady[:, 2:1 + nc].sum()) / its if its else 0.0
        extras = {e: (steady[:, 2 + nc + j].sum() / its if its else 0.0)
                  for j, e in enumerate(NEWTON_EXTRAS)}
        ksp = steady[:, 2 + nc]
        with_its = steady[steady[:, 0] > 0]
        table[name] = {'calls': int(n), 'iterations': int(a[:, 0].sum()),
                       'calls_steady': int(len(steady)), 'iterations_steady': int(its),
                       'seconds_steady': float(steady[:, 1].sum()),
                       'per_iteration_mean': per_it,
                       'per_iteration_extras_mean': extras,
                       'ksp_its_per_newton_min_max': [float(np.min(ksp / np.maximum(steady[:, 0], 1)))
                                                      if len(steady) else 0.0,
                                                      float(np.max(ksp / np.maximum(steady[:, 0], 1)))
                                                      if len(steady) else 0.0],
                       'per_iteration_total_mean': float(steady[:, 1].sum() / its) if its else 0.0,
                       'per_iteration_total_median': float(np.median(with_its[:, 1] / with_its[:, 0]))
                       if len(with_its) else 0.0,
                       'rows': a.tolist()}
    return table


def load_case(case_path, context, superlu=None, params=''):
    from proteus import Context, Profiling
    from proteus.cli.run import _load
    if superlu is not None or params:
        from proteus.TwoPhaseFlow.TwoPhaseFlowProblem import TwoPhaseFlowProblem
        initializeAll = TwoPhaseFlowProblem.initializeAll

        def initialize_with_solver_choice(self):
            if superlu is not None:
                self.SystemNumerics.useSuperlu = bool(int(superlu))
            for item in params.split():
                target, _, value = item.partition('=')
                model, _, attr = target.partition('.')
                for key, m in self.SystemPhysics.modelDict.items():
                    if model in (key, m.name):
                        setattr(m, attr, value)
            return initializeAll(self)
        TwoPhaseFlowProblem.initializeAll = initialize_with_solver_choice

    class Args(object):
        inputs = [case_path]
        petsc_file = None
        pumi = False
    Context.contextOptionsString = context
    return _load(Args(), Profiling.logEvent)


def apply_solver_overrides(opts, pList, nList):
    from petsc4py import PETSc
    db = PETSc.Options()
    tokens = opts['petsc'].split()
    i = 0
    while i < len(tokens):
        key = tokens[i]
        if key.startswith('!'):
            db.delValue(key[1:].lstrip('-'))
            i += 1
            continue
        value = tokens[i + 1] if i + 1 < len(tokens) and not tokens[i + 1].startswith(('-', '!')) else None
        db.setValue(key.lstrip('-'), value if value is not None else True)
        i += 2 if value is not None else 1
    if opts['ctest'] is not None:
        for p, n in zip(pList, nList):
            if p.name == 'rans2p' or getattr(n, 'linear_solver_options_prefix', '') == 'rans2p_':
                n.linearSolverConvergenceTest = opts['ctest']
    for item in opts['n'].split():
        target, _, value = item.partition('=')
        model, _, attr = target.partition('.')
        for p, n in zip(pList, nList):
            if p.name == model:
                setattr(n, attr, type(getattr(n, attr))(value) if getattr(n, attr) is not None else float(value))
    if rank == 0 and (tokens or opts['ctest'] or opts['n']):
        print("OVERRIDES petsc=%r ctest=%r n=%r" % (opts['petsc'], opts['ctest'], opts['n']))


def write_solution(ns, prefix):
    arrays = {}
    for m in ns.modelList:
        lm = m.levelModelList[-1]
        for ci, u in lm.u.items():
            arrays['%s_%d' % (m.name, ci)] = np.array(u.dof)
        #the integral of the first component over the owned elements (VOF: the water volume)
        if ('u', 0) in lm.q and 'dV' in lm.q:
            nE = lm.mesh.nElements_owned
            local = float((lm.q[('u', 0)][:nE] * lm.q['dV'][:nE]).sum())
            total = mpi.allreduce(local)
            if rank == 0:
                print("INTEGRAL %s %.15e" % (m.name, total))
    np.savez('%s.%d.npz' % (prefix, rank), **arrays)


def main(argv):
    opts = parse(argv)
    from proteus import NumericalSolution, Profiling, runtime
    from proteus.iproteus import opts as runopts  # the options object NS_base reads
    Profiling.procID = rank
    Profiling.logLevel = 1
    Profiling.verbose = False
    runopts.dataDir = opts['datadir']
    instrument_classes()
    instrument_linear_solvers()
    so, pList, nList, sList, two_phase = load_case(opts['case'], opts['context'], opts['superlu'], opts['params'])
    apply_solver_overrides(opts, pList, nList)
    ceilings = acc.load_ceilings(opts['ceilings'], size)
    simFlagsList = [{"simulationName": p.name, "simulationNameProc": p.name + repr(rank),
                     "dataFile": p.name + repr(rank) + ".dat",
                     "components": list(range(p.coefficients.nc))} for p in pList]
    acc.reset()
    mpi.Barrier()
    with acc.phase('build'):
        ns = NumericalSolution.NS_base(so, pList, nList, sList, runopts, simFlagsList, two_phase)
        mpi.Barrier()
    instrument_kernels(ns)
    instrument_newton(ns)
    elements = {m.name: m.levelModelList[-1].mesh.nElements_global for m in ns.modelList}
    dof = {m.name: sum(u.femSpace.dofMap.nDOF_all_processes for u in m.levelModelList[-1].u.values())
           for m in ns.modelList}
    with acc.phase('solve'):
        ns.calculateSolution(so.name)
        mpi.Barrier()
    if opts['solution']:
        write_solution(ns, opts['solution'])
    ph, ps = acc.collect(mpi)
    newton = newton_table(int(opts['warmup']))
    elements_max = {k: mpi.allreduce(v, op=__import__('mpi4py').MPI.MAX) for k, v in elements.items()}
    levels = {m.name: m.levelModelList[-1].name for m in ns.modelList}
    result = {'case': os.path.basename(opts['case']), 'context': opts['context'], 'ranks': size,
              'models': [m.name for m in ns.modelList], 'level_names': levels,
              'elements_per_rank_max': elements_max,
              'dof_global': dof, 'ceilings': ceilings,
              't_build': ph['build']['seconds'], 't_solve': ph['solve']['seconds'],
              'phases': ph, 'passes': {'%s | %s' % k: v for k, v in ps.items()},
              'warmup': int(opts['warmup']), 'superlu': opts['superlu'], 'newton': newton}
    counted = acc.perf_gate.counts(mpi)
    if rank == 0 and counted is not None:
        print("PERFGATE pass=%s calls_counted_per_rank=%s" % (acc.perf_gate.pattern, ",".join(map(str, counted))))
    if rank == 0:
        report(result, ph, ps)
        report_newton(result)
        out = opts['out'] or 'model_baseline_%s_%s_%d.json' % (
            os.path.splitext(result['case'])[0], platform.node().split('.')[0], size)
        import json
        with open(out, 'w') as f:
            json.dump(result, f, indent=1, default=acc.json_default)
        print('wrote', out)


def report(res, ph, ps):
    c = res['ceilings']
    print("RESULT case=%s ranks=%d models=%s t_build=%.2f t_solve=%.2f context=%r"
          % (res['case'], res['ranks'], ','.join(res['models']), res['t_build'], res['t_solve'],
             res['context']))
    if c:
        print("CEILING stream_per_rank=%.1f GB/s (P=%d measured) fma_per_rank=%.1f GFLOP/s from %s"
              % (c['stream_per_rank_gbps'], c['procs_measured'], c['fma_per_rank_gflops'], c['file']))
    #where the solve goes
    solve = ph['solve']['seconds']
    assembly = sum(v['seconds'] for k, v in ph.items()
                   if (k.endswith('/residual') or k.endswith('/jacobian')) and 'model ' in k
                   and k.count('model ') == 1)
    linear = sum(v['seconds'] for k, v in ph.items() if k.endswith('linear solve') or k.endswith('linear prepare'))
    print("SOLVE %.3f s: assembly (residuals+Jacobians, every model) %.3f s (%.0f%%), linear prepare+solve "
          "%.3f s (%.0f%%), rest %.3f s" % (solve, assembly, 100 * assembly / solve, linear,
                                           100 * linear / solve, solve - assembly - linear))
    #per model: residual and jacobian phases (any depth) and their kernel passes
    for name in res['models']:
        n_el = res['elements_per_rank_max'][name]
        level = res['level_names'][name]
        for tag in ('residual', 'jacobian'):
            paths = [k for k in ph if k.endswith('model %s/%s' % (level, tag))]
            calls = sum(ph[k]['calls'] for k in paths)
            inc = sum(ph[k]['seconds'] for k in paths)
            if not calls:
                continue
            kernel = [(k, v) for k, v in ps.items()
                      if any(k[0] == p or k[0].startswith(p + '/') for p in paths)
                      and k[1].startswith(level + '.')]
            k_secs = sum(v['seconds'] for _, v in kernel)
            k_bytes = sum(v['bytes'] for _, v in kernel)
            others = sum(acc.c_time_under(p, ps) for p in paths) - k_secs
            gbps = k_bytes / k_secs / 1e9 if k_secs else 0.0
            frac = ' (%3.0f%% of stream)' % (100 * gbps / c['stream_per_rank_gbps']) if c and k_secs else ''
            print("MODEL %-12s %-8s %4d calls %8.3f s  kernel %8.3f s = %6.0f ns/el/call at %6.1f GB/s%s  "
                  "other C %7.3f s  Python %7.3f s (%2.0f%%)"
                  % (name, tag, calls, inc, k_secs, 1e9 * k_secs / max(calls, 1) / n_el, gbps, frac,
                     others, inc - k_secs - others, 100 * (inc - k_secs - others) / inc))
    print("-- C passes by time")
    for (path, name), v in sorted(ps.items(), key=lambda kv: -kv[1]['seconds'])[:25]:
        gbps = v['bytes'] / v['seconds'] / 1e9 if v['seconds'] else 0.0
        print("PASS %-44s %-40s %5d calls %8.3f s %9.1f MB %7.1f GB/s"
              % (path[-44:], name[-40:], v['calls'], v['seconds'], v['bytes'] / 1e6, gbps))
    sys.stdout.flush()


def report_newton(res):
    """Per Newton iteration, per model, in ms; then per system step."""
    newton = res['newton']
    print("-- per Newton iteration, ms (steady state: first %d solves per model left out)" % res['warmup'])
    print("NEWTONHDR %-18s %5s %6s %8s | %-25s | %-25s | %7s %7s %7s | %7s %7s"
          % ('model', 'calls', 'its', 'its/call', 'residual kern/C/Py', 'jacobian kern/C/Py',
             'lin.prep', 'lin.solv', 'nwt.Py', 'mean', 'median'))
    for name in res['models']:
        t = newton.get(name)
        if not t or not t['iterations_steady']:
            continue
        m = {k: 1e3 * v for k, v in t['per_iteration_mean'].items()}
        print("NEWTON %-18s %5d %6d %8.2f | %7.2f %7.2f %7.2f   | %7.2f %7.2f %7.2f   | %7.2f %7.2f %7.2f | %7.2f %7.2f"
              % (name, t['calls_steady'], t['iterations_steady'], t['iterations_steady'] / t['calls_steady'],
                 m['residual kernel'], m['residual other C'], m['residual Python'],
                 m['jacobian kernel'], m['jacobian other C'], m['jacobian Python'],
                 m['linear prepare'], m['linear solve'], m['newton Python'],
                 1e3 * t['per_iteration_total_mean'], 1e3 * t['per_iteration_total_median']))
    print("-- inside the linear solves, per Newton iteration (ms; KSP its per Newton iteration, min-max over calls)")
    print("KSPHDR %-18s %8s %11s | %8s %8s %8s %8s %8s %8s %8s %8s"
          % ('model', 'ksp its', 'min-max', 'PCSetUp', 'PCSetUpOB', 'PCApply', 'MatSolve', 'MatMult', 'GMRESorth',
             'ScatEnd', 'ctest'))
    for name in res['models']:
        t = newton.get(name)
        if not t or not t['iterations_steady']:
            continue
        x = t['per_iteration_extras_mean']
        print("KSP %-18s %8.1f %5.0f-%-5.0f | %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f"
              % (name, x['ksp its'], t['ksp_its_per_newton_min_max'][0], t['ksp_its_per_newton_min_max'][1],
                 *(1e3 * x[e] for e in NEWTON_EXTRAS[1:])))
    #system steps: the first model is solved once per step
    first = newton.get(res['models'][0])
    if first:
        steps = first['calls']
        in_newton = sum(row[1] for t in newton.values() for row in t['rows'])
        driver = res['t_solve'] - in_newton
        print("STEP %d system steps (calls of %s's solver): %.3f s in Newton solves, %.3f s driver outside "
              "them = %.1f ms per step" % (steps, res['models'][0], in_newton, driver, 1e3 * driver / max(steps, 1)))
    sys.stdout.flush()


if __name__ == '__main__':
    main(sys.argv[1:])
