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
* where a time step goes: assembly, the linear solves, the rest.

Usage: model_baseline_mpi.py CASE.py [context='k=v ...'] [ceilings=FILE] [out=FILE] [datadir=DIR]

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
    opts = {'case': argv[0], 'context': None, 'ceilings': None, 'out': None, 'datadir': '.'}
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
    acc.wrap_phase(NonlinearSolvers.Newton, 'solve', 'newton')
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


def load_case(case_path, context):
    from proteus import Context, Profiling
    from proteus.cli.run import _load

    class Args(object):
        inputs = [case_path]
        petsc_file = None
        pumi = False
    Context.contextOptionsString = context
    return _load(Args(), Profiling.logEvent)


def main(argv):
    opts = parse(argv)
    from proteus import NumericalSolution, Profiling, runtime
    from proteus.iproteus import opts as runopts  # the options object NS_base reads
    Profiling.procID = rank
    Profiling.logLevel = 1
    Profiling.verbose = False
    runopts.dataDir = opts['datadir']
    instrument_classes()
    so, pList, nList, sList, two_phase = load_case(opts['case'], opts['context'])
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
    elements = {m.name: m.levelModelList[-1].mesh.nElements_global for m in ns.modelList}
    dof = {m.name: sum(u.femSpace.dofMap.nDOF_all_processes for u in m.levelModelList[-1].u.values())
           for m in ns.modelList}
    with acc.phase('solve'):
        ns.calculateSolution(so.name)
        mpi.Barrier()
    ph, ps = acc.collect(mpi)
    elements_max = {k: mpi.allreduce(v, op=__import__('mpi4py').MPI.MAX) for k, v in elements.items()}
    levels = {m.name: m.levelModelList[-1].name for m in ns.modelList}
    result = {'case': os.path.basename(opts['case']), 'context': opts['context'], 'ranks': size,
              'models': [m.name for m in ns.modelList], 'level_names': levels,
              'elements_per_rank_max': elements_max,
              'dof_global': dof, 'ceilings': ceilings,
              't_build': ph['build']['seconds'], 't_solve': ph['solve']['seconds'],
              'phases': ph, 'passes': {'%s | %s' % k: v for k, v in ps.items()}}
    if rank == 0 and acc.perf_gate.ctl is not None:
        print("PERFGATE pass=%s calls_counted=%d" % (acc.perf_gate.pattern, acc.perf_gate.calls))
    if rank == 0:
        report(result, ph, ps)
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


if __name__ == '__main__':
    main(sys.argv[1:])
