"""The on-node baseline: what one assembly costs, pass by pass, against what the node can do.

C30's measurement. Poisson on a regular hexahedral (Q1) or tetrahedral (P1) grid, one level,
run under mpiexec. After the solve, the finest level's residual and Jacobian are assembled
``repeat`` more times and every C pass they make is accounted for:

* **time**, as the max over ranks, in ns per element and in DOF/s;
* **bytes**: the sizes of the arrays each pass is handed. A pass that reads each array once moves
  at least this much, so bytes/time is its achieved bandwidth and, against the node's measured
  STREAM triad (``machine_ceilings.py``), the fraction of the bandwidth bound it reaches;
* **the Python remainder**: the time a phase spends outside any C pass -- loops over
  quadrature points, per-point coefficient and boundary-condition callbacks, dict lookups,
  temporaries. That is cost kinds 1 and 3 of the 2.0 task list, measured rather than inferred.

FLOP rates are not computed here: the arithmetic a pass does is not visible from Python. On Linux
with hardware counters available (``perf_event_paranoid`` <= 1) run the script under
``perf stat -e instructions,cycles,fp_ret_sse_avx_ops.all`` (AMD) or the Intel equivalent and
divide by the kernel times printed here. Everything is wall clock; no profiler is loaded.

Usage: kernel_baseline_mpi.py KIND NN[,NN..] [levels=L] [quad=N] [repeat=K] [ceilings=FILE] [out=FILE]

KIND      hex or tet
NN        nodes per axis; several values run in turn, which is the cache sweep (ns per element
          against working set)
quad=N    element quadrature order (default: what the test numerics ask for: 2 on hexahedra)
repeat=K  assemblies timed after the solve (default 5; the min over repeats is reported)
ceilings= the JSON machine_ceilings.py wrote (default: ceilings_<host>.json if present)
out=      JSON results file (default baseline_<kind>_<host>_<ranks>.json)

Prints one RESULT line per NN, PHASE lines (inclusive times, C time and Python remainder) and
PASS lines (the C passes, sorted by time).
"""
import json
import os
import platform
import resource
import sys
import time

import numpy as np
from proteus import Comm
comm = Comm.init()
from mpi4py import MPI
from proteus import (MeshTools, defaults, default_n, default_s, default_so, NumericalSolution,
                     LinearSolvers, NonlinearSolvers, Transport)
from proteus import (cfemIntegrals, cnumericalFlux, csubgridError, cshockCapturing, csparsity,
                     csmoothers, ctimeIntegration, cpostprocessing, cmeshTools, cpartitioning)
from proteus.iproteus import opts as runopts

rank, size = comm.rank(), comm.size()
mpi = comm.comm.tompi4py()
perf_counter = time.perf_counter

# accounting: test/perf/accounting.py, shared with model_baseline_mpi.py
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import accounting as acc  # noqa: E402
from accounting import (phase, wrap_module, wrap_phase, reset, c_time_under,  # noqa: E402
                        load_ceilings)
passes, phases = acc.passes, acc.phases


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2 if sys.platform == 'darwin' else 1024.0)


for mod, label in ((cfemIntegrals, 'femIntegrals'), (cnumericalFlux, 'numericalFlux'),
                   (csubgridError, 'subgridError'), (cshockCapturing, 'shockCapturing'),
                   (csparsity, 'sparsity'), (csmoothers, 'smoothers'),
                   (ctimeIntegration, 'timeIntegration'), (cpostprocessing, 'postprocessing'),
                   (cmeshTools, 'meshTools'), (cpartitioning, 'partitioning')):
    wrap_module(mod, label)

wrap_phase(Transport.OneLevelTransport, 'getResidual', 'residual')
wrap_phase(Transport.OneLevelTransport, 'calculateCoefficients', 'coefficients')
wrap_phase(Transport.OneLevelTransport, 'calculateElementCoefficients', 'element')
wrap_phase(Transport.OneLevelTransport, 'calculateExteriorElementBoundaryCoefficients', 'exterior boundary')
wrap_phase(Transport.OneLevelTransport, 'calculateElementResidual', 'element residual')
wrap_phase(Transport.OneLevelTransport, 'getJacobian', 'jacobian')
wrap_phase(Transport.OneLevelTransport, 'calculateElementJacobian', 'element jacobian')
wrap_phase(Transport.OneLevelTransport, 'calculateExteriorElementBoundaryJacobian', 'exterior boundary jacobian')
wrap_phase(LinearSolvers.KSP_petsc4py, 'prepare', 'linear prepare')
wrap_phase(LinearSolvers.KSP_petsc4py, 'solve', 'linear solve')
wrap_phase(NonlinearSolvers.Newton, 'solve', 'newton')

# --------------------------------------------------------------------------------------------
# the problem
# --------------------------------------------------------------------------------------------


def parse(argv):
    if len(argv) < 2 or argv[0] not in ('hex', 'tet'):
        raise SystemExit(__doc__)
    opts = {'kind': argv[0], 'nn': [int(v) for v in argv[1].split(',')], 'levels': 1, 'quad': None,
            'repeat': 5, 'ceilings': None, 'out': None}
    for arg in argv[2:]:
        key, _, value = arg.partition('=')
        if key not in opts:
            raise SystemExit(__doc__)
        opts[key] = int(value) if key in ('levels', 'quad', 'repeat') else value
    return opts


def problem(kind, nn, levels, quad):
    testroot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = defaults.load_physics('poisson_3d_p', os.path.join(testroot, 'ci'))
    n = defaults.load_numerics('poisson_3d_c0q1_n' if kind == 'hex' else 'poisson_3d_c0p1_n',
                               os.path.join(testroot, 'ci'))
    n.nnx = n.nny = n.nnz = nn
    n.nLevels = levels
    if quad is not None:
        Q = default_n.CubeGaussQuadrature if kind == 'hex' else default_n.SimplexGaussQuadrature
        n.elementQuadrature = Q(p.nd, quad)
        n.elementBoundaryQuadrature = Q(p.nd - 1, quad)
    n.parallelPartitioningType = MeshTools.MeshParallelPartitioningTypes.node
    n.nLayersOfOverlapForParallel = 1
    n.multilevelLinearSolver = LinearSolvers.KSP_petsc4py
    n.levelLinearSolver = LinearSolvers.KSP_petsc4py
    n.petscOptions = {'ksp_type': 'cg', 'pc_type': 'gamg', 'ksp_rtol': '1e-8'}
    n.nonlinearSolverConvergenceTest = 'its'
    n.levelNonlinearSolverConvergenceTest = 'its'
    n.useEisenstatWalker = False
    so = default_so
    so.name = p.name = 'kernelbaseline'
    so.sList = [default_s]
    return p, n, so


def reduce_max(x):
    return mpi.allreduce(x, op=MPI.MAX)


def collect():
    return acc.collect(mpi)


def c_time_under(path, ps):
    """Seconds in C passes attributed to this phase or any phase nested in it."""
    return sum(v['seconds'] for (p, _), v in ps.items() if p == path or p.startswith(path + '/'))


def load_ceilings(path, ranks):
    if path is None:
        path = 'ceilings_%s.json' % platform.node().split('.')[0]
        if not os.path.exists(path):
            return None
    with open(path) as f:
        c = json.load(f)
    m = c['measurements']['proteus']
    procs = sorted(int(p) for p in m['triad'])
    p = min(procs, key=lambda q: (abs(q - ranks), q))
    dram = m['triad'][str(p)][-1]
    return {'file': path, 'procs_measured': p, 'stream_per_rank_gbps': dram['per_proc_min_gbps'],
            'stream_aggregate_gbps': dram['aggregate_gbps'],
            'fma_per_rank_gflops': m['fma'][str(p)]['per_proc_min_gflops']}


def run_one(opts, nn, ceilings):
    reset()
    runopts.logLevel = 1
    runopts.verbose = False
    p, n, so = problem(opts['kind'], nn, opts['levels'], opts['quad'])
    mpi.Barrier()
    rss0 = rss_mb()
    with phase('build'):
        ns = NumericalSolution.NS_base(so, [p], [n], so.sList, runopts)
        mpi.Barrier()
    with phase('solve'):
        ns.calculateSolution('kernelbaseline')
        mpi.Barrier()
    build_phases, build_passes = collect()
    rss_after = rss_mb()

    lm = ns.modelList[0].levelModelList[-1]
    nls = ns.nlsList[0].solverList[-1]
    J = nls.J
    u = np.zeros(lm.dim)
    r = np.zeros(lm.dim)
    elements = lm.mesh.nElements_global
    elements_owned = lm.mesh.nElements_owned
    dof_global = lm.u[0].femSpace.dofMap.nDOF_all_processes
    dof_local = lm.dim
    nq = lm.q[('x',)].shape[1] if ('x',) in lm.q else lm.q['x'].shape[1]
    q_bytes = sum(v.nbytes for v in lm.q.values() if isinstance(v, np.ndarray))

    # steady-state assemblies, timed one at a time; the min over repeats is the cost of the
    # assembly itself, the max over ranks the cost the solve waits for
    t_res, t_jac = [], []
    for _ in range(opts['repeat']):
        reset()
        mpi.Barrier()
        t = perf_counter()
        lm.getResidual(u, r)
        mpi.Barrier()
        t_res.append(perf_counter() - t)
        t = perf_counter()
        lm.getJacobian(J)
        mpi.Barrier()
        t_jac.append(perf_counter() - t)
    # the pass table from the last repeat
    kernel_phases, kernel_passes = collect()
    t_res = reduce_max(min(t_res))
    t_jac = reduce_max(min(t_jac))
    elements_max = reduce_max(elements)
    q_bytes_max = reduce_max(q_bytes)
    rss_after = reduce_max(rss_after)
    rss0 = reduce_max(rss0)

    result = {
        'kind': opts['kind'], 'nn': nn, 'levels': opts['levels'], 'ranks': size,
        'elements_per_rank_max': elements_max, 'elements_owned_rank0': elements_owned,
        'dof_global': dof_global, 'dof_per_rank_rank0': dof_local, 'quad_points': nq,
        't_build': build_phases['build']['seconds'], 't_solve': build_phases['solve']['seconds'],
        't_residual': t_res, 't_jacobian': t_jac,
        'ns_per_element_residual': 1e9 * t_res / elements_max,
        'ns_per_element_jacobian': 1e9 * t_jac / elements_max,
        'dof_per_s_residual': dof_global / t_res, 'dof_per_s_jacobian': dof_global / t_jac,
        'q_mb_per_rank_max': q_bytes_max / 1e6,
        'q_bytes_per_dof': q_bytes_max / max(dof_local, 1),
        'rss_import_mb': rss0, 'rss_after_solve_mb': rss_after,
        'ceilings': ceilings,
        'build': {'phases': build_phases, 'passes': {'%s | %s' % k: v for k, v in build_passes.items()}},
        'kernel': {'phases': kernel_phases, 'passes': {'%s | %s' % k: v for k, v in kernel_passes.items()}},
    }
    if rank == 0:
        report(result, kernel_phases, kernel_passes, build_phases, build_passes)
    return result


def report(res, kph, kps, bph, bps):
    c = res['ceilings']
    print("RESULT kind=%s ranks=%d nn=%d levels=%d elements_per_rank=%d dof=%d quad_points=%d "
          "t_build=%.3f t_solve=%.3f t_residual=%.4f t_jacobian=%.4f ns_per_element_residual=%.0f "
          "ns_per_element_jacobian=%.0f dof_per_s_residual=%.3g dof_per_s_jacobian=%.3g "
          "q_mb_per_rank=%.1f q_bytes_per_dof=%.0f rss_after_solve_mb=%.0f"
          % (res['kind'], res['ranks'], res['nn'], res['levels'], res['elements_per_rank_max'],
             res['dof_global'], res['quad_points'], res['t_build'], res['t_solve'], res['t_residual'],
             res['t_jacobian'], res['ns_per_element_residual'], res['ns_per_element_jacobian'],
             res['dof_per_s_residual'], res['dof_per_s_jacobian'], res['q_mb_per_rank_max'],
             res['q_bytes_per_dof'], res['rss_after_solve_mb']))
    if c:
        print("CEILING stream_per_rank=%.1f GB/s (P=%d measured) fma_per_rank=%.1f GFLOP/s from %s"
              % (c['stream_per_rank_gbps'], c['procs_measured'], c['fma_per_rank_gflops'], c['file']))
    print("-- one assembly (last repeat): phase, inclusive s, in C passes s, Python remainder s")
    for path in sorted(kph):
        inc = kph[path]['seconds']
        inc_c = c_time_under(path, kps)
        print("PHASE %-52s %3d calls %9.4f s  C %9.4f s  Python %9.4f s (%3.0f%%)"
              % (path, kph[path]['calls'], inc, inc_c, inc - inc_c, 100.0 * (inc - inc_c) / inc if inc else 0))
    print("-- one assembly: C passes by time")
    rows = sorted(kps.items(), key=lambda kv: -kv[1]['seconds'])
    for (path, name), v in rows:
        gbps = v['bytes'] / v['seconds'] / 1e9 if v['seconds'] else 0.0
        frac = ' %4.0f%% of stream' % (100.0 * gbps / c['stream_per_rank_gbps']) if c else ''
        print("PASS %-26s %-56s %3d calls %9.4f s %9.1f MB %7.1f GB/s%s"
              % (path, name, v['calls'], v['seconds'], v['bytes'] / 1e6, gbps, frac))
    print("-- build and first solve: phases")
    for path in sorted(bph):
        inc = bph[path]['seconds']
        inc_c = c_time_under(path, bps)
        print("BUILDPHASE %-47s %3d calls %9.4f s  C %9.4f s  Python %9.4f s"
              % (path, bph[path]['calls'], inc, inc_c, inc - inc_c))
    rows = sorted(bps.items(), key=lambda kv: -kv[1]['seconds'])[:12]
    for (path, name), v in rows:
        gbps = v['bytes'] / v['seconds'] / 1e9 if v['seconds'] else 0.0
        print("BUILDPASS %-21s %-56s %3d calls %9.4f s %9.1f MB %7.1f GB/s"
              % (path, name, v['calls'], v['seconds'], v['bytes'] / 1e6, gbps))
    sys.stdout.flush()


def main(argv):
    opts = parse(argv)
    ceilings = load_ceilings(opts['ceilings'], size)
    results = [run_one(opts, nn, ceilings) for nn in opts['nn']]
    if rank == 0 and acc.perf_gate.ctl is not None:
        print("PERFGATE pass=%s calls_counted=%d" % (acc.perf_gate.pattern, acc.perf_gate.calls))
    if rank == 0:
        out = opts['out'] or 'baseline_%s_%s_%d.json' % (opts['kind'], platform.node().split('.')[0], size)
        with open(out, 'w') as f:
            json.dump({'host': platform.node(), 'date': time.strftime('%Y-%m-%d'), 'results': results}, f,
                      indent=1, default=acc.json_default)
        print('wrote', out)


if __name__ == '__main__':
    main(sys.argv[1:])
