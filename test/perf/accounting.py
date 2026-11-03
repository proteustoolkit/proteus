"""Wall-clock accounting of C passes and Python phases, shared by the baseline scripts.

A *pass* is one call into a compiled kernel: timed, and charged the bytes of the arrays it
was handed (a lower bound on the traffic it moved). A *phase* is a Python method that passes
run inside: its inclusive time minus the time of the passes under it is the *Python
remainder* -- loops over points, per-point callbacks, dict lookups, temporaries. Phases nest
and passes are attributed to the innermost phase, so a phase's remainder can be read at any
depth.

Everything is wall clock; no profiler is loaded. Times are reduced as the max over ranks
by :func:`collect`.
"""
import json
import os
import platform
import time

import numpy as np

perf_counter = time.perf_counter

phase_stack = ['(outside)']
passes = {}    # (phase path, pass name) -> [calls, seconds, bytes]
phases = {}    # phase path -> [calls, seconds]


def reset():
    passes.clear()
    phases.clear()


def nbytes_of(args, kwargs):
    """The bytes of the arrays among ``args``/``kwargs``, including those inside an
    ``ArgumentsDict`` wrapped by :class:`TrackingArgumentsDict`."""
    n = 0
    for a in list(args) + list(kwargs.values()):
        if isinstance(a, np.ndarray):
            n += a.nbytes
        else:
            n += getattr(a, 'tracked_nbytes', 0)
    return n


def record_pass(key, dt, nb):
    s = passes.setdefault((phase_stack[-1], key), [0, 0.0, 0])
    s[0] += 1
    s[1] += dt
    s[2] += nb


class PerfGate(object):
    """Counts hardware events around one named pass, under ``perf stat --control``.

    ``perf stat`` over a whole run counts the build, the solve and the Python between
    kernels along with the kernel; the FLOP rate of *one pass* needs the counters on only
    while that pass runs. ``perf stat -D -1 --control fifo:CTL,ACK`` starts with counting
    off and obeys ``enable``/``disable`` written to CTL, answering ``ack`` on ACK. Set
    ``PROTEUS_PERF_CTL`` and ``PROTEUS_PERF_ACK`` to the two fifos and ``PROTEUS_PERF_PASS``
    to a substring of the pass name (``updateDiffusionJacobian_weak_sd``); the gate then
    counts every call of the matching passes and records how many it counted. perf's
    totals divided by :attr:`calls` are the events per call. Off unless all three are set.
    """
    def __init__(self):
        self.pattern = os.environ.get('PROTEUS_PERF_PASS')
        self.ctl = self.ack = None
        self.calls = 0
        if self.pattern and os.environ.get('PROTEUS_PERF_CTL') and os.environ.get('PROTEUS_PERF_ACK'):
            self.ctl = open(os.environ['PROTEUS_PERF_CTL'], 'w', buffering=1)
            self.ack = open(os.environ['PROTEUS_PERF_ACK'], 'r')

    def matches(self, key):
        return self.ctl is not None and self.pattern in key

    def _send(self, word):
        self.ctl.write(word + '\n')
        self.ctl.flush()
        self.ack.readline()

    def enable(self):
        self._send('enable')
        self.calls += 1

    def disable(self):
        self._send('disable')


perf_gate = PerfGate()


def wrap_pass(f, key):
    """A timed, byte-counted version of the callable ``f``."""
    gated = perf_gate.matches(key)

    def wrapper(*args, **kwargs):
        nb = nbytes_of(args, kwargs)
        if gated:
            perf_gate.enable()
        t = perf_counter()
        try:
            return f(*args, **kwargs)
        finally:
            dt = perf_counter() - t
            if gated:
                perf_gate.disable()
            record_pass(key, dt, nb)
    wrapper.__name__ = getattr(f, '__name__', key)
    return wrapper


def wrap_module(mod, label):
    """Every public callable of a Cython kernel module becomes a pass."""
    for name in dir(mod):
        if name.startswith('_'):
            continue
        f = getattr(mod, name)
        if not callable(f) or isinstance(f, type):
            continue
        setattr(mod, name, wrap_pass(f, label + '.' + name))


def wrap_phase(owner, name, label):
    """A method becomes a phase; ``label`` is a string or a function of ``self``."""
    f = getattr(owner, name)

    def wrapper(self, *args, **kwargs):
        tag = label(self) if callable(label) else label
        path = phase_stack[-1] + '/' + tag if phase_stack[-1] != '(outside)' else tag
        phase_stack.append(path)
        t = perf_counter()
        try:
            return f(self, *args, **kwargs)
        finally:
            dt = perf_counter() - t
            phase_stack.pop()
            s = phases.setdefault(path, [0, 0.0])
            s[0] += 1
            s[1] += dt
    setattr(owner, name, wrapper)


class phase(object):
    """``with phase('build'): ...`` -- a phase around a block."""
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        phase_stack.append(self.label)
        self.t = perf_counter()

    def __exit__(self, *exc):
        dt = perf_counter() - self.t
        phase_stack.pop()
        s = phases.setdefault(self.label, [0, 0.0])
        s[0] += 1
        s[1] += dt


class KernelProxy(object):
    """Stands in for a model's compiled kernel object: its ``calculate*`` methods are passes.

    The mprans models hold their C++ kernel as an attribute (``self.rans2p``,
    ``self.clsvof``, ...) and call ``self.rans2p.calculateResidual(argsDict)``; a cdef
    or pybind11 class's methods cannot be replaced, so the object is.
    """
    def __init__(self, kernel, label):
        object.__setattr__(self, '_kernel', kernel)
        object.__setattr__(self, '_label', label)

    def __getattr__(self, name):
        attr = getattr(self._kernel, name)
        if callable(attr) and name.startswith('calculate'):
            return wrap_pass(attr, self._label + '.' + name)
        return attr

    def __setattr__(self, name, value):
        setattr(self._kernel, name, value)


def tracking_arguments_dict(base):
    """A subclass of ``cArgumentsDict.ArgumentsDict`` that counts the bytes put into it."""
    class TrackingArgumentsDict(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            object.__setattr__(self, 'tracked_nbytes', 0)

        def __setitem__(self, key, value):
            if isinstance(value, np.ndarray):
                object.__setattr__(self, 'tracked_nbytes', self.tracked_nbytes + value.nbytes)
            super().__setitem__(key, value)
    TrackingArgumentsDict.__name__ = 'ArgumentsDict'
    return TrackingArgumentsDict


def collect(mpi):
    """Phase and pass tables, max over ranks of every time (bytes and calls from the max rank)."""
    from mpi4py import MPI

    def reduce_max(x):
        return mpi.allreduce(x, op=MPI.MAX)
    keys = sorted(set(sum(mpi.allgather(list(phases)), [])))
    ph = {}
    for k in keys:
        calls, secs = phases.get(k, [0, 0.0])
        ph[k] = {'calls': reduce_max(calls), 'seconds': reduce_max(secs)}
    keys = sorted(set(sum(mpi.allgather(list(passes)), [])))
    ps = {}
    for k in keys:
        calls, secs, nb = passes.get(k, [0, 0.0, 0])
        ps[k] = {'calls': reduce_max(calls), 'seconds': reduce_max(secs), 'bytes': reduce_max(nb)}
    return ph, ps


def c_time_under(path, ps):
    """Seconds in C passes attributed to this phase or any phase nested in it."""
    return sum(v['seconds'] for (p, _), v in ps.items() if p == path or p.startswith(path + '/'))


def load_ceilings(path, ranks):
    """The node's measured STREAM and FMA ceilings per rank (machine_ceilings.py), or None."""
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


def json_default(o):
    return o.item() if hasattr(o, 'item') else str(o)
