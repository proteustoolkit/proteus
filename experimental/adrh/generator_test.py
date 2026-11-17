"""The generated VOF and NCLS functors (generated.h, module adrh_gen) against the hand-written ones (vof.h, ncls.h, module adrh_proto).

  python generator_test.py [fixtures...]            serial; `mpiexec -n 2 python generator_test.py` adds the partitioned check
Needs both modules built (`sh build.sh; sh build.sh gen`). It does not need ymf: the generator's inputs are the JSON in generator/inputs.

1. generated.h is what generator/generate.py emits from generator/inputs/*.json (regeneration check).
2. On the damBreak fixture (proteus's own arrays) and on random data (variable porosity, 3D tets): residual, Jacobian,
   shock-capturing viscosity and CFL of the generated against the hand-written functor, the largest difference relative to the
   largest value and the share of entries that are bitwise equal.
3. Partition independence: each rank assembles its share of the elements with the generated functor; the sum over ranks agrees
   with the serial hand-written assembly.
4. What the generator refuses (RDLS, MCorr, flow read from the same spec): each says what the engine's scalar template lacks.
5. One run-time comparison: the element kernels on the fixture tiled to ~73 000 elements, minimum of 7 repeats.
"""
import glob
import importlib
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
hand, gen = importlib.import_module('adrh_proto'), importlib.import_module('adrh_gen')
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:                                   # serial only
    comm = None
RANK, SIZE = (comm.Get_rank(), comm.Get_size()) if comm else (0, 1)
OK = True


def say(*a):
    if RANK == 0:
        print(*a)


def check(name, cond):
    global OK
    OK &= bool(cond)
    say('  %-62s %s' % (name, 'ok' if cond else 'FAIL'))


def compare(name, a, b, tol=1e-13):
    """a, b: tuples of arrays; largest |a-b| / largest |b| per array and the bitwise-equal share, over all arrays."""
    worst, same, total = 0.0, 0, 0
    for x, y in zip(a, b):
        x, y = np.asarray(x, float), np.asarray(y, float)
        worst = max(worst, np.abs(x - y).max() / max(np.abs(y).max(), 1e-300))
        same += int((x == y).sum())
        total += x.size
    say('  %-62s %.2e  bitwise equal %5.1f%%  %s' % (name, worst, 100.0 * same / total, 'ok' if worst <= tol else 'FAIL'))
    global OK
    OK &= worst <= tol


def regeneration():
    ins = ['vof_deck_2d', 'vof_1d', 'vof_2d', 'vof_3d', 'ncls_1d', 'ncls_2d', 'ncls_3d']
    out = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'generated_check_%d.h' % os.getpid())
    subprocess.check_call([sys.executable, os.path.join(HERE, 'generator', 'generate.py'), out] +
                          [os.path.join(HERE, 'generator', 'inputs', n + '.json') for n in ins])
    same = open(out).read() == open(os.path.join(HERE, 'generated.h')).read()
    os.remove(out)
    check('generated.h is the generator output for generator/inputs/*.json', same)


def tets(n=6):
    """A unit cube cut into n^3 cells of 6 tets each: xn (nE, 4, 3), and a 4-point degree-2 rule on the reference tet."""
    pts = np.array([[i, j, k] for k in range(n + 1) for j in range(n + 1) for i in range(n + 1)], float) / n
    idx = lambda i, j, k: i + (n + 1) * (j + (n + 1) * k)
    cells = []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                v = [idx(i + a, j + b, k + c) for c in (0, 1) for b in (0, 1) for a in (0, 1)]
                for t in ([0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7]):
                    cells.append([v[q] for q in t])
    cells = np.array(cells)
    a, b = 0.1381966011250105, 0.5854101966249685
    qp = np.array([[a, a, a], [b, a, a], [a, b, a], [a, a, b]])
    return pts, cells, qp, np.full(4, 1.0 / 24.0)


def random_case(rng, nd, n=6):
    if nd == 3:
        pts, l2g, qp, qw = tets(n)
    else:                                              # triangles of a unit square
        pts = np.array([[i, j, 0.0] for j in range(n * 4 + 1) for i in range(n * 4 + 1)]) / (n * 4)
        idx = lambda i, j: i + (n * 4 + 1) * j
        l2g = np.array([t for j in range(n * 4) for i in range(n * 4)
                        for t in ([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)], [idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)])])
        qp = np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]])
        qw = np.full(3, 1.0 / 6.0)
    nE, nQ = len(l2g), len(qw)
    xn = pts[l2g]
    u = rng.normal(size=(nE, nd + 1))
    d = dict(nd=nd, xn=xn, u=u, qp=qp, qw=qw, beta=rng.normal(size=(nE, nQ)), nl=np.abs(rng.normal(size=(nE, nQ))) * 1e-2,
             h=rng.uniform(0.05, 0.2, nE), v=rng.normal(size=(nE, nQ, nd)), por=rng.uniform(0.3, 1.0, (nE, nQ)))
    return d


def run(mod, kernel, d, por=True):
    f = np.concatenate([d['v'], d['por'][..., None]], axis=-1) if (kernel == 'vof' and por) else d['v']
    fn = getattr(mod, kernel + '_supg')
    out = fn(d['nd'], d['xn'], d['u'], d['qp'], d['qw'], f, d['beta'], d['nl'], d['h'], 2.5, 0.5, 1.0, 1.5, 0.0)
    return out[:4]


def fixture_case(path):
    f = dict(np.load(path))
    nS = f['mesh_l2g'].shape[1] - 1
    xn = f['mesh_dof'][f['mesh_l2g']]
    return f, dict(nd=nS, xn=xn, u=f['u_dof'][f['mesh_l2g']], qp=np.ascontiguousarray(f['x_ref'][:, :nS]), qw=f['dV_ref'],
                   beta=f['q_m_betaBDF'], nl=f['q_numDiff_u_last'], h=f['elementDiameter'],
                   v=f['velocity'][..., :nS], por=f['q_porosity'])


def timing(d, kernel):
    reps = 100
    big = {k: (np.concatenate([v] * reps) if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == d['xn'].shape[0] else v)
           for k, v in d.items()}
    best = {}
    for name, mod in (('hand-written', hand), ('generated', gen)):
        t = []
        for _ in range(7):
            t0 = time.perf_counter()
            run(mod, kernel, big)
            t.append(time.perf_counter() - t0)
        best[name] = min(t)
    n = big['xn'].shape[0]
    say('  %-6s %d elements: hand-written %.4f s, generated %.4f s (generated / hand-written = %.3f)'
        % (kernel, n, best['hand-written'], best['generated'], best['generated'] / best['hand-written']))


def partitioned(d, kernel):
    """Assemble R and J (dense) from the elements of this rank with the generated functor, sum over ranks, compare with the
    serial hand-written assembly. The element -> global dof map is the mesh's: rebuild it from the coordinates."""
    nE = d['xn'].shape[0]
    mine = np.arange(nE)[RANK::SIZE] if os.environ.get('PART', 'contiguous') == 'cyclic' else np.array_split(np.arange(nE), SIZE)[RANK]
    keys = {tuple(p): i for i, p in enumerate(map(tuple, np.unique(d['xn'].reshape(-1, 3), axis=0)))}
    l2g = np.array([[keys[tuple(p)] for p in e] for e in d['xn']])
    nN = len(keys)

    def assemble(mod, sel):
        sub = {k: (v[sel] if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == nE else v) for k, v in d.items()}
        R, J, _, _ = run(mod, kernel, sub)
        r = np.zeros(nN)
        Jg = np.zeros((nN, nN))
        for e, g in enumerate(l2g[sel]):
            np.add.at(r, g, R[e])
            Jg[np.ix_(g, g)] += J[e]
        return r, Jg
    r_s, J_s = assemble(hand, np.arange(nE))
    r_p, J_p = assemble(gen, mine)
    if comm:
        r_p, J_p = comm.allreduce(r_p), comm.allreduce(J_p)
    err = max(np.abs(r_p - r_s).max() / np.abs(r_s).max(), np.abs(J_p - J_s).max() / np.abs(J_s).max())
    say('  %-62s %.2e  %s' % ('%s generated on %d rank(s) vs serial hand-written' % (kernel, SIZE), err, 'ok' if err < 1e-13 else 'FAIL'))
    global OK
    OK &= err < 1e-13


def main(argv):
    say('1. regeneration')
    if RANK == 0:
        regeneration()
    rng = np.random.default_rng(20261116)
    fixtures = argv or [os.path.join(HERE, 'vof_damBreak_he0.05.npz')]
    say('2. generated vs hand-written functor')
    for p in fixtures:
        f, d = fixture_case(p)
        compare('VOF, damBreak fixture (porosity 1)', run(gen, 'vof', d), run(hand, 'vof', d), 1e-13)
        compare('NCLS, damBreak fixture', run(gen, 'ncls', d), run(hand, 'ncls', d), 1e-13)
    for nd in (2, 3):
        r = random_case(rng, nd)
        compare('VOF, random porosity/velocity, %dD, %d elements' % (nd, r['xn'].shape[0]), run(gen, 'vof', r), run(hand, 'vof', r))
        compare('NCLS, random velocity, %dD' % nd, run(gen, 'ncls', r), run(hand, 'ncls', r))
    say('3. partition independence (%d rank(s))' % SIZE)
    for nd in (2, 3):
        r = random_case(rng, nd, n=4)
        partitioned(r, 'vof')
        partitioned(r, 'ncls')
    if RANK == 0:
        say('4. refusals')
        expect = {'rdls': ['nonlinear in grad u', 'name h'], 'mcorr': ['diffusive flux', 'depends on u', 'name h'],
                  'flow': ['one equation, not 3']}
        for n, words in expect.items():
            p = subprocess.run([sys.executable, os.path.join(HERE, 'generator', 'generate.py'), os.devnull,
                                os.path.join(HERE, 'generator', 'inputs', 'refused_%s_2d.json' % n)], capture_output=True, text=True)
            check('%s is refused, naming: %s' % (n, ', '.join(words)), p.returncode == 1 and all(w in p.stderr for w in words))
    if RANK == 0 and SIZE == 1:
        say('5. run time')
        f, d = fixture_case(fixtures[0])
        timing(d, 'vof')
        timing(d, 'ncls')
    if comm:
        OK_all = comm.allreduce(int(OK), op=MPI.MIN)
    else:
        OK_all = int(OK)
    say('PASS' if OK_all else 'FAIL')
    return 0 if OK_all else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
