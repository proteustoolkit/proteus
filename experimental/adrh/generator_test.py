"""The generated functors (generated.h, module adrh_gen) against the hand-written ones (vof.h, ncls.h, module adrh_proto).

  python generator_test.py [fixtures...]            serial; `mpiexec -n 2 python generator_test.py` adds the partitioned check
Needs both modules built (`sh build.sh gen`). It does not need ymf: the generator's inputs are the JSON in generator/inputs, written
by generator/make_inputs.py from ymf's dam break specs (regenerate.sh).

The generator reads the spec and chooses nothing by the model's name: stabilization, shock capturing and its lagging, the boundary
kind (face kernel), the per-point inputs (the fields the sub-problem reads, the mesh symbol h) and porosity (a coefficient with the
role `porosity`, here a function of x inlined by the spec, so the hand-written kernel is given theta(x_q) as its porosity field).

1. generated.h is what generator/generate.py emits from generator/inputs/*.json (regeneration check).
2. What the engine reads from the spec: stabilization constants, tau form, face kernel (functor_info).
3. On the damBreak fixture (proteus's own arrays) and on random data (2D, 3D tets, around the porous block): residual, Jacobian,
   shock-capturing viscosity and CFL of the generated against the hand-written functor, the largest difference relative to the
   largest value and the share of entries that are bitwise equal; the faces, same comparison.
4. Partition independence: each rank assembles its share of the elements with the generated functor; the sum over ranks agrees
   with the serial hand-written assembly.
5. What the generator refuses: stabilizations and boundary kinds the template cannot take, and the spec's RDLS (a Hamiltonian
   nonlinear in grad u), MCorr (no stabilization stated: Galerkin) and flow (three equations).
6. One run-time comparison: the element kernels on the fixture tiled to ~73 000 elements, minimum of 7 repeats.
"""
import copy
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS = os.path.join(HERE, 'generator', 'inputs')
sys.path.insert(0, HERE)
hand, gen = importlib.import_module('adrh_proto'), importlib.import_module('adrh_gen')
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:                                   # serial only
    comm = None
RANK, SIZE = (comm.Get_rank(), comm.Get_size()) if comm else (0, 1)
OK = True
ALPHA, SC_C, SC_UREF, SC_BETA, DROP = 2.5, 0.5, 1.0, 1.5, 0.0     # what dambreak_formulation.ymf states of VOF and NCLS (alpha: the integrator's)
GENERATED = ['vof_2d', 'ncls_2d', 'porous_vof_2d', 'porous_vof_3d', 'porous_ncls_2d', 'porous_ncls_3d', 'transport_2d',
             'transport_lagged_2d', 'transport_step_2d']


def say(*a):
    if RANK == 0:
        print(*a)


def check(name, cond):
    global OK
    OK &= bool(cond)
    say('  %-70s %s' % (name, 'ok' if cond else 'FAIL'))


def compare(name, a, b, tol=1e-13):
    """a, b: tuples of arrays; largest |a-b| / largest |b| per array and the bitwise-equal share, over all arrays."""
    worst, same, total = 0.0, 0, 0
    for x, y in zip(a, b):
        x, y = np.asarray(x, float), np.asarray(y, float)
        worst = max(worst, np.abs(x - y).max() / max(np.abs(y).max(), 1e-300))
        same += int((x == y).sum())
        total += x.size
    say('  %-70s %.2e  bitwise equal %5.1f%%  %s' % (name, worst, 100.0 * same / total, 'ok' if worst <= tol else 'FAIL'))
    global OK
    OK &= worst <= tol
    return worst


def regeneration():
    out = os.path.join(tempfile.gettempdir(), 'generated_check_%d.h' % os.getpid())
    subprocess.check_call([sys.executable, os.path.join(HERE, 'generator', 'generate.py'), out] +
                          [os.path.join(INPUTS, n + '.json') for n in GENERATED])
    same = open(out).read() == open(os.path.join(HERE, 'generated.h')).read()
    os.remove(out)
    check('generated.h is the generator output for generator/inputs/*.json', same)


# ---------------------------------------------------------------------------------------------------------------------
# the porosity of dambreak_porous.ymf, written here independently of the generator: theta = 1 - 0.4 H(x - 2.2) H(0.3 - y) with
# the smoothed Heaviside of half-width 1.5 h (the ymf spec's regularization of `heaviside`)

def smoothed_heaviside(a, eps):
    return np.where(a > eps, 1.0, np.where(a < -eps, 0.0, 0.5 * (1.0 + a / eps + np.sin(np.pi * a / eps) / np.pi)))


def theta(x, y, h):
    return 1.0 - 0.4 * smoothed_heaviside(x - 2.2, 1.5 * h) * smoothed_heaviside(0.3 - y, 1.5 * h)


def points(d):
    """Physical coordinates of the quadrature points (nE, nQ, 3) of a P1 mesh."""
    qp = d['qp']
    w = np.concatenate([1.0 - qp.sum(axis=1, keepdims=True), qp], axis=1)        # barycentric (nQ, nD)
    return np.einsum('qb,ebi->eqi', w, d['xn'])


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
    """Random states on a mesh that straddles the porous block's corner (x = 2.2, y = 0.3): window [2.0, 2.6] x [0, 0.6]."""
    if nd == 3:
        pts, l2g, qp, qw = tets(n)
    else:                                              # triangles of a unit square
        pts = np.array([[i, j, 0.0] for j in range(n * 4 + 1) for i in range(n * 4 + 1)]) / (n * 4)
        idx = lambda i, j: i + (n * 4 + 1) * j
        l2g = np.array([t for j in range(n * 4) for i in range(n * 4)
                        for t in ([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)], [idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)])])
        qp = np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]])
        qw = np.full(3, 1.0 / 6.0)
    pts = pts * 0.6 + np.array([2.0, 0.0, 0.0])
    nE, nQ = len(l2g), len(qw)
    xn = pts[l2g]
    u = rng.normal(size=(nE, nd + 1))
    return dict(nd=nd, xn=xn, u=u, qp=qp, qw=qw, beta=rng.normal(size=(nE, nQ)), nl=np.abs(rng.normal(size=(nE, nQ))) * 1e-2,
                h=rng.uniform(0.02, 0.1, nE), v=rng.normal(size=(nE, nQ, nd)))


def fixture_case(path):
    f = dict(np.load(path))
    nS = f['mesh_l2g'].shape[1] - 1
    xn = f['mesh_dof'][f['mesh_l2g']]
    return f, dict(nd=nS, xn=xn, u=f['u_dof'][f['mesh_l2g']], qp=np.ascontiguousarray(f['x_ref'][:, :nS]), qw=f['dV_ref'],
                   beta=f['q_m_betaBDF'], nl=f['q_numDiff_u_last'], h=f['elementDiameter'],
                   v=f['velocity'][..., :nS])


def hand_fields(d, kernel, por):
    """The hand-written functor's per-point inputs: (v, porosity) for VOF, (v) for NCLS."""
    return np.concatenate([d['v'], por[..., None]], axis=-1) if kernel == 'vof' else d['v']


def gen_fields(d, layout):
    """The generated functor's: the velocity, then the mesh symbol h (the element diameter) where the spec's coefficients use it."""
    hq = np.broadcast_to(d['h'][:, None, None], d['v'].shape[:2] + (1,))
    return np.ascontiguousarray(np.concatenate([d['v'], hq], axis=-1) if layout == 'h' else d['v'])


def run_hand(kernel, d, por):
    fn = getattr(hand, kernel + '_supg')
    return fn(d['nd'], d['xn'], d['u'], d['qp'], d['qw'], hand_fields(d, kernel, por), d['beta'], d['nl'], d['h'],
              ALPHA, SC_C, SC_UREF, SC_BETA, DROP)[:4]


def run_gen(name, d, layout):
    """The generated functor with the stabilization the spec states (no sc_* arguments)."""
    return gen.generated_supg(name, d['nd'], d['xn'], d['u'], d['qp'], d['qw'], gen_fields(d, layout), d['beta'], d['nl'], d['h'],
                              ALPHA)[:4]


def compare_case(label, d, tol=1e-13):
    """Plain VOF and NCLS (no porosity) and the porous VOF (theta inlined by the spec) on the same data."""
    one = np.ones(d['qw'].shape[:1] + (0,))
    x = points(d)
    por1 = np.ones(x.shape[:2])
    porθ = theta(x[..., 0], x[..., 1], d['h'][:, None])
    nd = d['nd']
    out = {}
    if nd == 2:
        out['vof'] = compare('VOF (no porosity) %s' % label, run_gen('vof', d, 'v'), run_hand('vof', d, por1), tol)
        out['ncls'] = compare('NCLS %s' % label, run_gen('ncls', d, 'v'), run_hand('ncls', d, por1), tol)
    out['pvof'] = compare('VOF with the spec\'s porosity %s' % label, run_gen('porous_vof', d, 'h'), run_hand('vof', d, porθ), tol)
    out['pncls'] = compare('NCLS (porous spec) %s' % label, run_gen('porous_ncls', d, 'v'), run_hand('ncls', d, por1), tol)
    return out


# ---------------------------------------------------------------------------------------------------------------------

def face_data(f):
    """The exterior faces of a fixture, as vof_test.py / ncls_test.py hand them to the face kernels."""
    nS = f['mesh_l2g'].shape[1] - 1
    xn = f['mesh_dof'][f['mesh_l2g']]
    ue = f['u_dof'][f['mesh_l2g']]
    ebN = f['exteriorElementBoundariesArray']
    eN_f = f['elementBoundaryElementsArray'][ebN, 0]
    k_f = f['elementBoundaryLocalElementBoundariesArray'][ebN, 0]
    xi = np.ascontiguousarray(f['u_trial_trace_ref'][:, :, 1:])
    v = f['ebqe_velocity_ext'][..., :nS]
    return dict(nd=nS, xn=xn[eN_f], u=ue[eN_f], k=k_f, xi=xi, dS=f['dS_ref'], v=v, isD=f['isDOFBoundary_u'], g=f['ebqe_bc_u_ext'],
                isF=f['isFluxBoundary_u'], gf=f['ebqe_bc_flux_u_ext'], rd=f.get('ebqe_rd_u_ext', np.zeros(f['ebqe_bc_u_ext'].shape)), por=f['ebqe_porosity_ext'])


def faces(f):
    fd = face_data(f)
    n, nd = fd['v'].shape[:2], fd['nd']
    zero = np.zeros(fd['g'].shape)
    for name, kernel in (('vof', 'vof'), ('ncls', 'ncls')):
        fh = np.concatenate([fd['v'], fd['por'][..., None]], axis=-1) if kernel == 'vof' else fd['v']
        if kernel == 'vof':
            a = hand.vof_faces(nd, fd['xn'], fd['u'], fd['k'], fd['xi'], fd['dS'], fh, fd['isD'], fd['g'], fd['isF'], fd['gf'])
        else:
            a = hand.ncls_faces(nd, fd['xn'], fd['u'], fd['k'], fd['xi'], fd['dS'], fh, fd['isD'], fd['g'], fd['rd'])
        b = gen.generated_faces(name, nd, fd['xn'], fd['u'], fd['k'], fd['xi'], fd['dS'], np.ascontiguousarray(fd['v']), fd['isD'],
                                fd['g'], fd['isF'], fd['gf'], fd['rd'])
        compare('%s exterior faces, face kernel from the spec' % kernel.upper(), b, a, 1e-13)


def engine_reads(info):
    """What the engine read from the spec (functor_info) against what the spec states (dambreak_formulation.ymf)."""
    for name in ('vof', 'ncls'):
        i = gen.functor_info(name, 2)
        check('%s: shock capturing %.1f beta %.1f reference %.1f lagged, tau form %d, face %d (0 upwind, 1 inflow penalty)' %
              (name.upper(), i['scFactor'], i['scBeta'], i['scReference'], i['tauForm'], i['face']),
              (i['scFactor'], i['scBeta'], i['scReference'], i['scLagged'], i['tauForm']) == (0.5, 1.5, 1.0, True, 0) and
              i['face'] == {'vof': 0, 'ncls': 1}[name])
    i = gen.functor_info('porous_vof', 3)
    check('porous VOF 3D: reads v and h (nFields %d), evaluates at the point (usesPoint %s)' % (i['nFields'], i['usesPoint']),
          i['nFields'] == 4 and i['usesPoint'])


def lagged():
    """A coefficient the spec lags. kappa = 1 + u^2 multiplies the flux. Lagged to the previous iterate the functor evaluates kappa from
    the current u as a double, so its residual is the full one and its Jacobian is Picard's (kappa's dependence on u is dropped);
    lagged to the previous step the functor takes kappa as a per-point input and has no derivative of it. Giving the second the
    first's kappa at every point must therefore give the same residual and Jacobian, and both must differ from the unlagged
    Jacobian (which carries d kappa / d u). (Their residuals differ from the unlagged one too: the stabilization's characteristic
    velocity is df, which is kappa v in the lagged form and (3u^2 + 1) v in the free one.)"""
    rng = np.random.default_rng(7)
    for nd in (2,):
        d = random_case(rng, nd)
        qp = d['qp']
        w = np.concatenate([1.0 - qp.sum(axis=1, keepdims=True), qp], axis=1)
        uq = np.einsum('qb,eb->eq', w, d['u'])
        fields_step = np.ascontiguousarray(np.concatenate([d['v'], (1.0 + uq * uq)[..., None]], axis=-1))
        args = (d['nd'], d['xn'], d['u'], d['qp'], d['qw'])
        tail = (d['beta'], d['nl'], d['h'], ALPHA)
        free = gen.generated_supg('transport', *args, d['v'], *tail)
        prev_it = gen.generated_supg('transport_lagged', *args, d['v'], *tail)
        prev_step = gen.generated_supg('transport_step', *args, fields_step, *tail)
        compare('lagged to the previous iterate vs lagged to the previous step: R, J, numDiff, cfl', prev_it[:4], prev_step[:4], 1e-14)
        dJ = np.abs(free[1] - prev_it[1]).max() / np.abs(free[1]).max()
        say('  %-70s %.2e' % ('Jacobian of the free problem vs the lagged one: d kappa / d u (must differ)', dJ))
        check('the lagged Jacobian drops the dependence of kappa on u', dJ > 1e-2)


def timing(d, name, layout, kernel):
    reps = 100
    big = {k: (np.concatenate([v] * reps) if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == d['xn'].shape[0] else v)
           for k, v in d.items()}
    por = np.ones(points(big).shape[:2])
    best = {}
    for label, fn in (('hand-written', lambda: run_hand(kernel, big, por)), ('generated', lambda: run_gen(name, big, layout))):
        t = []
        for _ in range(7):
            t0 = time.perf_counter()
            fn()
            t.append(time.perf_counter() - t0)
        best[label] = min(t)
    say('  %-6s %d elements: hand-written %.4f s, generated %.4f s (generated / hand-written = %.3f)'
        % (kernel, big['xn'].shape[0], best['hand-written'], best['generated'], best['generated'] / best['hand-written']))


def partitioned(d, name, layout, kernel):
    """Assemble R and J (dense) from the elements of this rank with the generated functor, sum over ranks, compare with the
    serial hand-written assembly. The element -> global dof map is the mesh's: rebuild it from the coordinates."""
    nE = d['xn'].shape[0]
    mine = np.arange(nE)[RANK::SIZE] if os.environ.get('PART', 'contiguous') == 'cyclic' else np.array_split(np.arange(nE), SIZE)[RANK]
    keys = {tuple(p): i for i, p in enumerate(map(tuple, np.unique(d['xn'].reshape(-1, 3), axis=0)))}
    l2g = np.array([[keys[tuple(p)] for p in e] for e in d['xn']])
    nN = len(keys)
    x = points(d)
    por = theta(x[..., 0], x[..., 1], d['h'][:, None]) if name == 'porous_vof' else np.ones(x.shape[:2])

    def assemble(which, sel):
        sub = {k: (v[sel] if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == nE else v) for k, v in d.items()}
        R, J, _, _ = run_gen(name, sub, layout) if which == 'gen' else run_hand(kernel, sub, por[sel])
        r = np.zeros(nN)
        Jg = np.zeros((nN, nN))
        for e, g in enumerate(l2g[sel]):
            np.add.at(r, g, R[e])
            Jg[np.ix_(g, g)] += J[e]
        return r, Jg
    r_s, J_s = assemble('hand', np.arange(nE))
    r_p, J_p = assemble('gen', mine)
    if comm:
        r_p, J_p = comm.allreduce(r_p), comm.allreduce(J_p)
    err = max(np.abs(r_p - r_s).max() / np.abs(r_s).max(), np.abs(J_p - J_s).max() / np.abs(J_s).max())
    say('  %-70s %.2e  %s' % ('%s %dD generated on %d rank(s) vs serial hand-written' % (name, d['nd'], SIZE), err, 'ok' if err < 1e-13 else 'FAIL'))
    global OK
    OK &= err < 1e-13


def refusals():
    """The generator refuses what the template cannot take, naming every blocker. Each case edits a copy of the VOF input."""
    base = json.load(open(os.path.join(INPUTS, 'vof_2d.json'), encoding='utf-8'))

    def refused(label, edit, words):
        spec = copy.deepcopy(base)
        edit(spec)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'in.json')
            json.dump(spec, open(path, 'w', encoding='utf-8'), ensure_ascii=False)
            p = subprocess.run([sys.executable, os.path.join(HERE, 'generator', 'generate.py'), os.devnull, path],
                               capture_output=True, text=True)
        check('%s is refused, naming: %s' % (label, ', '.join(words)), p.returncode == 1 and all(w in p.stderr for w in words))

    refused('ASGS', lambda s: s['stabilization'].update(method='ASGS'), ['stabilization ASGS'])
    refused('a lagged subgrid error', lambda s: s['stabilization'].update(lagged=True), ['lagged: true'])
    refused('tau: element_size', lambda s: s['stabilization'].update(tau='element_size'), ['tau: element_size'])
    refused('a weak Dirichlet condition', lambda s: s['boundary_conditions'].append(
        {'variable': 'χ', 'kind': 'weak', 'type': 'dirichlet'}), ['weak Dirichlet'])
    refused('strong Dirichlet on the whole boundary', lambda s: s['boundary_conditions'][0].pop('only'), ['whole boundary'])
    refused('an unknown field', lambda s: s.update(reads={}), ['the name v_0 is not'])
    # the other sub-problems of the same spec, as they are
    for n, words in {'rdls': ['nonlinear in grad u'], 'mcorr': ['no stabilization is stated'], 'flow': ['one equation, not 3']}.items():
        p = subprocess.run([sys.executable, os.path.join(HERE, 'generator', 'generate.py'), os.devnull,
                            os.path.join(INPUTS, '%s_2d.json' % n)], capture_output=True, text=True)
        check('%s (the spec\'s own sub-problem) is refused, naming: %s' % (n, ', '.join(words)), p.returncode == 1 and all(w in p.stderr for w in words))


def main(argv):
    say('1. regeneration')
    if RANK == 0:
        regeneration()
    say('2. what the engine reads from the spec')
    engine_reads(None)
    rng = np.random.default_rng(20261117)
    fixtures = argv or [os.path.join(HERE, 'vof_damBreak_he0.05.npz')]
    say('3. generated vs hand-written functor (the porosity the spec states is theta(x_q) in the hand-written field)')
    for p in fixtures:
        f, d = fixture_case(p)
        compare_case('(damBreak fixture, %d elements)' % d['xn'].shape[0], d)
        faces(f)
    for nd in (2, 3):
        r = random_case(rng, nd)
        compare_case('(random, %dD, %d elements)' % (nd, r['xn'].shape[0]), r)
    say('3b. lagged coefficients (ymf round 5: previous_iterate, previous_step)')
    lagged()
    say('4. partition independence (%d rank(s))' % SIZE)
    for nd in (2, 3):
        r = random_case(rng, nd, n=4)
        partitioned(r, 'porous_vof', 'h', 'vof')
        partitioned(r, 'porous_ncls', 'v', 'ncls')
    if RANK == 0:
        say('5. refusals')
        refusals()
    if RANK == 0 and SIZE == 1:
        say('6. run time')
        f, d = fixture_case(fixtures[0])
        timing(d, 'vof', 'v', 'vof')
        timing(d, 'ncls', 'v', 'ncls')
    OK_all = comm.allreduce(int(OK), op=MPI.MIN) if comm else int(OK)
    say('PASS' if OK_all else 'FAIL')
    return 0 if OK_all else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
