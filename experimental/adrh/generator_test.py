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
5. What the generator refuses: stabilizations and boundary kinds the template cannot take, and the spec's flow (three equations).
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
             'transport_lagged_2d', 'transport_step_2d', 'reacdiff_2d', 'rdls_nosc_2d']
GENERATED = ['vof_2d', 'ncls_2d', 'rdls_2d', 'mcorr_2d'] + GENERATED[2:6] + ['porous_mcorr_2d', 'porous_mcorr_3d', 'porous_rdls_3d'] + GENERATED[6:]


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


def per_point_h(d, rng):
    """h at each point (nE, nQ): the mesh symbol h of the spec is a per-point input (the driver says which size: proteus's node
    diameters interpolated to the point, here an element diameter jittered point by point), not the element's constant."""
    d['hq'] = d['h'][:, None] * rng.uniform(0.7, 1.3, d['v'].shape[:2])
    return d


def gen_fields(d, layout):
    """The generated functor's: the velocity, then the mesh symbol h at the point where the spec's coefficients use it."""
    hq = d['hq'][..., None]
    return np.ascontiguousarray(np.concatenate([d['v'], hq], axis=-1) if layout == 'h' else d['v'])


def run_hand(kernel, d, por):
    fn = getattr(hand, kernel + '_supg')
    return fn(d['nd'], d['xn'], d['u'], d['qp'], d['qw'], hand_fields(d, kernel, por), d['beta'], d['nl'], d['h'],
              ALPHA, SC_C, SC_UREF, SC_BETA, DROP)[:4]


def run_gen(name, d, layout):
    """The generated functor with the stabilization the spec states (no sc_* arguments)."""
    return gen.generated_supg(name, d['nd'], d['xn'], d['u'], d['qp'], d['qw'], gen_fields(d, layout), d['beta'], d['nl'], d['h'],
                              ALPHA, False)[:4]


def compare_case(label, d, tol=1e-13):
    """Plain VOF and NCLS (no porosity) and the porous VOF (theta inlined by the spec) on the same data."""
    one = np.ones(d['qw'].shape[:1] + (0,))
    x = points(d)
    por1 = np.ones(x.shape[:2])
    porθ = theta(x[..., 0], x[..., 1], d['hq'])
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
        tail = (d['beta'], d['nl'], d['h'], ALPHA, False)
        free = gen.generated_supg('transport', *args, d['v'], *tail)
        prev_it = gen.generated_supg('transport_lagged', *args, d['v'], *tail)
        prev_step = gen.generated_supg('transport_step', *args, fields_step, *tail)
        compare('lagged to the previous iterate vs lagged to the previous step: R, J, numDiff, cfl', prev_it[:4], prev_step[:4], 1e-14)
        dJ = np.abs(free[1] - prev_it[1]).max() / np.abs(free[1]).max()
        say('  %-70s %.2e' % ('Jacobian of the free problem vs the lagged one: d kappa / d u (must differ)', dJ))
        check('the lagged Jacobian drops the dependence of kappa on u', dJ > 1e-2)


def finite_difference(name, d, fields, alpha, exact, eps=1e-6):
    """The element Jacobian of the generated functor against central differences of its residual in each nodal value (the viscosity
    is off in these specs, so the residual is the whole weak form): largest difference over largest entry."""
    args = (d['nd'], d['xn'])
    tail = (d['qp'], d['qw'], fields, d['beta'], d['nl'], d['h'], alpha, exact)
    R, J = gen.generated_supg(name, *args[:1], d['xn'], d['u'], *tail)[:2]
    fd = np.zeros_like(J)
    for j in range(J.shape[2]):
        up, um = d['u'].copy(), d['u'].copy()
        up[:, j] += eps
        um[:, j] -= eps
        Rp = gen.generated_supg(name, d['nd'], d['xn'], up, *tail)[0]
        Rm = gen.generated_supg(name, d['nd'], d['xn'], um, *tail)[0]
        fd[:, :, j] = (Rp - Rm) / (2 * eps)
    return np.abs(J - fd).max() / np.abs(fd).max()


def slots():
    """Slice 2: the slots the SUPG template gained, read from specs. (a) a reaction r(u) and an isotropic diffusion a(u) (reacdiff:
    dr/du and da/du from ymf, linear advection): the Jacobian is exact. (b) a Hamiltonian nonlinear in grad u (the spec's RDLS, shock
    capturing off, no time term): proteus's Jacobian freezes a = dH in the subgrid term; with the slot d(dH)/d(grad u) (dual numbers)
    and exactJacobian it is exact."""
    rng = np.random.default_rng(11)
    d = per_point_h(random_case(rng, 2, n=3), rng)
    err = finite_difference('reacdiff', d, np.ascontiguousarray(d['v']), ALPHA, False)
    say('  %-70s %.2e' % ('reaction + diffusion depending on u: Jacobian vs finite differences', err))
    check('dr and diffusion slots give an exact Jacobian (< 1e-7)', err < 1e-7)
    nE, nQ = d['qw'].shape[0], d['qw'].shape[0]
    nE = d['xn'].shape[0]
    phi = rng.normal(size=(nE, nQ)) * d['hq']                                    # the level set near the interface: |phi| ~ h
    fields = np.ascontiguousarray(np.stack([phi, d['hq']], axis=-1))
    d['beta'] = np.zeros((nE, nQ))
    d['nl'] = np.zeros((nE, nQ))
    off = finite_difference('rdls_nosc', d, fields, 0.0, False)
    on = finite_difference('rdls_nosc', d, fields, 0.0, True)
    say('  %-70s %.2e' % ('RDLS (nonlinear H): proteus Jacobian (a frozen) vs finite differences', off))
    say('  %-70s %.2e' % ('RDLS with d(dH)/d(grad u) (exactJacobian) vs finite differences', on))
    check('the d(dH)/d(grad u) slot makes the Jacobian exact (< 1e-6), and the frozen one is not (> 1e-3)', on < 1e-6 and off > 1e-3)


# ---------------------------------------------------------------------------------------------------------------------
# slice 3: RDLS and MCorr

RDLS_SC, RDLS_EPS, RDLS_WDF = 0.9, 0.75, 10.0      # what dambreak_formulation.ymf states of RDLS (factor, eps = 0.75 h); the pin's factor is the engine's


def rdls_case(d, rng):
    """A random RDLS state on a case dict (xn, u, qp, qw, hq, h): the level set phi at the points (|phi| ~ h), the frozen phi."""
    nE, nQ = d['hq'].shape
    d['phi'] = rng.normal(size=(nE, nQ)) * d['hq']
    return d


def rdls_hand(d, bg):
    """The hand-written RDLS with the interface pin switched off (the frozen level set is far from zero) and background diffusion bg."""
    nE, nQ = d['hq'].shape
    f3 = np.ascontiguousarray(np.stack([d['phi'], d['hq'], np.broadcast_to(d['h'][:, None], (nE, nQ))], axis=-1))
    far = np.full(d['u'].shape, 1e9)
    return hand.rdls_supg(d['nd'], d['xn'], d['u'], far, d['qp'], d['qw'], f3, d['h'], RDLS_SC, RDLS_EPS, bg, RDLS_WDF)[:4]


def rdls_gen(d):
    nE, nQ = d['hq'].shape
    f2 = np.ascontiguousarray(np.stack([d['phi'], d['hq']], axis=-1))
    z = np.zeros((nE, nQ))
    return gen.generated_supg('rdls' if d['nd'] == 2 else 'porous_rdls', d['nd'], d['xn'], d['u'], d['qp'], d['qw'], f2, z, z, d['h'], 0.0, False)[:4]


def mcorr_hand(d, por):
    nE, nQ = d['hq'].shape
    he = np.broadcast_to(d['h'][:, None], (nE, nQ))
    f6 = np.ascontiguousarray(np.stack([d['phi'], d['Hv'], por, d['hq'], he, np.ones((nE, nQ))], axis=-1))
    return hand.mcorr(d['nd'], d['xn'], d['u'], d['qp'], d['qw'], f6, 1.0, 1.5, 1.5, 10.0)[:2]


def mcorr_gen(name, d):
    f3 = np.ascontiguousarray(np.stack([d['phi'], d['Hv'], d['hq']], axis=-1))
    return gen.generated_galerkin(name, d['nd'], d['xn'], d['u'], d['qp'], d['qw'], f3)


def slice3(f_rdls, f_mcorr, rng):
    say('  RDLS (spec: S = sign smoothed at 0.75 h, H = S |grad u|, r = -S, SUPG metric tau without a time term, sc 0.9 not lagged)')
    if f_rdls is not None:
        l2g, psi = f_rdls['mesh_l2g'], f_rdls['u_trial_ref']
        nE, nD = l2g.shape
        h_phi = np.zeros((nE, psi.shape[0]))
        for j in range(nD):
            h_phi += f_rdls['nodeDiametersArray'][l2g][:, j][:, None] * psi[None, :, j]
        d = dict(nd=nD - 1, xn=f_rdls['mesh_dof'][l2g], u=f_rdls['phi_dof'][l2g] * 0 + f_rdls['u_dof'][l2g] if 'u_dof' in f_rdls else f_rdls['phi_dof'][l2g],
                 qp=np.ascontiguousarray(psi[:, 1:]), qw=f_rdls['dV_ref'], h=f_rdls['elementDiameter'], hq=h_phi, phi=f_rdls['phi_ls'])
        compare("RDLS, proteus's fixture (%d elements), no background diffusion" % nE, rdls_gen(d), rdls_hand(d, 0.0), 1e-9)
        gap = max(np.abs(a - b).max() / np.abs(b).max() for a, b in zip(rdls_gen(d)[:2], rdls_hand(d, float(f_rdls['backgroundDiffusionFactor']))[:2]))
        say('  %-70s %.2e' % ('  (with proteus\'s background diffusion %.2f the spec does not state: R, J differ by)' % float(f_rdls['backgroundDiffusionFactor']), gap))
    for nd in (2, 3):
        d = rdls_case(per_point_h(random_case(rng, nd), rng), rng)
        compare('RDLS, random state, %dD, %d elements' % (nd, d['xn'].shape[0]), rdls_gen(d), rdls_hand(d, 0.0), 1e-9)
    say('  MCorr (spec: reaction theta (H_eps(phi + u) - chi) with eps = 1.5 h, diffusion 10 h, no stabilization: Galerkin)')
    if f_mcorr is not None:
        l2g, psi = f_mcorr['mesh_l2g'], f_mcorr['u_trial_ref']
        nE, nD = l2g.shape
        h_phi = np.zeros((nE, psi.shape[0]))
        for j in range(nD):
            h_phi += f_mcorr['nodeDiametersArray'][l2g][:, j][:, None] * psi[None, :, j]
        assert f_mcorr['q_porosity'].min() == 1.0 == f_mcorr['q_porosity'].max() and f_mcorr['phi_solid'].min() > 0
        for label, u in (('the iterate proteus ran', f_mcorr['u_dof']), ('proteus\'s next iterate', f_mcorr['u_next'])):
            d = dict(nd=nD - 1, xn=f_mcorr['mesh_dof'][l2g], u=u[l2g], qp=np.ascontiguousarray(f_mcorr['x_ref'][:, :nD - 1]),
                     qw=f_mcorr['dV_ref'], h=f_mcorr['elementDiameter'], hq=h_phi, phi=f_mcorr['q_phi'], Hv=f_mcorr['q_H'])
            compare('MCorr, proteus\'s fixture (%d elements) at %s' % (nE, label), mcorr_gen('mcorr', d), mcorr_hand(d, np.ones(h_phi.shape)), 1e-12)
        # against proteus's own assembled residual and Jacobian
        import scipy.sparse as sp
        d['u'] = f_mcorr['u_dof'][l2g]
        R, J = mcorr_gen('mcorr', d)
        nN = f_mcorr['mesh_dof'].shape[0]
        r = np.zeros(nN)
        np.add.at(r, l2g, R)
        Jg = sp.csr_matrix((J.ravel(), (np.repeat(l2g, nD, axis=1).ravel(), np.tile(l2g, nD).ravel())), shape=(nN, nN))
        Jp = sp.csr_matrix((f_mcorr['out_globalJacobian'], f_mcorr['csrColumnOffsets_DofLoops'], f_mcorr['csrRowIndeces_DofLoops']), shape=(nN, nN))
        er = np.abs(r - f_mcorr['out_globalResidual']).max() / np.abs(f_mcorr['out_globalResidual']).max()
        ej = abs(Jg - Jp).max() / abs(Jp).max()
        say('  %-70s %.2e  Jacobian %.2e' % ('MCorr generated vs proteus MCorr kernel: global residual', er, ej))
        check('generated MCorr reproduces proteus (< 1e-12)', er < 1e-12 and ej < 1e-12)
    for nd in (2, 3):
        d = rdls_case(per_point_h(random_case(rng, nd), rng), rng)
        d['Hv'] = rng.uniform(0.0, 1.0, d['hq'].shape)
        x = points(d)
        compare('MCorr (no porosity), random state, %dD, %d elements' % (nd, d['xn'].shape[0]), mcorr_gen('mcorr', d),
                mcorr_hand(d, np.ones(d['hq'].shape)), 1e-12) if nd == 2 else None
        compare('MCorr with the spec\'s porosity theta(x, h), random, %dD' % nd, mcorr_gen('porous_mcorr', d),
                mcorr_hand(d, theta(x[..., 0], x[..., 1], d['hq'])), 1e-12)


def partitioned_pair(label, d, hand_fn, gen_fn):
    """Element-partitioned assembly with the generated functor, summed over ranks, against the serial hand-written one
    (R, J of every element from hand_fn(d) / gen_fn(d))."""
    nE = d['xn'].shape[0]
    mine = np.arange(nE)[RANK::SIZE] if os.environ.get('PART', 'contiguous') == 'cyclic' else np.array_split(np.arange(nE), SIZE)[RANK]
    keys = {tuple(p): i for i, p in enumerate(map(tuple, np.unique(d['xn'].reshape(-1, 3), axis=0)))}
    l2g = np.array([[keys[tuple(p)] for p in e] for e in d['xn']])
    nN = len(keys)

    def assemble(fn, sel):
        sub = {k: (v[sel] if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == nE else v) for k, v in d.items()}
        R, J = fn(sub)[:2]
        r, Jg = np.zeros(nN), np.zeros((nN, nN))
        for e, g in enumerate(l2g[sel]):
            np.add.at(r, g, R[e])
            Jg[np.ix_(g, g)] += J[e]
        return r, Jg
    r_s, J_s = assemble(hand_fn, np.arange(nE))
    r_p, J_p = assemble(gen_fn, mine)
    if comm:
        r_p, J_p = comm.allreduce(r_p), comm.allreduce(J_p)
    err = max(np.abs(r_p - r_s).max() / np.abs(r_s).max(), np.abs(J_p - J_s).max() / np.abs(J_s).max())
    tol = 1e-9 if label.startswith('rdls') else 1e-12
    say('  %-70s %.2e  %s' % ('%s generated on %d rank(s) vs serial hand-written' % (label, SIZE), err, 'ok' if err < tol else 'FAIL'))
    global OK
    OK &= err < tol


def slice3_partitioned(rng):
    for nd in (2, 3):
        d = rdls_case(per_point_h(random_case(rng, nd, n=4), rng), rng)
        d['Hv'] = rng.uniform(0.0, 1.0, d['hq'].shape)
        x = points(d)
        partitioned_pair('rdls %dD' % nd, d, lambda s: rdls_hand(s, 0.0), rdls_gen)
        partitioned_pair('porous mcorr %dD' % nd, d, lambda s, x=x: mcorr_hand(s, theta(*[points(s)[..., 0], points(s)[..., 1]], s['hq'])),
                         lambda s: mcorr_gen('porous_mcorr', s))


def timing3(rng):
    """RDLS and MCorr element kernels, hand-written against generated, on a random 2D case tiled to ~100 000 elements."""
    d = rdls_case(per_point_h(random_case(rng, 2), rng), rng)
    d['Hv'] = rng.uniform(0.0, 1.0, d['hq'].shape)
    reps = 90
    big = {k: (np.concatenate([v] * reps) if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == d['xn'].shape[0] else v) for k, v in d.items()}
    one = np.ones(big['hq'].shape)
    for label, hf, gf in (('RDLS', lambda: rdls_hand(big, 0.0), lambda: rdls_gen(big)),
                          ('MCorr', lambda: mcorr_hand(big, one), lambda: mcorr_gen('mcorr', big))):
        best = {}
        for k, fn in (('hand-written', hf), ('generated', gf)):
            ts = []
            for _ in range(5):
                t0 = time.perf_counter()
                fn()
                ts.append(time.perf_counter() - t0)
            best[k] = min(ts)
        say('  %-6s %d elements: hand-written %.4f s, generated %.4f s (generated / hand-written = %.3f)'
            % (label, big['xn'].shape[0], best['hand-written'], best['generated'], best['generated'] / best['hand-written']))


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
    por = theta(x[..., 0], x[..., 1], d['hq']) if name == 'porous_vof' else np.ones(x.shape[:2])

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
    for n, words in {'flow': ['one equation, not 3']}.items():
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
        per_point_h(d, rng)
        compare_case('(damBreak fixture, %d elements)' % d['xn'].shape[0], d)
        faces(f)
    for nd in (2, 3):
        r = per_point_h(random_case(rng, nd), rng)
        compare_case('(random, %dD, %d elements)' % (nd, r['xn'].shape[0]), r)
    say('3b. lagged coefficients (ymf round 5: previous_iterate, previous_step)')
    lagged()
    say('3c. slots: h per point, dr, diffusion, d(dH)/d(grad u)')
    slots()
    say('3d. RDLS and MCorr from their sub-problems')
    slice3(dict(np.load(os.path.join(HERE, 'rdls_damBreak_he0.05.npz'))), dict(np.load(os.path.join(HERE, 'mcorr_damBreak_he0.05.npz'))), rng)
    say('4. partition independence (%d rank(s))' % SIZE)
    for nd in (2, 3):
        r = per_point_h(random_case(rng, nd, n=4), rng)
        partitioned(r, 'porous_vof', 'h', 'vof')
        partitioned(r, 'porous_ncls', 'v', 'ncls')
    slice3_partitioned(rng)
    if RANK == 0:
        say('5. refusals')
        refusals()
    if RANK == 0 and SIZE == 1:
        say('6. run time')
        f, d = fixture_case(fixtures[0])
        per_point_h(d, rng)
        timing(d, 'vof', 'v', 'vof')
        timing(d, 'ncls', 'v', 'ncls')
        timing3(rng)
    OK_all = comm.allreduce(int(OK), op=MPI.MIN) if comm else int(OK)
    say('PASS' if OK_all else 'FAIL')
    return 0 if OK_all else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
