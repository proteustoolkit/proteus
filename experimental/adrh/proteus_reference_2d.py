"""The 2D sand deck (test/richards/re_vgm_sand_10x10m_2d) run in proteus and in richards_nd.py on the same mesh, step for step.

Needs a proteus that has FCT_TYPE='implicit' (the JNMA branch, cekees/proteus#123 / #124), built, and its parun on the PATH of the environment named by
PROTEUS_REF_BIN (a directory with parun); PROTEUS_REF_DECKS is that checkout's test/richards. The deck is copied to a scratch directory and patched there
(nothing in the checkout is changed): Context options for the scheme, FCT_TYPE, beta, a lens of a finer soil, the number of fixed steps; strong Dirichlet
conditions (forceStrongConditions) and the 1D decks' Richards_IIPG_exterior flux and line search; a structured grid of nodes with the pond's edges on nodes
(pond limits with a 1e-9 tolerance, both here and in the deck), because on the deck's Triangle mesh (--mesh triangle) the pond edge falls inside a boundary face and proteus's weak
boundary flux acts on a face that is part pond and part not, which this driver does not model (the README has the size of the difference). Both sides use the deck's element quadrature (order 4).

What is compared, each over `steps` fixed steps of 0.35 d / steps, on the mesh proteus wrote to its archive (richards_nd.py reads the nodes and elements back):
Galerkin, the K-upwinded low-order scheme (proteus's STABILIZATION_TYPE=2), the semi-implicit (post-step) FCT and the implicit FCT, with and without the lens, and at several beta. Proteus's
semi-implicit FCT vetoes any step whose head correction exceeds 1 m (Richards.Coefficients FCT_DU_MAX); the patched deck lifts it.

--steady compares the 1D Celia deck run to a steady state (the interior starts on the line between the boundary heads, psi = -1 below and -3 above, T = 200 d) with the
exact steady compressible column of richards_nd_test.compressible_exact, for proteus's Galerkin and K-upwinded low-order schemes and for richards_nd.py's K-upwinded scheme.

usage: PROTEUS_REF_BIN=<env>/bin PROTEUS_REF_DECKS=<checkout>/test/richards python proteus_reference_2d.py [--steps 35] [--nn 31] [--only LABEL]
       PROTEUS_REF_BIN=... PROTEUS_REF_DECKS=... python proteus_reference_2d.py --steady
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import h5py
import numpy as np

import richards_nd as nd
from richards_regression import dimensionless_ks

L = 10.0
TOL = 1e-9                                                      # pond limits: x in [L/3 - TOL, 2L/3 + TOL]


def make_deck(src, dst):
    for name in ('re_vgm_sand_10x10m_2d_p.py', 're_vgm_sand_10x10m_2d_c0p1_n.py'):
        shutil.copy(os.path.join(src, name), dst)
    p = os.path.join(dst, 're_vgm_sand_10x10m_2d_p.py'); s = open(p).read()
    s = s.replace("nd = 2\n", """nd = 2
ct = Context.Options([
    ("STABILIZATION_TYPE", 2, "0 Galerkin, 2 edge-based low order"),
    ("FCT", False, "apply FCT"),
    ("FCT_TYPE", "semi-implicit", "semi-implicit or implicit"),
    ("BETA", 0.0001, "compressibility"),
    ("LENS", False, "a lens of a finer soil"),
    ("NDT", 100, "fixed steps"),
    ("NN", 31, "nodes per side"),
    ("REGULAR", True, "structured grid, else the deck's Triangle mesh"),
])
""", 1)
    s = s.replace("regularGrid=False", "regularGrid = 'global' if ct.REGULAR else False", 1)
    s = s.replace("LevelModelType = Richards.LevelModel", """if ct.LENS:
    alphaVGtypes[1] = 1.0; nVGtypes[1] = 1.6; thetaRtypes[1] = 0.08; thetaStypes[1] = 0.43; thetaSRtypes[1] = 0.35
    KsTypes[1, :] = [0.1 * dimensionless_conductivity, 0.1 * dimensionless_conductivity]

class LensCoefficients(Richards.Coefficients):
    def initializeMesh(self, mesh):
        if ct.LENS:
            cx = mesh.nodeArray[mesh.elementNodesArray, 0].mean(axis=1); cy = mesh.nodeArray[mesh.elementNodesArray, 1].mean(axis=1)
            mesh.elementMaterialTypes[:] = ((abs(cx - 5.0) < 2.0) & (abs(cy - 6.0) < 1.0)).astype('i')
        super().initializeMesh(mesh)

LevelModelType = Richards.LevelModel""", 1)
    s = s.replace("coefficients = Richards.Coefficients(nd,", "coefficients = LensCoefficients(nd,", 1)
    s = s.replace("beta=0.0001,", "beta=ct.BETA,", 1)
    s = s.replace("STABILIZATION_TYPE=2,#0 for galerkin, 2 for Low-order monotone and FCT", "STABILIZATION_TYPE=ct.STABILIZATION_TYPE,", 1)
    # Richards.FCTStep (semi-implicit FCT, Newton invert, i.e. every 2D deck) throws the limited step away and keeps the low-order head
    # whenever the correction moves any node's head by more than 1 m, which a dry node's moisture content does by metres. Lift the cap
    # (FCT_DU_MAX) or the FCT cases below are the low-order scheme.
    s = s.replace("FCT=False,#True,", "FCT=ct.FCT,\n                                     FCT_TYPE=ct.FCT_TYPE,\n                                     FCT_DU_MAX=1e30,\n                                     forceStrongConditions=True,", 1)
    s = s.replace("x[0] >= L[0]/3.0", "x[0] >= L[0]/3.0 - %g" % TOL).replace("x[0] <= 2.0*L[0]/3.0", "x[0] <= 2.0*L[0]/3.0 + %g" % TOL)
    open(p, 'w').write(s)
    p = os.path.join(dst, 're_vgm_sand_10x10m_2d_c0p1_n.py'); s = open(p).read()
    s += """
stepController = FixedStep
nDTout = ct.NDT
DT = T / nDTout
tnList = [0.0] + [i * DT for i in range(1, nDTout + 1)]
numericalFluxType = Richards_IIPG_exterior
maxNonlinearIts = 100
maxLineSearches = 100
nl_atol_res = 1.0e-12
nnx = nny = ct.NN
triangleOptions = "pAq30Dena%f" % (0.5 * (L[0] / (nnx - 1)) ** 2,)
triangleFlag = 0
"""
    open(p, 'w').write(s)


def run_proteus(work, bin_dir, context):
    for f in os.listdir(work):
        if f.endswith('.h5'):
            os.remove(os.path.join(work, f))
    env = dict(os.environ); env.pop('PYTHONPATH', None); env['PATH'] = bin_dir + os.pathsep + env['PATH']
    r = subprocess.run(['parun', 're_vgm_sand_10x10m_2d_p.py', 're_vgm_sand_10x10m_2d_c0p1_n.py', '-C', context, '-l', '1'], cwd=work, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("proteus failed (%s):\n%s" % (context, (r.stdout + r.stderr)[-1500:]))
    with h5py.File(os.path.join(work, 're_vgm_sand_10x10m_2d.h5')) as f:
        steps = sorted(int(k.split('_t')[1]) for k in f if k.startswith('pressure_head_t'))
        return f['nodesSpatial_Domain0'][:], f['elementsSpatial_Domain0'][:].astype('i'), f['pressure_head_t%d' % steps[-1]][:]


def build(X, conn, lens, beta, interface='node'):
    x = X[:, :2]; Xc, Yc = x[:, 0], x[:, 1]
    top = (Yc == L) & (Xc >= L / 3 - TOL) & (Xc <= 2 * L / 3 + TOL); bottom = (Yc == 0.0); side = (Xc == 0.0) | (Xc == L)
    gD = np.where(top, 0.1, np.where(bottom, 0.0, np.where(side, 0.1 - Yc, 0.0))); fixed = np.where(top | bottom | side)[0]
    ks = dimensionless_ks(5.04)
    cx, cy = x[conn, 0].mean(axis=1), x[conn, 1].mean(axis=1)
    material = ((np.abs(cx - 5) < 2) & (np.abs(cy - 6) < 1)).astype('i') if lens else np.zeros(len(conn), dtype='i')
    model = nd.KUpwindRichards(x, conn, material, alpha=[5.47, 1.0], n_vg=[4.264, 1.6], theta_r=[0.093, 0.08], theta_s=[0.301, 0.43],
                               ks=[[ks, ks], [0.1 * ks, 0.1 * ks]], g=[0.0, -1.0], dirichlet=(fixed, gD[fixed]), beta=beta, interface=interface, quadrature_order=4)
    u0 = np.where(top, 0.1, -Yc); u0[fixed] = gD[fixed]
    return model, u0


def mine(model, u0, steps, kind, bounds='global'):    # 'global': proteus's bounds, whatever the engine's default
    dt = 0.35 / steps; u = u0.copy(); fb = 0
    for _ in range(steps):
        uo = u
        if kind == 'galerkin':
            u, _ = model.newton(uo, uo, dt, 1e-12, 2, maxit=100, system=lambda v: model.galerkin_system(v, uo, dt))
        elif kind == 'semi':
            uL, _ = model.newton(uo, uo, dt, 1e-12, 2)
            u, _ = model.fct_correct(uL, uo, dt, bounds_old=False, bounds='moisture', sat_low=True)
        elif kind == 'implicit':
            u, c = model.implicit_fct_step(uo, dt, tol=1e-12, norm=2, bounds=bounds); fb += c['fallback']
        else:
            u, _ = model.newton(uo, uo, dt, 1e-12, 2)
    return u, fb


CASES = (  # label, proteus context (without NDT/NN), my kind, lens, beta, interface
    ("Galerkin", "STABILIZATION_TYPE=0 FCT=False BETA=0.0", 'galerkin', False, 0.0, 'node'),
    ("Galerkin, beta 1e-4", "STABILIZATION_TYPE=0 FCT=False BETA=0.0001", 'galerkin', False, 1e-4, 'node'),
    ("Galerkin, beta 0.05", "STABILIZATION_TYPE=0 FCT=False BETA=0.05", 'galerkin', False, 0.05, 'node'),
    ("Galerkin, lens", "STABILIZATION_TYPE=0 FCT=False BETA=0.0 LENS=True", 'galerkin', True, 0.0, 'node'),
    ("low-order", "STABILIZATION_TYPE=2 FCT=False BETA=0.0", 'low', False, 0.0, 'node'),
    ("low-order, beta 1e-4", "STABILIZATION_TYPE=2 FCT=False BETA=0.0001", 'low', False, 1e-4, 'node'),
    ("low-order, beta 0.01", "STABILIZATION_TYPE=2 FCT=False BETA=0.01", 'low', False, 0.01, 'node'),
    ("low-order, beta 0.05", "STABILIZATION_TYPE=2 FCT=False BETA=0.05", 'low', False, 0.05, 'node'),
    ("low-order, lens, nodal rule", "STABILIZATION_TYPE=2 FCT=False BETA=0.0 LENS=True", 'low', True, 0.0, 'node'),
    ("low-order, lens, element rule", "STABILIZATION_TYPE=2 FCT=False BETA=0.0 LENS=True", 'low', True, 0.0, 'element'),
    ("semi-implicit FCT", "STABILIZATION_TYPE=2 FCT=True FCT_TYPE='semi-implicit' BETA=0.0", 'semi', False, 0.0, 'node'),
    ("implicit FCT", "STABILIZATION_TYPE=2 FCT=True FCT_TYPE='implicit' BETA=0.0", 'implicit', False, 0.0, 'node'),
    ("implicit FCT, lens", "STABILIZATION_TYPE=2 FCT=True FCT_TYPE='implicit' BETA=0.0 LENS=True", 'implicit', True, 0.0, 'node'),
)


def make_deck_1d(src, dst):
    src = os.path.join(src, 'test_1')
    for name in ('re_vgm_sand_10m_1d_p.py', 're_vgm_sand_10m_1d_c0p1_n.py'):
        shutil.copy(os.path.join(src, name), dst)
    p = os.path.join(dst, 're_vgm_sand_10m_1d_p.py'); s = open(p).read()
    s = s.replace('    ("FCT", True, "apply the flux-corrected-transport limiter"),', """    ("FCT", True, "apply the flux-corrected-transport limiter"),
    ("BETA", 0.0, "compressibility"),
    ("PTOP", -3.0, "head at the top"),
    ("PBOT", -1.0, "head at the bottom"),
    ("TEND", 3.0, "end time"),
    ("NN", 41, "nodes"),""", 1)
    s = s.replace("beta          = 0.0     #same for every scheme, so the comparison is controlled", "beta          = ct.BETA", 1)
    s = s.replace("pondingPressure= -0.75 #0.1", "pondingPressure= ct.PTOP", 1).replace("bottomPressure = -10.0", "bottomPressure = ct.PBOT", 1)
    s = s.replace("            return -10.0\n", "            return ct.PBOT + (ct.PTOP - ct.PBOT) * x[0]\n", 1)
    s = s.replace("T = 24*1/24/timeScale", "T = ct.TEND", 1)
    open(p, 'w').write(s)
    p = os.path.join(dst, 're_vgm_sand_10m_1d_c0p1_n.py'); s = open(p).read()
    open(p, 'w').write(s + "\nnn = ct.NN\n")


def run_proteus_1d(work, bin_dir, context):
    for f in os.listdir(work):
        if f.endswith('.h5'):
            os.remove(os.path.join(work, f))
    env = dict(os.environ); env.pop('PYTHONPATH', None); env['PATH'] = bin_dir + os.pathsep + env['PATH']
    r = subprocess.run(['parun', 're_vgm_sand_10m_1d_p.py', 're_vgm_sand_10m_1d_c0p1_n.py', '-C', context, '-l', '1'], cwd=work, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("proteus failed (%s):\n%s" % (context, (r.stdout + r.stderr)[-1500:]))
    with h5py.File(os.path.join(work, 're_vgm_sand_10m_1d.h5')) as f:
        frames = sorted(int(k.split('_t')[1]) for k in f if k.startswith('pressure_head_t'))
        return (f['nodesSpatial_Domain0'][:, 0], f['pressure_head_t%d' % frames[-1]][:], f['pressure_head_t%d' % (frames[-1] - 100)][:])


def steady(src, bin_dir):
    import richards_nd_test as T
    work = tempfile.mkdtemp(prefix='proteus_ref_steady_')
    make_deck_1d(src, work)
    print("steady compressible column, psi = -1 below and -3 above; max |psi - exact| (rate)   [proteus: max |psi(T) - psi(T - 0.2 T)|]; deck copy in %s" % work)
    for beta in (0.05, 0.5):
        sol = T.compressible_exact(beta); prev = {}
        print("beta = %g" % beta)
        for nn in (21, 41, 81):
            row = []
            for name, stab in (('proteus Galerkin', 0), ('proteus low-order', 2)):
                x, u, uh = run_proteus_1d(work, bin_dir, "STABILIZATION_TYPE=%d FCT=False BETA=%r NN=%d TEND=200.0" % (stab, beta, nn))
                row.append((name, np.abs(u - sol(x)[0]).max(), np.abs(u - uh).max()))
            xs = np.linspace(0, 1, nn)[:, None]; cn = np.stack([np.arange(nn - 1), np.arange(1, nn)], 1)
            mdl = nd.KUpwindRichards(xs, cn, np.zeros(nn - 1, dtype='i'), dirichlet=([0, nn - 1], [-1.0, -3.0]), alpha=[3.35], n_vg=[2.0], theta_r=[0.102], theta_s=[0.368],
                                     ks=[[7.967]], g=[-1.0], beta=beta)
            u0 = -1.0 - 2.0 * xs[:, 0]; ul, _ = mdl.newton(u0, u0, None, tol=1e-11, maxit=100)
            row.append(('richards_nd K-upwind', np.abs(ul - sol(xs[:, 0])).max(), 0.0))
            print("  nn %3d  " % nn + "   ".join("%s %.2e%s" % (n, e, "" if n not in prev else " (%.2f)" % np.log2(prev[n] / e)) for n, e, _ in row) +
                  "   [%.0e / %.0e]" % (row[0][2], row[1][2]))
            for n, e, _ in row:
                prev[n] = e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steady', action='store_true', help='the 1D steady compressible column against the exact solution')
    ap.add_argument('--steps', type=int, default=35)
    ap.add_argument('--nn', type=int, default=31)
    ap.add_argument('--only', default='', help='a case label (exact), or a substring of several')
    ap.add_argument('--mesh', choices=('structured', 'triangle'), default='structured', help="the deck's own Triangle mesh puts the pond's edges inside boundary faces")
    a = ap.parse_args()
    src, bin_dir = os.environ.get('PROTEUS_REF_DECKS'), os.environ.get('PROTEUS_REF_BIN')
    if not (src and bin_dir):
        sys.exit("set PROTEUS_REF_DECKS (a proteus checkout's test/richards) and PROTEUS_REF_BIN (the bin directory with its parun)")
    if a.steady:
        return steady(src, bin_dir)
    work = tempfile.mkdtemp(prefix='proteus_ref_2d_')
    make_deck(src, work)
    print("%d x %d grid, %d steps of %.4g d; deck copy in %s" % (a.nn, a.nn, a.steps, 0.35 / a.steps, work))
    print("%-32s %12s %12s %12s   %s" % ('case', 'max |dpsi|', 'max |dtheta|', 'mean |dpsi|', 'psi min (mine / proteus)'))
    for label, ctx, kind, lens, beta, interface in CASES:
        if a.only != label and (a.only in [l for l, *_ in CASES] or a.only not in label):
            continue
        X, conn, p = run_proteus(work, bin_dir, "%s NDT=%d NN=%d REGULAR=%s" % (ctx, a.steps, a.nn, a.mesh == 'structured'))
        model, u0 = build(X, conn, lens, beta, interface)
        u, fb = mine(model, u0, a.steps, kind)
        d = np.abs(u - p); th = np.abs(model.m(u)[0] - model.m(p)[0])
        print("%-32s %12.2e %12.2e %12.2e   %.4f / %.4f%s" % (label, d.max(), th.max(), d.mean(), u.min(), p.min(), "  (predictor fallbacks %d)" % fb if kind == 'implicit' else ""))


if __name__ == '__main__':
    main()
