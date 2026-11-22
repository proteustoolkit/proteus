"""Does a longer density-coupled Richards run feel the stale first-step mass (cekees/proteus#306)?

Diagnostic, not a test; run by hand on main and on 1.9.x (the deck and helpers come from the tree's own
test/salinity/test_vdft_compatible.py, the tree is the one on PYTHONPATH):

  python test/richards_first_step_mass_compare.py run edge 1 with.npz 365      # with #306's refresh (the default code)
  python test/richards_first_step_mass_compare.py run edge 0 without.npz 365   # refreshOldTimeMass disabled
  python test/richards_first_step_mass_compare.py compare label,with.npz,without.npz

case 'edge': the levee deck on the explicit edge-flux transport (c0 = 0.5, cin = $CIN, default 0.7, eps = 0.025, so the
coupled density is 1.0125 at t = 0), implicit flow FCT, first dt = the deck's output interval 1e-8 (with TADR's own
2.66e-14 the first step is never accepted whatever the old-time mass, see richards_instr.py). Output times
1, 10, 30, 100, 365 d. It records, per accepted step of the flow, the fluid mass and the head.
"""
import os, sys, tempfile, time
import numpy as np
if sys.argv[1] == 'compare':
    sys.argv = sys.argv[1:]
    MODE = 'compare'
else:
    MODE = 'run'
    sys.argv = sys.argv[1:]

if MODE == 'compare':
    def load(p): return np.load(p)
    for line in sys.argv[1:]:
        tag, a, b = line.split(',')   # with-refresh file, without-refresh file
        A, B = load(a), load(b)
        print('==', tag)
        print('  refresh   : steps=%d solves=%d failed=%d tLast=%g mass(t_end)=%.9g' % (len(A['t']), A['solves'], A['failed'], A['tLast'], A['mass'][-1]))
        print('  no refresh: steps=%d solves=%d failed=%d tLast=%g mass(t_end)=%.9g' % (len(B['t']), B['solves'], B['failed'], B['tLast'], B['mass'][-1]))
        for k in sorted(A.files):
            if k.startswith('u_') and k in B.files:
                d = np.abs(A[k] - B[k]); ref = np.abs(A[k]).max()
                print('  %-10s max|du|=%.3e  rel L2=%.3e  (|u|max %.4g)' % (k, d.max(), np.linalg.norm(A[k]-B[k])/np.linalg.norm(A[k]), ref))
        # mass histories at common times
        ta, tb = A['t'], B['t']
        for T in (1e-8, 2e-8, 1.0, 10.0, 30.0, 100.0, 365.0):
            ia = np.argmin(abs(ta - T)); ib = np.argmin(abs(tb - T))
            if abs(ta[ia]-T) < 1e-6*max(1,T) and abs(tb[ib]-T) < 1e-6*max(1,T):
                print('  t=%-8g mass refresh=%.9g no-refresh=%.9g  diff=%.3e (rel %.2e)' % (T, A['mass'][ia], B['mass'][ib], B['mass'][ib]-A['mass'][ia], (B['mass'][ib]-A['mass'][ia])/A['mass'][ia]))
        print('  c(t_end) max|dc|=%.3e' % np.abs(A['c']-B['c']).max())
    sys.exit(0)

case, refresh, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
tEnd = float(sys.argv[4]) if len(sys.argv) > 4 else 100.0
import proteus
tree = os.path.dirname(os.path.dirname(proteus.__file__))
sys.path.insert(0, os.path.join(tree, 'test', 'salinity'))
import test_vdft_compatible as T
from proteus.kernels import TADR
from proteus.kernels import Richards
from proteus import NumericalSolution, NonlinearSolvers
if not refresh:
    Richards.LevelModel.refreshOldTimeMass = lambda self: None
os.chdir(tempfile.mkdtemp())
T._load(T.CASE_D, T.CASE_D_MODULES)
domain, rp, rn, tp, tn, so = [sys.modules[n] for n in T.CASE_D_MODULES]
if case == 'edge':
    rp.coefficients.FCT = True
    rp.coefficients.FCT_TYPE = 'implicit'
    T._edge_flux_levee_transport(tp, tn, 4, 1, c0=0.5, cin=float(os.environ.get('CIN', '0.7')))
    class R(TADR.RKEV):   # first dt = output interval, so that the run leaves its first step
        def initialize_dt(self, t0, tOut, q):
            TADR.RKEV.initialize_dt(self, t0, tOut, q)
            self.dt = tOut - t0; self.t = t0 + self.dt
            self.substeps = [self.t for i in range(self.nStages)]
    tn.timeIntegration = R
else:
    T._compatible_levee_transport(tp, tn, c0=0.5, cin=0.7)
T._direct_solvers(rn, tn)
outs = [1.0, 10.0, 30.0, 100.0, 365.0]
so.tnList = [0.0, 1.0e-8, 2.0e-8] + [t for t in outs if t <= tEnd]
rec = {'t': [], 'mass': [], 'umax': [], 'umin': []}
snap = {}
nl = {'solves': 0, 'failed': 0}
orig_solve = NonlinearSolvers.Newton.solve
def solve(self, *a, **k):
    r = orig_solve(self, *a, **k)
    if type(self.F).__module__.endswith('Richards'):
        nl['solves'] += 1; nl['failed'] += bool(self.failedFlag)
    return r
NonlinearSolvers.Newton.solve = solve
orig_post = NumericalSolution.NS_base.postStep
def post(self, model, firstSolve=False):
    r = orig_post(self, model, firstSolve) if firstSolve else orig_post(self, model)
    lm = model.levelModelList[-1]
    if type(lm).__module__.endswith('Richards'):
        m = float((lm.q[('m', 0)] * lm.q[('dV_u', 0)]).sum())
        t = float(model.stepController.t_model)
        rec['t'].append(t); rec['mass'].append(m)
        rec['umax'].append(float(lm.u[0].dof.max())); rec['umin'].append(float(lm.u[0].dof.min()))
        for o in so.tnList[1:]:
            if abs(t - o) <= 1e-9 * max(1, o):
                snap['u_%g' % o] = lm.u[0].dof.copy()
    return r
NumericalSolution.NS_base.postStep = post
t0 = time.time()
ns = T._run(so, [rp, tp], [rn, tn], 'q3')
flow = ns.modelList[0].levelModelList[-1]
tr = ns.modelList[1].levelModelList[-1]
np.savez(out, t=np.array(rec['t']), mass=np.array(rec['mass']), umax=np.array(rec['umax']), umin=np.array(rec['umin']),
         c=tr.u[0].dof.copy(), solves=nl['solves'], failed=nl['failed'], tLast=flow.timeIntegration.tLast, **snap)
print('Q3 done case=%s refresh=%d steps=%d solves=%d failed=%d tLast=%g wall=%.0fs' % (case, refresh, len(rec['t']), nl['solves'], nl['failed'], flow.timeIntegration.tLast, time.time() - t0))
