"""DIAGNOSTIC (branch richards-r2; the plugin of the never-merged richards-nan-diag branch plus ARGMAX lines): pytest plugin that traces the Richards implicit FCT
levee test and prints, to stderr, compact lines starting with RICHARDS_NAN_DIAG.

Run:  PYTHONPATH=test python -m pytest -s -p richards_instr <the levee test>

What it prints
  NEWTON   one line per Newton.solve: model, t, dt, fct phase, iterations, ||r0||, ||r||, failed flags, and the
           history (max|u|, ||r||) of every residual evaluation of the solve (first 10 and last 3)
  DT       every change of dt (or of t) between two solves of one model
  MASS     the first residual calls of Richards: old-time mass (m_last) against the mass at the iterate and at u_old
  FIRST    the first non-finite quantity of a kind (residual, Jacobian entry, iterate, dt, du, matrix), with the model,
           time step, solve and iteration, the first bad node and the values that produced it, every non-finite array of
           the model, and the last residual calls before it
  DT0      where the first dt comes from (system step controller, transport's RKEV edge_based_cfl, its M_low = 0 state)
           RICHARDS_FIRST_DT=<dt> (experiment) sets the transport's first dt, hence the system's, to <dt>
  ARGMAX   (RICHARDS_ARGMAX=1) at the last residual of every Richards Newton solve: the node with the largest |r|, its
           x, u, u_old, the strong-Dirichlet value g(x, t) and bc_mask, r, the storage term (m - m_last)/dt there, the
           number of nodes with |r| > half the maximum and how many of them are a Dirichlet node or touch one
"""
import os
import sys

import numpy as np

TAG = 'RICHARDS_NAN_DIAG'
S = {'lines': 0, 'ncall': {}, 'ring': [], 'first': set(), 'cur': None, 'solve': 0, 'last_dt': {},
     'last_model': None, 'last_phase': None}
MAXLINES = int(os.environ.get('RICHARDS_NAN_DIAG_MAXLINES', '2500'))


def P(*a):
    S['lines'] += 1
    if S['lines'] > MAXLINES:
        return
    print(TAG, *a, file=sys.stderr, flush=True)
    if S['lines'] == MAXLINES:
        print(TAG, 'LINE LIMIT REACHED', file=sys.stderr, flush=True)


def name(model):
    return type(model).__module__.split('.')[-1]


def fin(x):
    return bool(np.all(np.isfinite(x)))


def st(x):
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    nb = int((~ok).sum())
    if ok.any():
        return '%d/%.3e/%.3e' % (nb, x[ok].min(), x[ok].max())   # nbad/min/max
    return '%d/allbad' % nb


def amax(x):
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    return np.abs(x[ok]).max() if ok.any() else float('nan')


def tinfo(m):
    ti = m.timeIntegration
    return 't=%.9e dt=%.4e' % (ti.t, ti.dt)


def ring_add(s):
    S['ring'].append(s)
    del S['ring'][:-8]


def scan_model(m, i=None):
    """every non-finite ndarray attribute of the model, its time integration and coefficients"""
    out = []
    for owner, obj in (('model', m), ('ti', m.timeIntegration), ('coef', m.coefficients)):
        for k, v in list(vars(obj).items()):
            vs = [(k, v)]
            if isinstance(v, dict):
                vs = [('%s[%s]' % (k, kk), vv) for kk, vv in v.items()]
            for kk, vv in vs:
                if isinstance(vv, np.ndarray) and vv.dtype.kind == 'f' and vv.size and not fin(vv):
                    bad = np.flatnonzero(~np.isfinite(vv.ravel()))
                    out.append('%s.%s nbad=%d first=%d(%r)' % (owner, kk, bad.size, bad[0], vv.ravel()[bad[0]]))
                elif isinstance(vv, float) and not np.isfinite(vv):
                    out.append('%s.%s=%r' % (owner, kk, vv))
    return out


NODE_FIELDS = ('u_dof_old', 'mn', 'mLow', 'fluxCorrection', 'm_fct', 'ML', 'alpha_nv', 'Rpos', 'Rneg',
               'rho_dof_in', 'rho_nodal', 'bc_mask', 'nodalSource', 'psi_old')


def node_values(m, i):
    v = []
    u = m.u[0].dof
    v.append('u=%r' % (u[i],))
    for k in NODE_FIELDS:
        a = getattr(m, k, None)
        if isinstance(a, np.ndarray) and a.ndim == 1 and a.shape[0] > i and a.shape[0] == u.shape[0]:
            v.append('%s=%r' % (k, a[i]))
    ti = m.timeIntegration
    for k in ('m_last', 'm_tmp', 'beta_bdf'):
        a = getattr(ti, k, None)
        if a is not None:
            try:
                v.append('%s=%r' % (k, a[0][i]))
            except Exception:
                pass
    return ' '.join(v)


def bc_info(m):
    """Dirichlet dofs of the flow model: ({dof: g(x, t)}, mask of them, mask of dofs touching one, strong?).
    Strong (forceStrongConditions) pins the dof; otherwise the condition is weak, on the exterior faces whose
    numericalFlux.isDOFBoundary is set, and a dof is a Dirichlet dof if it is a node of such a face."""
    c = S.get('bc')
    if c is None or c[0] is not m:
        n = m.u[0].dof.shape[0]
        en = m.mesh.elementNodesArray
        bc = {}
        strong = bool(m.dirichletConditionsForceDOF)
        if strong:
            d = m.dirichletConditionsForceDOF[0]
            bc = {k: (lambda t, g=g, x=d.DOFBoundaryPointDict[k]: g(x, t)) for k, g in d.DOFBoundaryConditionsDict.items()}
        else:
            nf = m.numericalFlux
            g_of = nf.DOFBoundaryConditionsDictList[0]
            mesh = m.mesh
            for (ebNE, k), g in g_of.items():
                ebN = mesh.exteriorElementBoundariesArray[ebNE]
                for node in mesh.elementBoundaryNodesArray[ebN]:
                    x = mesh.nodeArray[node]
                    bc[int(node)] = (lambda t, g=g, x=x: g(x, t))
        isbc = np.zeros(n, bool)
        isbc[[k for k in bc if k < n]] = True
        touch = np.zeros_like(isbc)
        touch[en[isbc[en].any(axis=1)].ravel()] = True
        c = S['bc'] = (m, bc, isbc, touch, strong)
    return c[1], c[2], c[3]


def argmax_record(m, u, r):
    bc, isbc, touch = bc_info(m)
    ti = m.timeIntegration
    a = np.abs(r)
    i = int(np.argmax(a))
    nbig = a > 0.5 * a[i]
    rec = {'i': i, 'x': m.mesh.nodeArray[i][:2].tolist(), 'u': float(u[i]), 'u_old': float(m.u_dof_old[i]),
           'r': float(r[i]), 'rmax': float(a[i]), 'bc': None, 'bc_mask': float(m.bc_mask[i]),
           't': float(ti.t), 'dt': float(ti.dt), 'nbig': int(nbig.sum()), 'nbig_bc': int((nbig & isbc).sum()),
           'nbig_touch': int((nbig & touch).sum()), 'nbc': int(isbc.sum()), 'strong': S['bc'][4], 'rnorm': float(np.linalg.norm(r)),
           'r_bc_nodes': float(a[isbc].max()), 'r_touch_free': float(a[touch & ~isbc].max()),
           'r_far': float(a[~touch].max())}
    if i in bc and isbc[i]:
        rec['bc'] = float(bc[i](ti.t))
    n = np.array(sorted(k for k in bc if k < u.shape[0]))
    g = np.array([bc[k](ti.t) for k in n])
    rec['bc_g_minus_u'] = float(np.abs(g - u[n]).max())
    rec['bc_g_minus_uold'] = float(np.abs(g - m.u_dof_old[n]).max())
    ML = getattr(m, 'ML', None)
    mn = getattr(m, 'mn', None)
    if ML is not None and mn is not None and ML.shape[0] > i and mn.shape[0] > i:
        # the round-off of the stored mass, ML_i m_i eps, divided by dt, against |r_i|
        rec['ML'] = float(ML[i])
        rec['mn'] = float(mn[i])
        rec['roundoff'] = float(np.finfo(float).eps * ML[i] * abs(mn[i]) / ti.dt)
        rec['ratio'] = float(abs(r[i]) / rec['roundoff']) if rec['roundoff'] > 0 else float('nan')
    mlast = ti.m_last[0]
    mtmp = ti.m_tmp[0]
    if mlast.ndim == 1 and mlast.shape[0] == u.shape[0]:
        rec['m_tmp'] = float(mtmp[i])
        rec['m_last'] = float(mlast[i])
        rec['stor'] = float((mtmp[i] - mlast[i]) / ti.dt)
    return rec


def argmax_line(rec):
    return ('node %(i)d x=%(x)s u=%(u).6g u_old=%(u_old).6g bc=%(bc)s bc_mask=%(bc_mask)g r=%(r).4e |r|max=%(rmax).4e '
            'dt=%(dt).4e t=%(t).6e ||r||=%(rnorm).4e; |r|>half max: %(nbig)d nodes (%(nbig_bc)d Dirichlet, %(nbig_touch)d touch one, '
            'of %(nbc)d Dirichlet dofs, strong=%(strong)s); max|r| at Dirichlet %(r_bc_nodes).3e, at free nodes touching one %(r_touch_free).3e, '
            'elsewhere %(r_far).3e; max|g-u| %(bc_g_minus_u).3e max|g-u_old| %(bc_g_minus_uold).3e' % rec
            + (' ML=%(ML).4g mn=%(mn).5g eps*ML*mn/dt=%(roundoff).3e |r|/that=%(ratio).3g' % rec if 'ratio' in rec else '')
            + (' stor=(m-m_last)/dt=%(stor).4e m=%(m_tmp).6g m_last=%(m_last).6g' % rec if 'stor' in rec else ''))


def first_event(kind, m, it_info, i=None):
    """print once per (kind, model): the full context of the first non-finite quantity"""
    key = (kind, name(m))
    if key in S['first']:
        return
    S['first'].add(key)
    ti = m.timeIntegration
    P('FIRST', kind, name(m), tinfo(m), 'phase=%s' % getattr(m, 'fct_phase', None), 'solve#%d' % S['solve'],
      'residual_calls=%d' % S['ncall'].get(name(m), 0), it_info)
    P('FIRST   alpha_bdf=%r dt_last=%r' % (getattr(ti, 'alpha_bdf', None), getattr(ti, 'dt_last', None)),
      'nonfinite:', '; '.join(scan_model(m)) or 'none')
    if i is not None:
        P('FIRST   node %d x=%s: %s' % (i, m.mesh.nodeArray[i][:3].tolist(), node_values(m, i)))
    for s in S['ring']:
        P('FIRST   ring:', s)


def newton_iteration_hint():
    c = S['cur']
    return 'newton_res_eval#%d' % (len(c['hist']) if c else -1)


def install():
    from proteus import LinearSolvers, NonlinearSolvers
    from proteus.kernels import Richards
    from proteus.kernels import TADR

    for mod in (Richards, TADR):
        LM = mod.LevelModel
        if getattr(LM, '_rnd_patched', False):
            continue
        LM._rnd_patched = True

        def make(LM=LM):
            orig_res = LM.getResidual
            orig_jac = LM.getJacobian

            def getResidual(self, u, r):
                nm = name(self)
                S['ncall'][nm] = S['ncall'].get(nm, 0) + 1
                nc = S['ncall'][nm]
                u_in_fin = fin(u)
                out = orig_res(self, u, r)
                rn = float(np.linalg.norm(r)) if fin(r) else float('nan')
                ti = self.timeIntegration
                phase = getattr(self, 'fct_phase', None)
                S['last_model'], S['last_phase'] = nm, phase
                line = '%s#%d %s phase=%s |u|max=%.3e r(nbad/min/max)=%s ||r||=%.4e' % (
                    nm, nc, tinfo(self), phase, amax(u), st(r), rn)
                ring_add(line)
                if S['cur'] is not None and S['cur']['model'] == nm:
                    S['cur']['hist'].append((amax(u), rn))
                if nc <= 40:
                    P('RES', line, 'mLow=%s fluxCorr=%s' % (st(getattr(self, 'mLow', 0.0)),
                                                              st(getattr(self, 'fluxCorrection', 0.0))))
                if nm == 'Richards' and os.environ.get('RICHARDS_ARGMAX') and fin(r):
                    S['argmax'] = argmax_record(self, u, r)
                if nm == 'Richards' and nc <= 3:
                    mlast = ti.m_last[0]
                    mtmp = ti.m_tmp[0]
                    q_rho = self.q['rho'] if 'rho' in self.q else None
                    P('MASS', 'call#%d' % nc, tinfo(self),
                      'm_last(nbad/min/max)=%s m(u)=%s max|m(u)-m_last|=%.4e (m(u)-m_last)/dt max=%.4e' % (
                          st(mlast), st(mtmp), amax(mtmp - mlast), amax(mtmp - mlast) / max(ti.dt, 1e-300)),
                      'q_rho=%s' % (st(q_rho) if q_rho is not None else None),
                      'u_dof_old=%s' % st(self.u_dof_old), '|u-u_old|max=%.3e' % amax(u - self.u_dof_old))
                if not u_in_fin:
                    bad = np.flatnonzero(~np.isfinite(u))
                    first_event('NONFINITE_ITERATE_INPUT_TO_RESIDUAL', self, newton_iteration_hint(), int(bad[0]))
                if not fin(r):
                    bad = np.flatnonzero(~np.isfinite(r))
                    first_event('NONFINITE_RESIDUAL', self, newton_iteration_hint(), int(bad[0]))
                return out

            def getJacobian(self, jacobian):
                out = orig_jac(self, jacobian)
                rowptr, colind, nz = jacobian.getCSRrepresentation()
                nm = name(self)
                if S['ncall'].get('jac_' + nm, 0) < 12:
                    P('JAC', nm, tinfo(self), 'phase=%s' % getattr(self, 'fct_phase', None), 'nz(nbad/min/max)=%s' % st(nz))
                S['ncall']['jac_' + nm] = S['ncall'].get('jac_' + nm, 0) + 1
                if not fin(nz):
                    bad = ~np.isfinite(nz)
                    rows = np.repeat(np.arange(len(rowptr) - 1), np.diff(rowptr))
                    badrows = np.unique(rows[bad])
                    i = int(badrows[0])
                    first_event('NONFINITE_JACOBIAN', self, '%d bad entries of %d, %d bad rows of %d, first rows %s; row %d nz=%s' % (
                        bad.sum(), nz.size, badrows.size, len(rowptr) - 1, badrows[:6].tolist(), i,
                        nz[rowptr[i]:rowptr[i + 1]].tolist()), i)
                return out

            LM.getResidual = getResidual
            LM.getJacobian = getJacobian
        make()

    # ---- Richards.Coefficients.preStep: the density and the old-time mass around it --------------------------------
    C = Richards.Coefficients
    if not getattr(C, '_rnd_patched', False):
        C._rnd_patched = True
        orig_pre = C.preStep

        def rinfo(c):
            m = getattr(c, 'model', None)
            if m is None:
                return 'no model'
            ti = m.timeIntegration
            qr = m.q['rho'] if 'rho' in m.q else None
            return ('density_model=%s exportTransportFluxes=%s use_rho_nodal=%s q_rho=%s rho_dof_in=%s rho_nodal=%s '
                    'm_last=%s mn=%s u_dof_old=%s' % (c.density_model, getattr(c, 'exportTransportFluxes', None),
                                                    m.use_rho_nodal, st(qr) if qr is not None else None, st(m.rho_dof_in),
                                                    st(m.rho_nodal), st(ti.m_last[0]), st(m.mn), st(m.u_dof_old)))

        def preStep(self, t, firstStep=False):
            n = S['ncall'].get('prestep', 0)
            S['ncall']['prestep'] = n + 1
            if n < 6:
                P('PRESTEP before t=%.6e firstStep=%s' % (t, firstStep), rinfo(self))
            out = orig_pre(self, t, firstStep)
            if os.environ.get('RICHARDS_FIX_EXPERIMENT') and firstStep and getattr(self, 'model', None) is not None \
                    and self.density_model is not None:
                # EXPERIMENT (what the fix does): the first step's old-time mass at the coupled density
                lm = self.model
                ti = lm.timeIntegration
                before = ti.m_last[0].copy()
                dof = lm.u[0].dof.copy()
                r = np.zeros(dof.shape, 'd')
                lm.getResidual(lm.u_dof_old.copy(), r)
                ti.m_last[0][:] = ti.m_tmp[0]
                lm.u[0].dof[:] = dof
                if n < 6:
                    P('FIXEXP m_last refreshed: max|new-old|=%.4e' % amax(ti.m_last[0] - before))
            if n < 6:
                P('PRESTEP after ', rinfo(self))
            return out
        C.preStep = preStep

    # ---- linear solver: matrix and du --------------------------------------------------------------------------
    orig_prep = LinearSolvers.LU.prepare
    orig_solve = LinearSolvers.LU.solve

    def prepare(self, b=None, newton_its=None):
        if type(self.L).__name__ == 'SparseMatrix':
            rowptr, colind, nz = self.L.getCSRrepresentation()
            if not fin(nz):
                P('LU.prepare NONFINITE MATRIX n=%d model(last residual)=%s phase=%s nz(nbad/min/max)=%s b=%s' % (
                    self.n, S['last_model'], S['last_phase'], st(nz), st(b) if b is not None else None))
        return orig_prep(self, b, newton_its)

    def solve(self, u, *a, **k):
        out = orig_solve(self, u, *a, **k)
        if not fin(u):
            b = k.get('b')
            P('LU.solve du NONFINITE n=%d du=%s b=%s failed=%s' % (self.n, st(u), st(b) if b is not None else None,
                                                                   self.failed()))
        return out

    LinearSolvers.LU.prepare = prepare
    LinearSolvers.LU.solve = solve

    # ---- one line per Newton solve ------------------------------------------------------------------------------
    def wrap_newton(cls):
        if 'solve' not in vars(cls) or getattr(cls.solve, '_rnd', False):
            return
        orig = cls.solve

        def newton_solve(self, u, *a, **k):
            F = self.F
            nm = name(F)
            if nm not in ('Richards', 'TADR'):
                return orig(self, u, *a, **k)
            S['solve'] += 1
            ti = F.timeIntegration
            key = nm
            last = S['last_dt'].get(key)
            if last is None or last != (ti.t, ti.dt):
                P('DT', nm, 'solve#%d' % S['solve'], 'phase=%s' % getattr(F, 'fct_phase', None),
                  'before: t=%.9e dt=%.4e' % last if last else 'first', '-> t=%.9e dt=%.4e' % (ti.t, ti.dt))
                S['last_dt'][key] = (ti.t, ti.dt)
            if not (np.isfinite(ti.dt) and ti.dt > 0):
                first_event('NONFINITE_OR_NONPOSITIVE_DT', F, 'at Newton.solve entry')
            if not fin(u):
                first_event('NONFINITE_ITERATE_AT_SOLVE_ENTRY', F, 'at Newton.solve entry', int(np.flatnonzero(~np.isfinite(u))[0]))
            outer = S['cur']
            S['cur'] = {'model': nm, 'hist': []}
            try:
                return orig(self, u, *a, **k)
            finally:
                cur = S['cur']
                h = cur['hist']
                hs = h if len(h) <= 13 else h[:10] + [None] + h[-3:]
                if S['solve'] <= 400 or S['solve'] % 25 == 0 or not fin(u):
                    P('NEWTON', nm, 'solve#%d' % S['solve'], tinfo(F), 'phase=%s' % getattr(F, 'fct_phase', None),
                      'its=%s' % getattr(self, 'its', None), 'norm_r0=%.4e norm_r=%.4e' % (
                          getattr(self, 'norm_r0', float('nan')), getattr(self, 'norm_r', float('nan'))),
                      'failedFlag=%s linFailed=%s' % (getattr(self, 'failedFlag', None), getattr(self, 'linearSolverFailed', None)),
                      '|u|max_out=%.3e' % amax(u), 'hist(|u|max,||r||)=' + ' '.join(
                          '..' if e is None else '(%.2e,%.2e)' % e for e in hs))
                if nm == 'Richards' and S.get('argmax'):
                    P('ARGMAX', 'solve#%d' % S['solve'], argmax_line(S['argmax']))
                    S['argmax'] = None
                S['cur'] = outer
        newton_solve._rnd = True
        cls.solve = newton_solve

    def all_subclasses(c):
        for s in c.__subclasses__():
            yield s
            yield from all_subclasses(s)

    wrap_newton(NonlinearSolvers.Newton)
    # ---- where the first dt comes from: the transport's BackwardEuler_cfl ---------------------------------------------
    from proteus import TimeIntegration
    BE = TimeIntegration.BackwardEuler_cfl
    if not getattr(BE, '_rnd_patched', False):
        BE._rnd_patched = True
        orig_choose = BE.choose_dt

        def choose_dt(self):
            out = orig_choose(self)
            n = S['ncall'].get('choose_dt', 0)
            S['ncall']['choose_dt'] = n + 1
            P('DT0 choose_dt call#%d class=%s dt=%.5e dtLast=%s cfl keys=%s' % (n + 1, type(self).__name__, self.dt, self.dtLast, list(self.cfl)))
            if n < 4 and getattr(self, 'cfl', None):
                try:
                    c = self.cfl[0]
                    tr = self.transport
                    e, k = np.unravel_index(int(np.argmax(c)), c.shape)
                    v = tr.q_v[e, k] if getattr(tr, 'q_v', None) is not None else None
                    h = tr.mesh.elementDiameterArray[e]
                    P('DT0 choose_dt#%d BackwardEuler_cfl runCFL=%g max cell cfl=%.4e (element %d qp %d, h=%.4g, x=%s, |v|=%s) '
                      'edge_based_cfl max=%.4e -> dt=%.5e (dtLast=%s)' % (
                          n + 1, self.runCFL, c.max(), e, k, h, tr.q['x'][e, k][:2].tolist(),
                          None if v is None else '%.4e' % np.linalg.norm(v),
                          tr.edge_based_cfl.max(), self.dt, self.dtLast))
                    if v is not None:
                        vm = np.linalg.norm(tr.q_v, axis=-1)
                        P('DT0   |v| of the coupling velocity: min/max %.4e/%.4e; flow q_velocity from the Richards model'
                          % (vm.min(), vm.max()))
                except Exception as ex:
                    P('DT0 probe failed: %r' % (ex,))
            return out
        BE.choose_dt = choose_dt
    RK = TADR.RKEV
    if '_rnd_patched' not in vars(RK):
        RK._rnd_patched = True
        orig_rk = RK.choose_dt

        def rk_choose_dt(self):
            out = orig_rk(self)
            n = S['ncall'].get('rk_choose', 0)
            S['ncall']['rk_choose'] = n + 1
            if n == 0 and os.environ.get('RICHARDS_FIRST_DT'):
                # EXPERIMENT: the first system dt is this instead of runCFL / (the CFL of an M_low = 0 state)
                self.dt = float(os.environ['RICHARDS_FIRST_DT'])
                self.t = self.tLast + self.dt
                self.substeps = [self.t for i in range(self.nStages)]
                P('DT0 EXPERIMENT first dt forced to %.4e' % self.dt)
            if n < 3:
                try:
                    tr = self.transport
                    c = tr.edge_based_cfl
                    i = int(np.argmax(c))
                    vm = getattr(tr, 'vModel', None)
                    P('DT0 RKEV.choose_dt#%d runCFL=%g -> dt=%.5e; edge_based_cfl max=%.4e at node %d x=%s; ML=%.4e M_low=%.6g '
                      'M_dof=%.6g nodalBoundaryFlux=%.4e; flow model attached=%s; _use_edge_flux=%s; transport dt=%.4e t=%.4e' % (
                          n + 1, self.runCFL, self.dt, c.max(), i, tr.mesh.nodeArray[i][:2].tolist(), tr.ML[i], tr.M_low[i],
                          tr.M_dof[i], tr.nodalBoundaryFlux[i], vm is not None, getattr(tr, '_use_edge_flux', None), tr.timeIntegration.dt, tr.timeIntegration.t))
                    P('DT0   edge_based_cfl: nodes above 1e6: %d, above 1e10: %d, median %.3e; max|nodalBoundaryFlux|=%.4e; '
                      'max|M_low-M_dof|=%.4e' % ((c > 1e6).sum(), (c > 1e10).sum(), np.median(c), amax(tr.nodalBoundaryFlux),
                                               amax(tr.M_low - tr.M_dof)))
                except Exception as ex:
                    P('DT0 RKEV probe failed: %r' % (ex,))
            return out
        RK.choose_dt = rk_choose_dt
    from proteus import SplitOperator
    for cname in ('Sequential_MinModelStep',):
        SC = getattr(SplitOperator, cname)
        orig_init = SC.initialize_dt_system

        def init_dt_system(self, t0, tOut, orig_init=orig_init):
            P('DT0 initialize_dt_system t0=%g tOut=%g: model dt_model before = %s' % (
                t0, tOut, [(m.name, type(m.stepController).__name__, type(m.levelModelList[-1].timeIntegration).__name__,
                            m.stepController.dt_model) for m in self.modelList]))
            out = orig_init(self, t0, tOut)
            P('DT0 initialize_dt_system -> dt_system=%.5e' % self.dt_system)
            return out
        SC.initialize_dt_system = init_dt_system
        orig_choose_sys = SC.choose_dt_system

        def choose_dt_system(self, orig_choose_sys=orig_choose_sys):
            n = S['ncall'].get('choose_sys', 0)
            S['ncall']['choose_sys'] = n + 1
            if n < 4:
                P('DT0 choose_dt_system#%d: dt_model before = %s' % (n + 1, [(m.name, m.stepController.dt_model) for m in self.modelList]))
            return orig_choose_sys(self)
        SC.choose_dt_system = choose_dt_system
    P('plugin installed')


def pytest_configure(config):
    install()
