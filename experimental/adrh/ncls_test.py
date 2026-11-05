"""The engine's NCLS (ncls.h) against proteus's NCLS kernel on the damBreak case.

The fixture (ncls_damBreak_he0.05.npz, written by ``vof_capture.py OUT model=ncls`` from test/TwoPhaseFlow/damBreakLevelSet.py)
holds one NCLS Newton iteration as proteus ran it: every array its kernel was handed, the residual and CSR Jacobian it
returned, what it wrote for the other models (the lagged shock-capturing viscosity, CFL, grad u at the points, the boundary
trace and its normalized gradient, the interface locator), and the solution after the solve. The engine gets the same
arrays, element-local; its element and face contributions are assembled here, and

* residual, Jacobian and every output above agree with proteus;
* one Newton step with the engine's residual and Jacobian gives proteus's solution, and the engine's residual there is zero.

Usage: ncls_test.py [fixture.npz]    (ADRH_MODULE picks adrh_proto or adrh_kokkos; only adrh_proto has NCLS so far)
"""
import importlib
import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
adrh = importlib.import_module(os.environ.get('ADRH_MODULE', 'adrh_proto'))
TOL = 1e-12
# as in vof_test.py: the metric shock-capturing viscosity amplifies last-bit differences where u is flat, by up to 1e3
TOLS = {'numDiff (next lagged viscosity)': 1e-10}


def assemble(d, u_dof):
    """The engine's global residual and Jacobian at u_dof, plus its point and face outputs."""
    nodes, l2g = d['mesh_dof'], d['mesh_l2g']
    nN, (nE, nD) = nodes.shape[0], l2g.shape
    nS = nD - 1
    assert float(d['useMetrics']) == 1.0 and float(d['MOVING_DOMAIN']) == 0.0
    assert int(d['STABILIZATION_TYPE']) == 0 and int(d['PURE_BDF']) == 0 and int(d['lag_shockCapturing']) == 1
    assert int(d['offset_u']) == 0 and int(d['stride_u']) == 1
    xn = nodes[l2g]
    ue = u_dof[l2g]
    # P1: the basis values at a point are its barycentric coordinates, so the reference point is (w_1, ..., w_nS)
    qp = np.ascontiguousarray(d['u_trial_ref'][:, 1:])
    fields = np.ascontiguousarray(d['velocity'][..., :nS])
    R, J, numDiff, cfl, cut = adrh.ncls_supg(nS, xn, ue, qp, d['dV_ref'], fields, d['q_m_betaBDF'], d['q_numDiff_u_last'],
                                             d['elementDiameter'], float(d['alphaBDF']), float(d['shockCapturingDiffusion']),
                                             float(d['sc_uref']), float(d['sc_alpha']), float(d['sc_drop_time']))
    ebN = d['exteriorElementBoundariesArray']
    eN_f = d['elementBoundaryElementsArray'][ebN, 0]
    k_f = d['elementBoundaryLocalElementBoundariesArray'][ebN, 0]
    xi = np.ascontiguousarray(d['u_trial_trace_ref'][:, :, 1:])
    ffields = np.ascontiguousarray(d['ebqe_velocity_ext'][..., :nS])
    Rf, Jf, uT, gN = adrh.ncls_faces(nS, xn[eN_f], ue[eN_f], k_f, xi, d['dS_ref'], ffields, d['isDOFBoundary_u'],
                                     d['ebqe_bc_u_ext'], d['ebqe_rd_u_ext'])
    r = np.zeros(nN)
    np.add.at(r, d['r_l2g'], R)
    np.add.at(r, d['r_l2g'][eN_f], Rf)
    rows = np.concatenate([np.repeat(l2g, nD, axis=1).ravel(), np.repeat(l2g[eN_f], nD, axis=1).ravel()])
    cols = np.concatenate([np.tile(l2g, nD).ravel(), np.tile(l2g[eN_f], nD).ravel()])
    Jg = sp.csr_matrix((np.concatenate([J.ravel(), Jf.ravel()]), (rows, cols)), shape=(nN, nN))
    locator = np.zeros(nN)
    locator[l2g[cut.astype(bool)].ravel()] = 1.0
    return r, Jg, numDiff, cfl, uT, gN, locator


def grad_p1(d, u_dof):
    """grad u per element (P1), the value proteus stores at every point of the element in q_n."""
    nodes, l2g = d['mesh_dof'], d['mesh_l2g']
    nS = l2g.shape[1] - 1
    X = nodes[l2g][..., :nS]
    Jm = (X[:, 1:, :] - X[:, :1, :]).transpose(0, 2, 1)  # dx/dxi, (nE, nS, nS)
    du = (u_dof[l2g][:, 1:] - u_dof[l2g][:, :1])          # du/dxi
    return np.linalg.solve(Jm.transpose(0, 2, 1), du[..., None])[..., 0]


def rel(a, b):
    scale = max(np.abs(b).max(), 1e-300)
    return np.abs(a - b).max() / scale


def main(argv):
    path = argv[0] if argv else os.path.join(HERE, 'ncls_damBreak_he0.05.npz')
    d = dict(np.load(path))
    nN = d['mesh_dof'].shape[0]
    nS = d['mesh_l2g'].shape[1] - 1
    r, Jg, numDiff, cfl, uT, gN, locator = assemble(d, d['u_dof'])
    Jp = sp.csr_matrix((d['out_globalJacobian'], d['csrColumnOffsets_DofLoops'], d['csrRowIndeces_DofLoops']), shape=(nN, nN))
    gq = d['out_q_n'][..., :nS]
    checks = [
        ('residual', rel(r, d['out_globalResidual'])),
        ('jacobian', abs(Jg - Jp).max() / abs(Jp).max()),
        ('numDiff (next lagged viscosity)', rel(numDiff, d['out_q_numDiff_u'])),
        ('cfl', rel(cfl, d['out_cfl'])),
        ('grad u at the points (P1, per element)', rel(np.broadcast_to(grad_p1(d, d['u_dof'])[:, None, :], gq.shape), gq)),
        ('boundary trace', rel(uT, d['out_ebqe_u'])),
        ('boundary normalized gradient', rel(gN, d['out_ebqe_grad_u'][..., :nS])),
        ('interface locator', rel(locator, d['out_interface_locator'])),
    ]
    # one Newton step (NCLS is linear): proteus's solution, and a zero residual there
    u_new = d['u_dof'] - spla.spsolve(Jg.tocsc(), r)
    r_new = assemble(d, u_new)[0]
    checks.append(('solution after the step', rel(u_new, d['u_next'])))
    checks.append(('residual at the new solution / initial', np.abs(r_new).max() / np.abs(r).max()))
    print('NCLS engine vs proteus, damBreakLevelSet he=%g, t=%.4f, %d nodes, %d elements, %d interface nodes'
          % (float(d['he']), float(d['t']), nN, d['mesh_l2g'].shape[0], int(locator.sum())))
    ok = True
    for name, err in checks:
        tol = TOLS.get(name, TOL)
        print('  %-42s %.2e  %s' % (name, err, 'ok' if err <= tol else 'FAIL'))
        ok &= err <= tol
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
