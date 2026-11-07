"""The engine's RDLS (rdls.h) against proteus's RDLS kernel on the damBreak case.

The fixture (rdls_damBreak_he0.05.npz, written by ``vof_capture.py OUT model=rdls`` from test/TwoPhaseFlow/damBreakLevelSet.py)
holds one RDLS Newton iteration as proteus ran it: every array its kernel was handed, the residual and CSR Jacobian it
returned, what it wrote at the points and on the boundary (the shock-capturing viscosity, CFL, the boundary trace and its
normalized gradient), the next Newton iterate and proteus's residual there. The engine gets the same arrays, element-local;
the two point fields RDLS derives inside its element loop from mesh data (h_phi, the node diameters interpolated to the
point, and h_e) are computed here in the kernel's order. Checks:

* residual, Jacobian and every output above agree with proteus;
* one Newton step with the engine's residual and Jacobian gives proteus's next iterate, and the engine's residual there is
  proteus's (RDLS is nonlinear, so it is not zero; at the next iterate the interface pin's rows are nonzero too).

Usage: rdls_test.py [fixture.npz]    (ADRH_MODULE picks the module; only adrh_proto has RDLS so far)
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


def point_fields(d):
    """(phi, h_phi, h_e) at every point: RDLS.h's u0 (freezeLevelSet), calculateH_element and elementDiameter."""
    l2g, psi = d['mesh_l2g'], d['u_trial_ref']
    nE, nD = l2g.shape
    nQ = psi.shape[0]
    h_phi = np.zeros((nE, nQ))
    for j in range(nD):  # CompKernel calculateH_element's summation order
        h_phi += d['nodeDiametersArray'][l2g[:, j]][:, None] * psi[None, :, j]
    he = np.broadcast_to(d['elementDiameter'][:, None], (nE, nQ))
    return np.ascontiguousarray(np.stack([d['phi_ls'], h_phi, he], axis=-1))


def assemble(d, u_dof, fields):
    """The engine's global residual and Jacobian at u_dof, plus its point and face outputs."""
    nodes, l2g = d['mesh_dof'], d['mesh_l2g']
    nN, (nE, nD) = nodes.shape[0], l2g.shape
    nS = nD - 1
    assert float(d['useMetrics']) == 1.0 and int(d['ELLIPTIC_REDISTANCING']) == 0 and int(d['freezeLevelSet']) == 1
    assert int(d['useTimeIntegration']) == 0 and int(d['lag_shockCapturing']) == 0 and int(d['lag_subgridError']) == 0
    assert int(d['offset_u']) == 0 and int(d['stride_u']) == 1 and (d['u_l2g'] == l2g).all()
    xn = nodes[l2g]
    ue = u_dof[l2g]
    qp = np.ascontiguousarray(d['u_trial_ref'][:, 1:])
    R, J, numDiff, cfl, pinned = adrh.rdls_supg(nS, xn, ue, d['u_weak_internal_bc_dofs'][l2g], qp, d['dV_ref'], fields,
                                                d['elementDiameter'], float(d['shockCapturingDiffusion']),
                                                float(d['epsFact_redist']), float(d['backgroundDiffusionFactor']),
                                                float(d['weakDirichletFactor']))
    ebN = d['exteriorElementBoundariesArray']
    eN_f = d['elementBoundaryElementsArray'][ebN, 0]
    k_f = d['elementBoundaryLocalElementBoundariesArray'][ebN, 0]
    xi = np.ascontiguousarray(d['u_trial_trace_ref'][:, :, 1:])
    uT, gN = adrh.rdls_faces(nS, xn[eN_f], ue[eN_f], k_f, xi)
    r = np.zeros(nN)
    np.add.at(r, l2g, R)
    Jg = sp.csr_matrix((J.ravel(), (np.repeat(l2g, nD, axis=1).ravel(), np.tile(l2g, nD).ravel())), shape=(nN, nN))
    return r, Jg, numDiff, cfl, uT, gN, pinned


def rel(a, b):
    scale = max(np.abs(b).max(), 1e-300)
    return np.abs(a - b).max() / scale


def main(argv):
    path = argv[0] if argv else os.path.join(HERE, 'rdls_damBreak_he0.05.npz')
    d = dict(np.load(path))
    nN = d['mesh_dof'].shape[0]
    nS = d['mesh_l2g'].shape[1] - 1
    fields = point_fields(d)
    r, Jg, numDiff, cfl, uT, gN, pinned = assemble(d, d['u_dof'], fields)
    Jp = sp.csr_matrix((d['out_globalJacobian'], d['csrColumnOffsets_DofLoops'], d['csrRowIndeces_DofLoops']), shape=(nN, nN))
    checks = [
        ('residual', rel(r, d['out_globalResidual'])),
        ('jacobian', abs(Jg - Jp).max() / abs(Jp).max()),
        ('numDiff (shock capturing, not lagged)', rel(numDiff, d['out_q_numDiff_u'])),
        ('cfl', rel(cfl, d['out_q_cfl'])),
        ('boundary trace', rel(uT, d['out_ebqe_u'])),
        ('boundary normalized gradient', rel(gN, d['out_ebqe_n'][..., :nS])),
    ]
    # one Newton step (no line search): proteus's next iterate, and proteus's residual there
    u_new = d['u_dof'] - spla.spsolve(Jg.tocsc(), r)
    r_new, _, _, _, _, _, pinned_new = assemble(d, d['u_next'], fields)
    checks.append(('next Newton iterate', rel(u_new, d['u_next'])))
    checks.append(('residual at the next iterate', rel(r_new, d['out_residual_next'])))
    pinned_nodes = np.zeros(nN, bool)
    l2g = d['mesh_l2g']
    eps_last = float(d['epsFact_redist']) * fields[:, -1, 1]
    pinned_nodes[l2g[np.abs(d['u_weak_internal_bc_dofs'][l2g]) < eps_last[:, None]]] = True
    print('RDLS engine vs proteus, damBreakLevelSet he=%g, t=%.4f, %d nodes, %d elements, %d pinned nodes (%d element rows)'
          % (float(d['he']), float(d['t']), nN, l2g.shape[0], int(pinned_nodes.sum()), int(pinned.sum())))
    ok = True
    for name, err in checks:
        print('  %-42s %.2e  %s' % (name, err, 'ok' if err <= TOL else 'FAIL'))
        ok &= err <= TOL
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
