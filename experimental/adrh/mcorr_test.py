"""The engine's MCorr (mcorr.h, on the generic element kernel of engine.h) against proteus's MCorr kernel on the damBreak case.

The fixture (mcorr_damBreak_he0.05.npz, written by ``vof_capture.py OUT model=mcorr`` from test/TwoPhaseFlow/damBreakLevelSet.py)
holds one MCorr Newton iteration as proteus ran it: every array its kernel was handed, the residual and CSR Jacobian it
returned, what it wrote at the points for the other models (q_u, q_r, q_n) and the active-DOF marks, the next Newton iterate
and proteus's residual there. The engine gets the same arrays, element-local; h_phi (the node diameters interpolated to the
point), which MCorr derives inside its element loop, is computed here in the kernel's order. Checks:

* residual, Jacobian and the point outputs agree with proteus;
* every DOF is active (useExact_s = 0), so proteus's masking of the residual and the Jacobian rows is the identity;
* one Newton step with the engine's residual and Jacobian gives proteus's next iterate, and the engine's residual and point
  outputs there are proteus's (MCorr is nonlinear through H_eps(phi + u)). Newton cuts the residual by 1e4 in that step, so
  the residual's terms cancel there; its difference is measured against the first residual, the size of the terms.

Usage: mcorr_test.py [fixture.npz]    (ADRH_MODULE picks the module; only adrh_proto has MCorr so far)
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
    """(phi, H_vof, porosity, h_phi, h_e, phi_s) at every point; h_phi in CompKernel calculateH_element's summation order."""
    l2g, psi = d['mesh_l2g'], d['u_trial_ref']
    nE, nD = l2g.shape
    nQ = psi.shape[0]
    h_phi = np.zeros((nE, nQ))
    for j in range(nD):
        h_phi += d['nodeDiametersArray'][l2g[:, j]][:, None] * psi[None, :, j]
    he = np.broadcast_to(d['elementDiameter'][:, None], (nE, nQ))
    return np.ascontiguousarray(np.stack([d['q_phi'], d['q_H'], d['q_porosity'], h_phi, he, d['phi_solid']], axis=-1))


def assemble(d, u_dof, fields):
    """The engine's global residual and Jacobian at u_dof, plus its point outputs."""
    nodes, l2g = d['mesh_dof'], d['mesh_l2g']
    nN, (nE, nD) = nodes.shape[0], l2g.shape
    nS = nD - 1
    assert int(d['useExact']) == 0 and int(d['useExact_s']) == 0
    assert int(d['offset_u']) == 0 and int(d['stride_u']) == 1 and (d['u_l2g'] == l2g).all() and (d['r_l2g'] == l2g).all()
    qp = np.ascontiguousarray(d['x_ref'][:, :nS])
    R, J, qu, qr, qn = adrh.mcorr(nS, nodes[l2g], u_dof[l2g], qp, d['dV_ref'], fields, float(d['useMetrics']),
                                  float(d['epsFactHeaviside']), float(d['epsFactDirac']), float(d['epsFactDiffusion']))
    r = np.zeros(nN)
    np.add.at(r, l2g, R)
    Jg = sp.csr_matrix((J.ravel(), (np.repeat(l2g, nD, axis=1).ravel(), np.tile(l2g, nD).ravel())), shape=(nN, nN))
    return r, Jg, qu, qr, qn


def rel(a, b):
    scale = max(np.abs(b).max(), 1e-300)
    return np.abs(a - b).max() / scale


def main(argv):
    path = argv[0] if argv else os.path.join(HERE, 'mcorr_damBreak_he0.05.npz')
    d = dict(np.load(path))
    nN = d['mesh_dof'].shape[0]
    nS = d['mesh_l2g'].shape[1] - 1
    fields = point_fields(d)
    r, Jg, qu, qr, qn = assemble(d, d['u_dof'], fields)
    Jp = sp.csr_matrix((d['out_globalJacobian'], d['csrColumnOffsets_DofLoops'], d['csrRowIndeces_DofLoops']), shape=(nN, nN))
    active = (d['out_isActiveR'] == 1.0).all() and (d['out_isActiveDOF'] == 1.0).all() and (d['out_isActiveElement'] == 1).all()
    checks = [
        ('residual', rel(r, d['out_globalResidual'])),
        ('jacobian', abs(Jg - Jp).max() / abs(Jp).max()),
        ('q_u', rel(qu, d['out_q_u']) if np.abs(d['out_q_u']).max() > 0 else np.abs(qu).max()),
        ('q_r', rel(qr, d['out_q_r'])),
        ('q_n', rel(qn, d['out_q_n'][..., :nS]) if np.abs(d['out_q_n']).max() > 0 else np.abs(qn).max()),
        ('every DOF and element active', 0.0 if active else 1.0),
    ]
    # one Newton step (MCorr has maxLineSearches = 0): proteus's next iterate, and proteus's residual there
    u_new = d['u_dof'] - spla.spsolve(Jg.tocsc(), r)
    r_new, _, qu_new, qr_new, qn_new = assemble(d, d['u_next'], fields)
    checks.append(('next Newton iterate', rel(u_new, d['u_next'])))
    # Newton has cut the residual by 1e4 there, so the terms cancel: scale the difference by the first residual (the terms' size)
    dr_next = np.abs(r_new - d['out_residual_next']).max()
    checks.append(('residual at the next iterate (of |R_0|)', dr_next / np.abs(d['out_globalResidual']).max()))
    checks.append(('q_u, q_r, q_n at the next iterate', max(rel(qu_new, d['out_next_q_u']), rel(qr_new, d['out_next_q_r']),
                                                             rel(qn_new, d['out_next_q_n'][..., :nS]))))
    cut = int(((d['q_phi'] + 0.0 < 1.5 * fields[..., 3]) & (d['q_phi'] > -1.5 * fields[..., 3])).any(axis=1).sum())
    print('MCorr engine vs proteus, damBreakLevelSet he=%g, t=%.4f, %d nodes, %d elements (%d within eps of the interface), '
          '|R| %.2e, |R| at the next iterate %.2e'
          % (float(d['he']), float(d['t']), nN, d['mesh_l2g'].shape[0], cut, np.abs(d['out_globalResidual']).max(),
             np.abs(d['out_residual_next']).max()))
    ok = True
    for name, err in checks:
        print('  %-42s %.2e  %s' % (name, err, 'ok' if err <= TOL else 'FAIL'))
        ok &= err <= TOL
    print('  residual at the next iterate: %.1e absolute, %.1e of its own max' % (dr_next, rel(r_new, d['out_residual_next'])))
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
