"""The engine's MCorr on the cut-cell path (mcorr.h steps 1-4) against proteus's MCorr with a solid, on the damBreak case.

The fixture (mcorr_cut_damBreak_he0.05.npz, written by ``vof_capture.py OUT model=mcorr solid=1.0,0.45,0.4``) is one MCorr
Newton iteration as proteus ran it with a circular solid of radius 0.4 at (1.0, 0.45) put into RANS2P's coefficients
(useExact = True, phi_s its signed distance): the arrays MCorr's kernel was handed, the residual and CSR Jacobian it returned
(before MCorr.py masks the inactive rows), the active marks it wrote, the next Newton iterate (after the masked solve) and
proteus's kernel residual there. The solid cuts the water front, so the cut elements carry both the moment-fit H_s and the
level set's interface. Checks:

* the engine's element cases give proteus's isActiveElement, isActiveDOF and isActiveR;
* residual and Jacobian (element integrals with the moment-fit H_s, plus the ghost penalty on the cut faces) agree with
  proteus's kernel, and so do the point outputs;
* after the engine's masking (replace_inactive_rows: MCorr.py's r *= isActiveR, identity rows, inactive DOFs zeroed), one
  Newton step gives proteus's next iterate; the engine's residual there is proteus's.

Tolerance. The moment fit of degree 4 behind H_s amplifies rounding by about 1e5: built with FMA contraction (clang's
default, which is how proteus's MCorr was built for the fixture), the engine gives proteus's residual to 5.3e-15, its
Jacobian to 4.2e-16 and its next iterate to 4.3e-14; built as CMakeLists.txt builds it (ADRH_BITWISE: no contraction), H_s
moves by up to 1.4e-11 on the cut elements, and the residual, the next iterate and the residual there by 2.7e-12, 2.8e-12
and 4.3e-12. The checks that go through H_s therefore use TOL_HS = 1e-10; the rest keep 1e-12.

Usage: mcorr_solid_test.py [fixture.npz]    (ADRH_MODULE picks the module; only adrh_proto has MCorr so far)
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
from mcorr_test import point_fields, rel  # noqa: E402
TOL = 1e-12
TOL_HS = 1e-10   # through the moment-fit H_s (see the docstring)


def assemble(d, u_dof, fields):
    """The engine's kernel residual and Jacobian at u_dof (before the masking), its point outputs and the solid's marks."""
    nodes, l2g = d['mesh_dof'], d['mesh_l2g']
    nN, (nE, nD) = nodes.shape[0], l2g.shape
    nS = nD - 1
    assert int(d['useExact']) == 0 and int(d['useExact_s']) == 1
    assert int(d['offset_u']) == 0 and int(d['stride_u']) == 1 and (d['u_l2g'] == l2g).all() and (d['r_l2g'] == l2g).all()
    qp = np.ascontiguousarray(d['x_ref'][:, :nS])
    ebE = d['elementBoundaryElementsArray']
    (R, J, qu, qr, qn, icase, Hs, faces, Rf, Jf) = adrh.mcorr_solid(
        nS, nodes[l2g], u_dof[l2g], qp, d['dV_ref'], fields, float(d['useMetrics']), float(d['epsFactHeaviside']),
        float(d['epsFactDirac']), float(d['epsFactDiffusion']), d['x_ref'], d['phi_solid_nodes'], d['phi_solid'],
        int(d['useExact_s']), l2g, d['elementBoundariesArray'], ebE, d['elementBoundaryLocalElementBoundariesArray'],
        d['elementBoundaryDiameter'], float(d['ghost_penalty_constant']))
    r = np.zeros(nN)
    np.add.at(r, l2g, R)
    rows, cols, vals = [np.repeat(l2g, nD, axis=1).ravel()], [np.tile(l2g, nD).ravel()], [J.ravel()]
    # the cut faces: side s of face k is element ebE[faces[k], s]
    side = l2g[ebE[faces]]                                    # (nF, 2, nD)
    np.add.at(r, side.ravel(), Rf.ravel())
    nF = len(faces)
    rows.append(np.broadcast_to(side[:, :, :, None, None], (nF, 2, nD, 2, nD)).ravel())
    cols.append(np.broadcast_to(side[:, None, None, :, :], (nF, 2, nD, 2, nD)).ravel())
    vals.append(Jf.ravel())
    Jg = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nN, nN))
    active_el = ((icase == 0) | (icase == 1)).astype(np.int32)
    active = np.zeros(nN)
    active[l2g[active_el == 1].ravel()] = 1.0
    return r, Jg, qu, qr, qn, active_el, active, faces


def main(argv):
    path = argv[0] if argv else os.path.join(HERE, 'mcorr_cut_damBreak_he0.05.npz')
    d = dict(np.load(path))
    nN = d['mesh_dof'].shape[0]
    nS = d['mesh_l2g'].shape[1] - 1
    fields = point_fields(d)
    r, Jg, qu, qr, qn, active_el, active, faces = assemble(d, d['u_dof'], fields)
    rowptr, colind = d['csrRowIndeces_DofLoops'].astype(np.int32), d['csrColumnOffsets_DofLoops'].astype(np.int32)
    Jp = sp.csr_matrix((d['out_globalJacobian'], colind, rowptr), shape=(nN, nN))
    missing = abs(Jg - Jg.multiply(Jp != 0)).max()           # engine entries outside proteus's sparsity
    checks = [
        ('isActiveElement', float(np.abs(active_el - d['out_isActiveElement']).max())),
        ('isActiveDOF, isActiveR', float(max(np.abs(active - d['out_isActiveDOF']).max(),
                                             np.abs(active - d['out_isActiveR']).max()))),
        ('residual (kernel, before the masking)', rel(r, d['out_globalResidual']), TOL_HS),
        ('jacobian (kernel, before the masking)', abs(Jg - Jp).max() / abs(Jp).max()),
        ('ghost-penalty entries inside the sparsity', missing),
        ('q_u', rel(qu, d['out_q_u']) if np.abs(d['out_q_u']).max() > 0 else np.abs(qu).max()),
        ('q_r', rel(qr, d['out_q_r'])),
        ('q_n', rel(qn, d['out_q_n'][..., :nS]) if np.abs(d['out_q_n']).max() > 0 else np.abs(qn).max()),
    ]
    # the masking, on proteus's sparsity, then one Newton step (MCorr has maxLineSearches = 0)
    vals = np.asarray(Jg[np.repeat(np.arange(nN), np.diff(rowptr)), colind]).ravel()
    rm, u = r.copy(), d['u_dof'].copy()
    adrh.replace_inactive_rows(rowptr, colind, vals, rm, active, u, active)
    Jm = sp.csr_matrix((vals, colind, rowptr), shape=(nN, nN))
    u_new = u - spla.spsolve(Jm.tocsc(), rm)
    checks.append(('next Newton iterate (masked solve)', rel(u_new, d['u_next']), TOL_HS))
    r_new, _, qu_new, qr_new, qn_new = assemble(d, d['u_next'], fields)[:5]
    dr_next = np.abs(r_new - d['out_residual_next']).max()
    checks.append(('residual at the next iterate (of |R_0|)', dr_next / np.abs(d['out_globalResidual']).max(), TOL_HS))
    checks.append(('q_u, q_r, q_n at the next iterate', max(rel(qu_new, d['out_next_q_u']), rel(qr_new, d['out_next_q_r']),
                                                             rel(qn_new, d['out_next_q_n'][..., :nS]))))
    inactive = int((active == 0).sum())
    print('MCorr engine vs proteus with a solid (cut-cell path), damBreakLevelSet he=%g, t=%.4f, solid %s: %d nodes, %d '
          'elements (%d inactive), %d inactive DOFs, %d cut faces, |R| %.2e'
          % (float(d['he']), float(d['t']), d['solid'], nN, d['mesh_l2g'].shape[0], int((active_el == 0).sum()), inactive,
             len(faces), np.abs(d['out_globalResidual']).max()))
    ok = inactive > 0 and len(faces) > 0
    for name, err, *tol in checks:
        tol = tol[0] if tol else TOL
        print('  %-42s %.2e  %s' % (name, err, ('ok' if err <= tol else 'FAIL') + (' (tol %.0e)' % tol if tol != TOL else '')))
        ok &= err <= tol
    print('  residual at the next iterate: %.1e absolute, %.1e of its own max' % (dr_next, rel(r_new, d['out_residual_next'])))
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
