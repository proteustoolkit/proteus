"""The engine's VOF (vof.h) against proteus's VOF kernel on the damBreak case.

The fixture (vof_damBreak_he0.05.npz, written by vof_capture.py from test/TwoPhaseFlow/damBreakLevelSet.py) holds one
VOF Newton iteration as proteus ran it: every array its kernel was handed, the residual and CSR Jacobian it returned, the
shock-capturing viscosity and CFL it wrote, and the solution after the solve. The engine gets the same arrays, element-local,
and its element and face contributions are assembled here; then

* residual, Jacobian, numDiff (next step's lagged viscosity), cfl, boundary flux and upwinded trace agree with proteus;
* one Newton step with the engine's residual and Jacobian gives proteus's solution, and the engine's residual there is zero.

Usage: vof_test.py [fixture.npz]    (ADRH_MODULE picks adrh_proto or adrh_kokkos; only adrh_proto has VOF so far)
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
# The shock-capturing viscosity is c |R(u)| (grad u . G grad u + 1e-12)^(-1/4) (uref sqrt(G:G))^(1/2): where u is flat, R(u) is
# round-off and the factor reaches 1e3, so last-bit differences in the geometry (proteus inverts the 2D map its own way) come
# out amplified. Measured on the fixture: 2.3e-13 absolute (5.5e-12 of the maximum) in an element whose dofs differ by 1.5e-9;
# 1.9e-9 relative wherever numDiff is above 1e-3 of its maximum. The other checks hold to round-off.
TOLS = {'numDiff (next lagged viscosity)': 1e-10, 'solution after the step': 1e-12}


def assemble(d, u_dof):
    """The engine's global residual and Jacobian at u_dof, plus its point outputs."""
    nodes, l2g = d['mesh_dof'], d['mesh_l2g']
    nN, (nE, nD) = nodes.shape[0], l2g.shape
    nS = nD - 1
    assert float(d['useMetrics']) == 1.0 and float(d['MOVING_DOMAIN']) == 0.0 and int(d['STABILIZATION_TYPE']) == 0
    # no solid in this deck: H_s = 1 everywhere (phi_solid is the 'far away' value)
    assert d['phi_solid'].min() > 0 and d['ebqe_phi_s'].min() > 0
    xn = nodes[l2g]
    ue = u_dof[l2g]
    fields = np.concatenate([d['velocity'][..., :nS], d['q_porosity'][..., None]], axis=-1)
    # static mesh: proteus's history term is beta * dV_last / dV with dV_last the same dV
    R, J, numDiff, cfl = adrh.vof_supg(nS, xn, ue, d['x_ref'][:, :nS], d['dV_ref'], fields, d['q_m_betaBDF'],
                                      d['q_numDiff_u_last'], d['elementDiameter'], float(d['alphaBDF']),
                                      float(d['shockCapturingDiffusion']), float(d['sc_uref']), float(d['sc_alpha']),
                                      float(d['sc_drop_time']))
    ebN = d['exteriorElementBoundariesArray']
    eN_f = d['elementBoundaryElementsArray'][ebN, 0]
    k_f = d['elementBoundaryLocalElementBoundariesArray'][ebN, 0]
    # the face points in the element's reference coordinates: for P1 the trace basis values are barycentric coordinates
    xi = np.ascontiguousarray(d['u_trial_trace_ref'][:, :, 1:])
    ffields = np.concatenate([d['ebqe_velocity_ext'][..., :nS], d['ebqe_porosity_ext'][..., None]], axis=-1)
    Rf, Jf, flux, uUp = adrh.vof_faces(nS, xn[eN_f], ue[eN_f], k_f, xi, d['dS_ref'], ffields, d['isDOFBoundary_u'],
                                       d['ebqe_bc_u_ext'], d['isFluxBoundary_u'], d['ebqe_bc_flux_u_ext'])
    r = np.zeros(nN)
    np.add.at(r, d['r_l2g'], R)
    np.add.at(r, d['r_l2g'][eN_f], Rf)
    rows = np.concatenate([np.repeat(l2g, nD, axis=1).ravel(), np.repeat(l2g[eN_f], nD, axis=1).ravel()])
    cols = np.concatenate([np.tile(l2g, nD).ravel(), np.tile(l2g[eN_f], nD).ravel()])
    Jg = sp.csr_matrix((np.concatenate([J.ravel(), Jf.ravel()]), (rows, cols)), shape=(nN, nN))
    return r, Jg, numDiff, cfl, flux, uUp


def rel(a, b):
    scale = max(np.abs(b).max(), 1e-300)
    return np.abs(a - b).max() / scale


def main(argv):
    path = argv[0] if argv else os.path.join(HERE, 'vof_damBreak_he0.05.npz')
    d = dict(np.load(path))
    nN = d['mesh_dof'].shape[0]
    r, Jg, numDiff, cfl, flux, uUp = assemble(d, d['u_dof'])
    Jp = sp.csr_matrix((d['out_globalJacobian'], d['csrColumnOffsets_DofLoops'], d['csrRowIndeces_DofLoops']), shape=(nN, nN))
    checks = [
        ('residual', rel(r, d['out_globalResidual'])),
        ('jacobian', abs(Jg - Jp).max() / abs(Jp).max()),
        ('numDiff (next lagged viscosity)', rel(numDiff, d['out_q_numDiff_u'])),
        ('cfl', rel(cfl, d['out_cfl'])),
        ('boundary flux', rel(flux, d['out_ebqe_flux'])),
        ('upwinded boundary trace', rel(uUp, d['out_ebqe_u'])),
    ]
    # one Newton step (VOF is linear): proteus's solution, and a zero residual there
    u_new = d['u_dof'] - spla.spsolve(Jg.tocsc(), r)
    r_new = assemble(d, u_new)[0]
    checks.append(('solution after the step', rel(u_new, d['u_next'])))
    checks.append(('residual at the new solution / initial', np.abs(r_new).max() / np.abs(r).max()))
    print('VOF engine vs proteus, damBreakLevelSet he=%g, t=%.4f, %d nodes, %d elements'
          % (float(d['he']), float(d['t']), nN, d['mesh_l2g'].shape[0]))
    ok = True
    for name, err in checks:
        tol = TOLS.get(name, TOL)
        print('  %-42s %.2e  %s' % (name, err, 'ok' if err <= tol else 'FAIL'))
        ok &= err <= tol
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
