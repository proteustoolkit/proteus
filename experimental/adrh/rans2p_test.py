"""The engine's RANS2P element integrals (rans2p.h) against proteus's RANS2P2D kernel on the damBreak case.

The fixture (rans2p_damBreak_he0.05.npz, written by ``vof_capture.py OUT model=rans2p``) holds RANS2P's first Newton iteration
at t >= 0.05 as proteus ran it: the arguments its kernel was handed, its residual, CSR Jacobian and point outputs, and the
same residual and Jacobian with the exterior faces switched off (the kernel called again with
nExteriorElementBoundaries_global = 0), which is what the engine reproduces so far. Checks:

* the configuration is the one rans2p.h implements (what it leaves out is zero in the deck);
* the element residual and Jacobian (pressure and two velocity components, interlaced as proteus numbers them) agree with
  proteus's, and so do the point outputs (velocity, the next shock-capturing viscosity, the CFL);
* the exterior faces' share of proteus's residual and Jacobian, which the engine does not compute yet, is reported.

Usage: rans2p_test.py [fixture.npz]    (ADRH_MODULE picks the module; only adrh_proto has RANS2P so far)
"""
import importlib
import os
import sys

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
adrh = importlib.import_module(os.environ.get('ADRH_MODULE', 'adrh_proto'))
TOL = 1e-12
PARAMS = ('rho_0', 'nu_0', 'rho_1', 'nu_1', 'g', 'alphaBDF', 'Ct_sge', 'Cd_sge', 'C_dc', 'useMetrics', 'hFactor', 'useVF',
          'MOMENTUM_SGE', 'PRESSURE_SGE', 'VELOCITY_SGE')


def check_configuration(d):
    """What rans2p.h leaves out must be zero (or off) in the fixture."""
    zero = ('q_dragAlpha', 'q_dragBeta', 'q_mass_source', 'forcex', 'forcey', 'q_eddy_viscosity_last', 'mesh_velocity_dof')
    for k in zero:
        assert not np.any(d[k]), k
    flags = {'NONCONSERVATIVE_FORM': 1.0, 'MOVING_DOMAIN': 0.0, 'useRBLES': 0.0, 'PRESSURE_PROJECTION_STABILIZATION': 0.0,
             'nParticles': 0, 'useExact': 0, 'turbulenceClosureModel': 0, 'useVF': 1.0}
    for k, v in flags.items():
        assert float(d[k]) == v, (k, d[k])
    assert (d['phi_solid'] > 1.0).all()                        # H_s = 1: no solid
    sd = {'u_u': ([0, 1, 2], [0, 1]), 'u_v': ([0, 0, 1], [0]), 'v_u': ([0, 1, 1], [1]), 'v_v': ([0, 1, 2], [0, 1])}
    for k, (rowptr, colind) in sd.items():                     # the diffusion sparsity rans2p.h writes out
        assert list(d['sdInfo_%s_rowptr' % k]) == rowptr and list(d['sdInfo_%s_colind' % k]) == colind, k
    l2g = d['mesh_l2g']
    assert (d['p_l2g'] == l2g).all() and (d['vel_l2g'] == l2g).all()


def element_terms(d):
    l2g, nodes = d['mesh_l2g'], d['mesh_dof']
    nE, nD = l2g.shape
    nS = nD - 1
    ue = np.stack([d['p_dof'][l2g], d['u_dof'][l2g], d['v_dof'][l2g]], axis=1)
    fields = np.concatenate([d['vf'][..., None], d['q_porosity'][..., None], d['q_dV_last'][..., None],
                             d['q_mom_u_acc_beta_bdf'][..., None], d['q_mom_v_acc_beta_bdf'][..., None],
                             d['q_numDiff_u_last'][..., None], d['q_numDiff_v_last'][..., None],
                             d['q_velocity_sge'][..., :nS]], axis=-1)
    params = np.concatenate([np.atleast_1d(np.asarray(d[k], dtype=float))[:3] if k == 'g' else [float(d[k])] for k in PARAMS])
    return adrh.rans2p_element(nS, nodes[l2g], np.ascontiguousarray(ue), np.ascontiguousarray(d['x_ref'][:, :nS]),
                               d['dV_ref'], np.ascontiguousarray(fields), d['elementDiameter'], params)


def assemble(d, R, J):
    l2g = d['mesh_l2g']
    nE, nD = l2g.shape
    comps = ('p', 'u', 'v')
    dof = np.stack([int(d['offset_' + c]) + int(d['stride_' + c]) * l2g for c in comps], axis=1)   # (nE, nc, nD)
    n = len(d['out_globalResidual'])
    r = np.zeros(n)
    np.add.at(r, dof.ravel(), R.ravel())
    nc = len(comps)
    rows = np.broadcast_to(dof[:, :, :, None, None], (nE, nc, nD, nc, nD)).ravel()
    cols = np.broadcast_to(dof[:, None, None, :, :], (nE, nc, nD, nc, nD)).ravel()
    return r, sp.csr_matrix((J.ravel(), (rows, cols)), shape=(n, n))


def rel(a, b):
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-300)


def main(argv):
    path = argv[0] if argv else os.path.join(HERE, 'rans2p_damBreak_he0.05.npz')
    d = dict(np.load(path))
    check_configuration(d)
    R, J, qv, qnd, qcfl = element_terms(d)
    r, Jg = assemble(d, R, J)
    n = len(r)
    rowptr, colind = d['csrRowIndeces_DofLoops'], d['csrColumnOffsets_DofLoops']
    Jp = sp.csr_matrix((d['out_elem_globalJacobian'], colind, rowptr), shape=(n, n))
    Jfull = sp.csr_matrix((d['out_globalJacobian'], colind, rowptr), shape=(n, n))
    re = d['out_elem_globalResidual']
    checks = [
        ('residual, element integrals', rel(r, re)),
        ('jacobian, element integrals', abs(Jg - Jp).max() / abs(Jp).max()),
        ('jacobian entries inside the sparsity', abs(Jg - Jg.multiply(Jp != 0)).max() / abs(Jp).max()),
        ('q_velocity', rel(qv, d['out_q_velocity'][..., :2])),
        ('q_numDiff_u (next lagged viscosity)', rel(qnd, d['out_q_numDiff_u'])),
        ('q_cfl', rel(qcfl, d['out_q_cfl'])),
    ]
    print('RANS2P engine vs proteus (RANS2P2D), damBreakLevelSet he=%g, t=%.4f, %d nodes, %d elements, %d unknowns; '
          '|R| %.2e (element integrals %.2e)'
          % (float(d['he']), float(d['t']), d['mesh_dof'].shape[0], d['mesh_l2g'].shape[0], n,
             np.abs(d['out_globalResidual']).max(), np.abs(re).max()))
    ok = True
    for name, err in checks:
        print('  %-42s %.2e  %s' % (name, err, 'ok' if err <= TOL else 'FAIL'))
        ok &= err <= TOL
    print('  not reproduced yet, the exterior faces: %.2e of the residual, %.2e of the Jacobian'
          % (np.abs(d['out_globalResidual'] - re).max() / np.abs(d['out_globalResidual']).max(),
             abs(Jfull - Jp).max() / abs(Jfull).max()))
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
