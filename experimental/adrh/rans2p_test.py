"""The engine's RANS2P (rans2p.h) against proteus's RANS2P2D kernel on the damBreak case.

The fixture (rans2p_damBreak_he0.05.npz, written by ``vof_capture.py OUT model=rans2p``) holds RANS2P's first Newton iteration
at t >= 0.05 as proteus ran it: the arguments its kernel was handed, its residual, CSR Jacobian and point and boundary
outputs, the same residual and Jacobian with the exterior faces switched off (the kernel called again with
nExteriorElementBoundaries_global = 0), and the next Newton iterate with proteus's residual there (the deck solves with
SuperLU, so the iterate is u - J^-1 r). Checks:

* the configuration is the one rans2p.h implements (what it leaves out is zero in the deck);
* the element integrals alone (residual and Jacobian, pressure and two velocity components, interlaced as proteus numbers
  them) and the point outputs (velocity, the next shock-capturing viscosity, the CFL);
* elements plus exterior faces (the weak boundary conditions: upwinded inflow, the pressure Dirichlet term, the Nitsche
  penalty and adjoint terms of the stress) against proteus's full residual and Jacobian, and the boundary outputs (the mass
  flux and ebqe_velocity);
* one Newton step with the engine's residual and Jacobian gives proteus's next iterate, and the engine's residual there
  is proteus's.

Usage: rans2p_test.py [fixture.npz]    (ADRH_MODULE picks the module; only adrh_proto has RANS2P so far)
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
    # the exterior faces: every face flagged (RANS2P2D.h skips flag < 1), every element active, no solid (H_s = 1),
    # no eddy viscosity, no pressure normalization
    ebN = d['exteriorElementBoundariesArray']
    assert (d['boundaryFlags'][ebN] >= 1).all() and (d['out_isActiveElement'] == 1).all()
    assert d['ebqe_phi_s'].min() > 1.0 and not np.any(d['ebqe_eddy_viscosity_last'])
    assert float(d['normalize_pressure']) == 0.0
    for c in 'puv':                               # each flag 0 or 1 (the engine compares with 1.0)
        assert set(np.unique(d['isDOFBoundary_' + c])) <= {0, 1}


def element_terms(d, dofs=None):
    l2g, nodes = d['mesh_l2g'], d['mesh_dof']
    nE, nD = l2g.shape
    nS = nD - 1
    dofs = (d['p_dof'], d['u_dof'], d['v_dof']) if dofs is None else dofs
    ue = np.stack([x[l2g] for x in dofs], axis=1)
    fields = np.concatenate([d['vf'][..., None], d['q_porosity'][..., None], d['q_dV_last'][..., None],
                             d['q_mom_u_acc_beta_bdf'][..., None], d['q_mom_v_acc_beta_bdf'][..., None],
                             d['q_numDiff_u_last'][..., None], d['q_numDiff_v_last'][..., None],
                             d['q_velocity_sge'][..., :nS]], axis=-1)
    return adrh.rans2p_element(nS, nodes[l2g], np.ascontiguousarray(ue), np.ascontiguousarray(d['x_ref'][:, :nS]),
                               d['dV_ref'], np.ascontiguousarray(fields), d['elementDiameter'], params_of(d))


def params_of(d):
    return np.concatenate([np.atleast_1d(np.asarray(d[k], dtype=float))[:3] if k == 'g' else [float(d[k])] for k in PARAMS])


def face_terms(d, dofs):
    """The engine's exterior-face residual and Jacobian at dofs = (p, u, v), per face of the face's element."""
    l2g, nodes = d['mesh_l2g'], d['mesh_dof']
    nS = l2g.shape[1] - 1
    ebN = d['exteriorElementBoundariesArray']
    eN = d['elementBoundaryElementsArray'][ebN, 0]
    k = d['elementBoundaryLocalElementBoundariesArray'][ebN, 0]
    ue = np.stack([x[l2g[eN]] for x in dofs], axis=1)
    xi = np.ascontiguousarray(d['vel_trial_trace_ref'][:, :, 1:])   # P1: the trace basis values are barycentric coordinates
    nF, nQf = d['ebqe_vf_ext'].shape
    one = np.ones((nF, nQf))
    cols = [d['ebqe_vf_ext'], d['ebqe_porosity_ext'], d['ebqe_penalty_ext'], d['isDOFBoundary_p'],
            d['isAdvectiveFluxBoundary_p'], d['ebqe_bc_p_ext'], d['ebqe_bc_flux_mass_ext'], float(d['C_b']) * one,
            float(d['eb_adjoint_sigma']) * one]
    comps = 'uv'
    cols += [d['isDOFBoundary_' + c] for c in comps] + [d['isAdvectiveFluxBoundary_' + c] for c in comps]
    cols += [d['isDiffusiveFluxBoundary_' + c] for c in comps] + [d['ebqe_bc_%s_ext' % c] for c in comps]
    cols += [d['ebqe_bc_flux_mom_%s_adv_ext' % c] for c in comps] + [d['ebqe_bc_flux_%s_diff_ext' % c] for c in comps]
    ff = np.ascontiguousarray(np.stack([np.asarray(c, dtype=float) for c in cols], axis=-1))
    Rf, Jf, flux, vel = adrh.rans2p_faces(nS, nodes[l2g[eN]], np.ascontiguousarray(ue), k, xi, d['dS_ref'], ff, params_of(d))
    return eN, Rf, Jf, flux, vel


def assemble(d, R, J, eN=None):
    l2g = d['mesh_l2g'] if eN is None else d['mesh_l2g'][eN]
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


def full(d, dofs):
    """Elements plus exterior faces at dofs: the global residual and Jacobian, and the faces' boundary outputs."""
    R, J = element_terms(d, dofs)[:2]
    r, Jg = assemble(d, R, J)
    eN, Rf, Jf, flux, vel = face_terms(d, dofs)
    rf, Jfg = assemble(d, Rf, Jf, eN)
    return r + rf, (Jg + Jfg).tocsr(), flux, vel


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
    re, rfull = d['out_elem_globalResidual'], d['out_globalResidual']
    dofs = (d['p_dof'], d['u_dof'], d['v_dof'])
    rF, JF, flux, vel = full(d, dofs)
    ebN = d['exteriorElementBoundariesArray']
    nS = d['mesh_l2g'].shape[1] - 1
    # one Newton step (no line search) with the engine's residual and Jacobian; the unknowns are interlaced (p, u, v)
    comps = ('p', 'u', 'v')
    idx = [int(d['offset_' + c]) + int(d['stride_' + c]) * np.arange(len(d[c + '_dof'])) for c in comps]
    x = np.zeros(n)
    for i, c in zip(idx, comps):
        x[i] = d[c + '_dof']
    x_new = x - spla.spsolve(JF.tocsc(), rF)
    x_next = np.zeros(n)
    for i, c in zip(idx, comps):
        x_next[i] = d[c + '_next']
    r_next = full(d, tuple(x_next[i] for i in idx))[0]
    checks = [
        ('residual, element integrals', rel(r, re)),
        ('jacobian, element integrals', abs(Jg - Jp).max() / abs(Jp).max()),
        ('jacobian entries inside the sparsity', abs(Jg - Jg.multiply(Jp != 0)).max() / abs(Jp).max()),
        ('q_velocity', rel(qv, d['out_q_velocity'][..., :2])),
        ('q_numDiff_u (next lagged viscosity)', rel(qnd, d['out_q_numDiff_u'])),
        ('q_cfl', rel(qcfl, d['out_q_cfl'])),
        ('residual, elements + exterior faces', rel(rF, rfull)),
        ('residual, the faces\' share', rel(rF - r, rfull - re)),
        ('jacobian, elements + exterior faces', abs(JF - Jfull).max() / abs(Jfull).max()),
        ('jacobian, the faces\' share', abs((JF - Jg) - (Jfull - Jp)).max() / abs(Jfull - Jp).max()),
        ('jacobian entries inside the sparsity', abs(JF - JF.multiply(Jfull != 0)).max() / abs(Jfull).max()),
        ('boundary mass flux', rel(flux, d['out_flux'].reshape(-1, flux.shape[1])[ebN])),
        ('ebqe_velocity', rel(vel, d['out_ebqe_velocity'][..., :nS])),
        ('next Newton iterate', rel(x_new, x_next)),
        ('residual at the next iterate', rel(r_next, d['out_residual_next'])),
    ]
    print('RANS2P engine vs proteus (RANS2P2D), damBreakLevelSet he=%g, t=%.4f, %d nodes, %d elements, %d exterior faces, '
          '%d unknowns; |R| %.2e (element integrals %.2e)'
          % (float(d['he']), float(d['t']), d['mesh_dof'].shape[0], d['mesh_l2g'].shape[0], len(ebN), n,
             np.abs(rfull).max(), np.abs(re).max()))
    ok = True
    for name, err in checks:
        print('  %-42s %.2e  %s' % (name, err, 'ok' if err <= TOL else 'FAIL'))
        ok &= err <= TOL
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
