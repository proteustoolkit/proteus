"""Record what proteus's VOF (or NCLS) kernel is handed on the damBreak case, for vof_test.py (ncls_test.py).

Runs test/TwoPhaseFlow/damBreakLevelSet.py (RANS2P, VOF, NCLS, RDLS, MCorr; useDefaultModels(0, 0)) in one process
and keeps, for the first VOF Newton solve at or after time T, the arrays its calculateResidual and calculateJacobian
were handed (through an ArgumentsDict that remembers what was put in it), the residual, the CSR Jacobian, the
shock-capturing viscosity and CFL the kernel wrote, and the solution after the solve with the residual there. VOF is
linear, so its Newton solve is one linear solve; the fixture pins one whole VOF step. ``model=ncls`` does the same for
NCLS (also linear), with NCLS's arguments and outputs (grad u at the points, the boundary trace and its normalized gradient,
the interface locator). ``model=rdls`` records the first RDLS Newton iteration at or after T (RDLS is nonlinear: the
fixture's u_next is the next Newton iterate and out_residual_next proteus's residual there), with the Jacobian's CSR
structure taken from the matrix (RDLS's kernel gets element-local offsets only). ``model=mcorr`` records the first MCorr
Newton iteration the same way (the kernel's residual and Jacobian before MCorr.py masks inactive rows, and the active-DOF
marks the kernel wrote). ``model=rans2p`` records the first RANS2P (RANS2P2D) Newton iteration at or after T: every
array and scalar its kernel was handed, the residual, the CSR Jacobian and the point outputs (q_velocity, q_numDiff_u, q_cfl),
(rans2p_test.py's inputs; KEEP_ALL in the script keeps every argument), and the same residual and Jacobian with the
exterior faces switched off (the kernel called again with
nExteriorElementBoundaries_global = 0 into fresh arrays: out_elem_globalResidual, out_elem_globalJacobian), so that the
element integrals can be checked on their own. ``solid=CX,CY,R`` (model=mcorr) puts a circular solid of radius R at (CX, CY) on MCorr's cut-cell
path for the captured solve: just before that solve's first residual, RANS2P's coefficients (which MCorr reads) get
useExact = True and phi_s, q_phi_solid = the circle's signed distance (positive in the fluid), so MCorr marks the
elements, adds the ghost penalty on cut faces and masks the inactive rows (MCorr.py). Only MCorr sees the solid: the capture
stops after that Newton iteration.

Needs a built proteus (PYTHONPATH at a worktree with its extensions) and runs from a scratch directory (the case
writes its archive there):

    python vof_capture.py OUT.npz [he=0.05] [T=0.05] [model=vof|ncls|rdls|mcorr|rans2p] [solid=CX,CY,R]
"""
import os
import sys

import numpy as np

KEEP = ('dt', 'sc_drop_time', 'MOVING_DOMAIN', 'mesh_dof', 'mesh_l2g', 'x_ref', 'dV_ref', 'u_trial_trace_ref', 'dS_ref',
        'useMetrics', 'alphaBDF', 'lag_shockCapturing', 'shockCapturingDiffusion', 'sc_uref', 'sc_alpha', 'q_porosity',
        'u_l2g', 'r_l2g', 'elementDiameter', 'u_dof', 'velocity', 'q_m_betaBDF', 'q_dV_last', 'q_numDiff_u_last',
        'offset_u', 'stride_u', 'exteriorElementBoundariesArray', 'elementBoundaryElementsArray',
        'elementBoundaryLocalElementBoundariesArray', 'ebqe_velocity_ext', 'ebqe_porosity_ext', 'isDOFBoundary_u',
        'ebqe_bc_u_ext', 'isFluxBoundary_u', 'ebqe_bc_flux_u_ext', 'STABILIZATION_TYPE', 'LUMPED_MASS_MATRIX',
        'csrRowIndeces_DofLoops', 'csrColumnOffsets_DofLoops', 'phi_solid_nodes', 'ebqe_phi_s', 'phi_solid')
OUTPUTS_R = ('globalResidual', 'q_numDiff_u', 'cfl', 'ebqe_flux', 'ebqe_u')
KEEP_NCLS = ('dt', 'sc_drop_time', 'MOVING_DOMAIN', 'mesh_dof', 'mesh_l2g', 'dV_ref', 'u_trial_ref', 'u_trial_trace_ref',
             'dS_ref', 'useMetrics', 'alphaBDF', 'lag_shockCapturing', 'shockCapturingDiffusion', 'sc_uref', 'sc_alpha',
             'u_l2g', 'r_l2g', 'elementDiameter', 'u_dof', 'velocity', 'q_m_betaBDF', 'q_dV_last', 'q_numDiff_u_last',
             'offset_u', 'stride_u', 'exteriorElementBoundariesArray', 'elementBoundaryElementsArray',
             'elementBoundaryLocalElementBoundariesArray', 'ebqe_velocity_ext', 'isDOFBoundary_u', 'ebqe_rd_u_ext',
             'ebqe_bc_u_ext', 'PURE_BDF', 'STABILIZATION_TYPE', 'csrRowIndeces_DofLoops', 'csrColumnOffsets_DofLoops')
OUTPUTS_R_NCLS = ('globalResidual', 'q_numDiff_u', 'cfl', 'q_n', 'ebqe_u', 'ebqe_grad_u', 'interface_locator')
KEEP_RDLS = ('mesh_dof', 'mesh_l2g', 'dV_ref', 'u_trial_ref', 'u_trial_trace_ref', 'dS_ref', 'useMetrics', 'alphaBDF',
             'epsFact_redist', 'backgroundDiffusionFactor', 'weakDirichletFactor', 'freezeLevelSet', 'useTimeIntegration',
             'lag_shockCapturing', 'lag_subgridError', 'shockCapturingDiffusion', 'u_l2g', 'elementDiameter',
             'nodeDiametersArray', 'u_dof', 'phi_dof', 'phi_ls', 'u_weak_internal_bc_dofs', 'q_numDiff_u_last', 'offset_u',
             'stride_u', 'exteriorElementBoundariesArray', 'elementBoundaryElementsArray',
             'elementBoundaryLocalElementBoundariesArray', 'ELLIPTIC_REDISTANCING')
OUTPUTS_R_RDLS = ('globalResidual', 'q_numDiff_u', 'q_cfl', 'q_dH', 'q_m', 'q_u', 'ebqe_u', 'ebqe_n')
KEEP_MCORR = ('mesh_dof', 'mesh_l2g', 'x_ref', 'dV_ref', 'u_trial_ref', 'useMetrics', 'epsFactHeaviside', 'epsFactDirac',
              'epsFactDiffusion', 'u_l2g', 'r_l2g', 'elementDiameter', 'nodeDiametersArray', 'u_dof', 'phi_dof', 'q_phi', 'q_H',
              'q_porosity', 'offset_u', 'stride_u', 'phi_solid_nodes', 'phi_solid', 'useExact_s')
OUTPUTS_R_MCORR = ('globalResidual', 'q_u', 'q_r', 'q_n', 'isActiveR', 'isActiveDOF', 'isActiveElement')
#RANS2P: what rans2p_test.py reads (the element integrals and the inputs that must be zero for them); KEEP_ALL keeps
#every argument the kernel was handed
KEEP_RANS2P = ('mesh_dof', 'mesh_l2g', 'x_ref', 'dV_ref', 'p_l2g', 'vel_l2g', 'elementDiameter', 'p_dof', 'u_dof', 'v_dof',
               'rho_0', 'nu_0', 'rho_1', 'nu_1', 'g', 'alphaBDF', 'Ct_sge', 'Cd_sge', 'C_dc', 'useMetrics', 'hFactor', 'useVF',
               'MOMENTUM_SGE', 'PRESSURE_SGE', 'VELOCITY_SGE', 'vf', 'q_porosity', 'q_dV_last', 'q_mom_u_acc_beta_bdf',
               'q_mom_v_acc_beta_bdf', 'q_numDiff_u_last', 'q_numDiff_v_last', 'q_velocity_sge', 'q_dragAlpha', 'q_dragBeta',
               'q_mass_source', 'forcex', 'forcey', 'q_eddy_viscosity_last', 'mesh_velocity_dof', 'NONCONSERVATIVE_FORM',
               'MOVING_DOMAIN', 'useRBLES', 'PRESSURE_PROJECTION_STABILIZATION', 'nParticles', 'useExact',
               'turbulenceClosureModel', 'phi_solid', 'sdInfo_u_u_rowptr', 'sdInfo_u_u_colind', 'sdInfo_u_v_rowptr',
               'sdInfo_u_v_colind', 'sdInfo_v_u_rowptr', 'sdInfo_v_u_colind', 'sdInfo_v_v_rowptr', 'sdInfo_v_v_colind',
               'offset_p', 'offset_u', 'offset_v', 'stride_p', 'stride_u', 'stride_v',
               # the exterior faces (rans2p.h navierStokesExteriorFace): geometry, boundary conditions, the face fields
               'nExteriorElementBoundaries_global', 'exteriorElementBoundariesArray', 'elementBoundaryElementsArray',
               'elementBoundaryLocalElementBoundariesArray', 'boundaryFlags', 'nElementBoundaries_owned', 'dS_ref',
               'vel_trial_trace_ref', 'C_b', 'eb_adjoint_sigma', 'ebqe_penalty_ext', 'ebqe_vf_ext', 'bc_ebqe_vf_ext',
               'ebqe_porosity_ext', 'ebqe_phi_s', 'ebqe_eddy_viscosity_last', 'isDOFBoundary_p', 'isDOFBoundary_u',
               'isDOFBoundary_v', 'isAdvectiveFluxBoundary_p', 'isAdvectiveFluxBoundary_u', 'isAdvectiveFluxBoundary_v',
               'isDiffusiveFluxBoundary_u', 'isDiffusiveFluxBoundary_v', 'ebqe_bc_p_ext', 'ebqe_bc_u_ext', 'ebqe_bc_v_ext',
               'ebqe_bc_flux_mass_ext', 'ebqe_bc_flux_mom_u_adv_ext', 'ebqe_bc_flux_mom_v_adv_ext',
               'ebqe_bc_flux_u_diff_ext', 'ebqe_bc_flux_v_diff_ext', 'particle_epsFact',
               'use_ball_as_particle', 'normalize_pressure')
OUTPUTS_R_RANS2P = ('globalResidual', 'q_velocity', 'q_numDiff_u', 'q_cfl', 'ebqe_velocity', 'flux', 'isActiveElement')
KEEP_ALL = object()
#with a solid (MCorr's cut-cell path): the interior-face data its ghost penalty uses
KEEP_MCORR_SOLID = KEEP_MCORR + ('ghost_penalty_constant', 'elementBoundaryElementsArray', 'elementBoundariesArray',
                                 'elementBoundaryLocalElementBoundariesArray', 'elementBoundaryDiameter')


class Stop(Exception):
    pass


def main(argv):
    if not argv:
        raise SystemExit(__doc__)
    out = os.path.abspath(argv[0])
    opts = dict(a.split('=', 1) for a in argv[1:])
    he, T = float(opts.get('he', 0.05)), float(opts.get('T', 0.05))
    model = opts.get('model', 'vof')
    solid = [float(v) for v in opts['solid'].split(',')] if 'solid' in opts else None
    keep, outputs_r = {'ncls': (KEEP_NCLS, OUTPUTS_R_NCLS), 'rdls': (KEEP_RDLS, OUTPUTS_R_RDLS),
                       'mcorr': (KEEP_MCORR, OUTPUTS_R_MCORR), 'rans2p': (KEEP_RANS2P, OUTPUTS_R_RANS2P)}.get(model, (KEEP, OUTPUTS_R))
    if solid is not None:
        assert model == 'mcorr', 'solid= is for model=mcorr'
        keep = KEEP_MCORR_SOLID
    here = os.path.dirname(os.path.abspath(__file__))
    case = os.path.join(here, '..', '..', 'test', 'TwoPhaseFlow', 'damBreakLevelSet.py')

    from proteus import Comm
    Comm.init()
    from proteus.mprans import cArgumentsDict

    base = cArgumentsDict.ArgumentsDict

    class Recording(base):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            object.__setattr__(self, 'seen', {})

        def __setitem__(self, key, value):
            self.seen[key] = value
            super().__setitem__(key, value)
    cArgumentsDict.ArgumentsDict = Recording

    from proteus import Context, NumericalSolution, Profiling
    from proteus.cli.run import _load
    from proteus.iproteus import opts as runopts

    class Args(object):
        inputs = [case]
        petsc_file = None
        pumi = False
    Context.contextOptionsString = 'final_time=%r dt_output=%r he=%r genMesh=True' % (T + 0.05, T + 0.05, he)
    Profiling.logLevel = 1
    Profiling.verbose = False
    so, pList, nList, sList, two_phase = _load(Args(), Profiling.logEvent)
    flags = [{"simulationName": p.name, "simulationNameProc": p.name + '0', "dataFile": p.name + '0.dat',
              "components": list(range(p.coefficients.nc))} for p in pList]
    ns = NumericalSolution.NS_base(so, pList, nList, sList, runopts, flags, two_phase)
    mlm = [m for m in ns.modelList if m.name == model][0]
    vof = mlm.levelModelList[-1]
    jac_matrix = mlm.jacobianList[-1]
    state = {'phase': 'wait'}
    fixture = {}

    def copy(d, keys, prefix=''):
        if keys is KEEP_ALL:
            keys = [k for k, v in d.items() if v is not None and np.asarray(v).dtype != object]
        for k in keys:
            v = d[k]
            fixture[prefix + k] = np.array(v, copy=True)

    #NCLS and RDLS bind self.calculateResidual = self.<kernel>.calculateResidual on every call, so wrap the kernel object
    #(MCorr calls self.mcorr.calculateResidual directly, which the same wrapper catches)
    kernel = getattr(vof, model, None) if model in ('ncls', 'rdls', 'mcorr', 'rans2p') else None
    if kernel is not None:
        residual, jacobian = kernel.calculateResidual, kernel.calculateJacobian
    else:
        residual, jacobian = vof.calculateResidual, vof.calculateJacobian

    def calculateResidual(argsDict, *extra):
        seen = argsDict.seen
        #NS_base's initialization calls the residual at a huge t and dt; a step of the run has T <= t <= the final time
        ti = vof.timeIntegration
        if state['phase'] == 'wait' and T <= ti.t <= T + 0.05 and ti.dt < 1.0:
            copy(seen, keep)
            fixture['t'] = np.array(vof.timeIntegration.t)
            if extra:  # MCorr's and RDLS's kernels take useExact as a second argument
                fixture['useExact'] = np.array(int(extra[0]))
            state['phase'] = 'residual'
            residual(argsDict, *extra)
            copy(seen, outputs_r, 'out_')
            if model == 'rans2p':
                fixture['out_elem_globalResidual'] = element_only(argsDict, 'globalResidual', residual)
            return
        if state['phase'] == 'jacobian':
            fixture['u_next'] = np.array(seen['u_dof'], copy=True)
            if model == 'rans2p':  # all three components of the next Newton iterate
                fixture['p_next'] = np.array(seen['p_dof'], copy=True)
                fixture['v_next'] = np.array(seen['v_dof'], copy=True)
            residual(argsDict, *extra)
            fixture['out_residual_next'] = np.array(seen['globalResidual'], copy=True)
            if model == 'mcorr':  # MCorr starts from u = 0, so its point outputs say something only at the next iterate
                copy(seen, ('q_u', 'q_r', 'q_n'), 'out_next_')
            state['phase'] = 'done'
            return
        if state['phase'] == 'done' and solid is not None:
            raise Stop()
        residual(argsDict, *extra)

    def element_only(argsDict, out, call):
        #the kernel again, exterior faces off, into a fresh output array; the arguments are restored afterwards
        seen = argsDict.seen
        saved, nExt = seen[out], seen['nExteriorElementBoundaries_global']
        fresh = np.zeros_like(saved)
        argsDict[out] = fresh
        argsDict['nExteriorElementBoundaries_global'] = 0
        call(argsDict)
        argsDict[out] = saved
        argsDict['nExteriorElementBoundaries_global'] = nExt
        return fresh.copy()

    def calculateJacobian(argsDict, *extra):
        jacobian(argsDict, *extra)
        if state['phase'] == 'residual':
            fixture['out_globalJacobian'] = np.array(argsDict.seen['globalJacobian'], copy=True)
            if model == 'rans2p':
                fixture['out_elem_globalJacobian'] = element_only(argsDict, 'globalJacobian', jacobian)
            if model in ('rdls', 'mcorr', 'rans2p'):
                #RDLS's and MCorr's kernels get element-local CSR offsets only; keep the matrix's own CSR structure, under NCLS's names
                rowptr, colind = jac_matrix.getCSRrepresentation()[:2]
                fixture['csrRowIndeces_DofLoops'] = np.array(rowptr, copy=True)
                fixture['csrColumnOffsets_DofLoops'] = np.array(colind, copy=True)
            fixture['jac_u_dof'] = np.array(argsDict.seen['u_dof'], copy=True)
            state['phase'] = 'jacobian'

    if solid is not None:
        #put the solid into RANS2P's coefficients just before the captured MCorr solve (MCorr.py reads them per call)
        getResidual = vof.getResidual

        def sdf(x):
            return np.hypot(x[..., 0] - solid[0], x[..., 1] - solid[1]) - solid[2]

        def getResidualWithSolid(u, r):
            ti, flow = vof.timeIntegration, vof.coefficients.flowCoefficients
            if state['phase'] == 'wait' and T <= ti.t <= T + 0.05 and ti.dt < 1.0 and not flow.useExact:
                flow.useExact = True
                flow.phi_s[:] = sdf(vof.mesh.nodeArray)
                flow.q_phi_solid[:] = sdf(flow.model.q['x'])  # RANS2P's points: the same element quadrature
            return getResidual(u, r)
        vof.getResidual = getResidualWithSolid

    if kernel is not None:
        class Kernel(object):
            def __getattr__(self, name):
                return getattr(kernel, name)
        proxy = Kernel()
        proxy.calculateResidual, proxy.calculateJacobian = calculateResidual, calculateJacobian
        setattr(vof, model, proxy)
    else:
        vof.calculateResidual, vof.calculateJacobian = calculateResidual, calculateJacobian
    try:
        ns.calculateSolution(so.name)
    except Stop:
        pass
    if state['phase'] != 'done':
        raise SystemExit('no ' + model + ' solve captured at t >= %g (phase %s)' % (T, state['phase']))
    fixture['he'] = np.array(he)
    fixture['model'] = np.array(model)
    if solid is not None:
        fixture['solid'] = np.array(solid)
    np.savez_compressed(out, **fixture)
    print('wrote %s: t=%.4f dt=%.3e nodes=%d elements=%d' % (out, float(fixture['t']), float(fixture.get('dt', np.nan)),
                                                           fixture['mesh_dof'].shape[0], fixture['mesh_l2g'].shape[0]))


if __name__ == '__main__':
    main(sys.argv[1:])
