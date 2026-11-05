"""Record what proteus's VOF (or NCLS) kernel is handed on the damBreak case, for vof_test.py (ncls_test.py).

Runs test/TwoPhaseFlow/damBreakLevelSet.py (RANS2P, VOF, NCLS, RDLS, MCorr; useDefaultModels(0, 0)) in one process
and keeps, for the first VOF Newton solve at or after time T, the arrays its calculateResidual and calculateJacobian
were handed (through an ArgumentsDict that remembers what was put in it), the residual, the CSR Jacobian, the
shock-capturing viscosity and CFL the kernel wrote, and the solution after the solve with the residual there. VOF is
linear, so its Newton solve is one linear solve; the fixture pins one whole VOF step. ``model=ncls`` does the same for
NCLS (also linear), with NCLS's arguments and outputs (grad u at the points, the boundary trace and its normalized gradient,
the interface locator).

Needs a built proteus (PYTHONPATH at a worktree with its extensions) and runs from a scratch directory (the case
writes its archive there):

    python vof_capture.py OUT.npz [he=0.05] [T=0.05] [model=vof|ncls]
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


def main(argv):
    if not argv:
        raise SystemExit(__doc__)
    out = os.path.abspath(argv[0])
    opts = dict(a.split('=', 1) for a in argv[1:])
    he, T = float(opts.get('he', 0.05)), float(opts.get('T', 0.05))
    model = opts.get('model', 'vof')
    keep, outputs_r = (KEEP_NCLS, OUTPUTS_R_NCLS) if model == 'ncls' else (KEEP, OUTPUTS_R)
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
    vof = [m for m in ns.modelList if m.name == model][0].levelModelList[-1]
    state = {'phase': 'wait'}
    fixture = {}

    def copy(d, keys, prefix=''):
        for k in keys:
            v = d[k]
            fixture[prefix + k] = np.array(v, copy=True)

    #NCLS binds self.calculateResidual = self.ncls.calculateResidual on every call, so wrap its kernel object instead
    kernel = getattr(vof, 'ncls', None) if model == 'ncls' else None
    if kernel is not None:
        residual, jacobian = kernel.calculateResidual, kernel.calculateJacobian
    else:
        residual, jacobian = vof.calculateResidual, vof.calculateJacobian

    def calculateResidual(argsDict):
        seen = argsDict.seen
        #NS_base's initialization calls the residual at a huge t and dt; a step of the run has T <= t <= the final time
        ti = vof.timeIntegration
        if state['phase'] == 'wait' and T <= ti.t <= T + 0.05 and ti.dt < 1.0:
            copy(seen, keep)
            fixture['t'] = np.array(vof.timeIntegration.t)
            state['phase'] = 'residual'
            residual(argsDict)
            copy(seen, outputs_r, 'out_')
            return
        if state['phase'] == 'jacobian':
            fixture['u_next'] = np.array(seen['u_dof'], copy=True)
            residual(argsDict)
            fixture['out_residual_next'] = np.array(seen['globalResidual'], copy=True)
            state['phase'] = 'done'
            return
        residual(argsDict)

    def calculateJacobian(argsDict):
        jacobian(argsDict)
        if state['phase'] == 'residual':
            fixture['out_globalJacobian'] = np.array(argsDict.seen['globalJacobian'], copy=True)
            fixture['jac_u_dof'] = np.array(argsDict.seen['u_dof'], copy=True)
            state['phase'] = 'jacobian'

    if kernel is not None:
        class Kernel(object):
            def __getattr__(self, name):
                return getattr(kernel, name)
        proxy = Kernel()
        proxy.calculateResidual, proxy.calculateJacobian = calculateResidual, calculateJacobian
        vof.ncls = proxy
    else:
        vof.calculateResidual, vof.calculateJacobian = calculateResidual, calculateJacobian
    ns.calculateSolution(so.name)
    if state['phase'] != 'done':
        raise SystemExit('no ' + model + ' solve captured at t >= %g (phase %s)' % (T, state['phase']))
    fixture['he'] = np.array(he)
    fixture['model'] = np.array(model)
    np.savez_compressed(out, **fixture)
    print('wrote %s: t=%.4f dt=%.3e nodes=%d elements=%d' % (out, float(fixture['t']), float(fixture['dt']),
                                                           fixture['mesh_dof'].shape[0], fixture['mesh_l2g'].shape[0]))


if __name__ == '__main__':
    main(sys.argv[1:])
