"""Per-run conservation, bounds and cost measures for the coupled VDFT decks.

Usage, at the end of a transport ``*_p.py`` (TADR STABILIZATION_TYPE 5 with the flow
model's edge fluxes, i.e. ``RICHARDS_EDGE_VELOCITY=1``)::

    try:
        import vdft_metrics
        vdft_metrics.install(coefficients)
    except ImportError:
        pass

It wraps ``coefficients.attachModels`` (to count Newton iterations of both models) and
``coefficients.postStep`` (to accumulate the measures once per accepted step), and rewrites
``vdft_metrics.json`` in the run directory after every step, so a killed run still leaves
its numbers.  Nothing in the solution changes.

Measures (sums over owned DOFs, then MPI reductions):

* ``mb_error``: the solute mass-balance error the manuscript reports, i.e. TADR's
  ``continuityStats['total_frac']``: the time-integrated c-weighted flow/transport continuity
  defect plus the stored-history jump, over the current solute mass.  With the flow model's
  edge fluxes the boundary exchange B_i is defined as the residual of the carried storage, so
  this measure is ~round-off by construction; it does NOT see the two flow errors below.
* ``flow_residual``: the flow model's own fluid balance,
  ``sum_n [ sum_i ML_i (m_i^{n+1} - m_i^n) - dt sum_i b_i ]``, where ``m^{n+1}`` is the accepted
  (limited) mass, ``m^n`` Richards' old-time mass and ``b_i`` Richards' physical boundary flux
  (``boundaryFluxDOF`` plus the row sums of ``edgeFluxLow``, which carry the weak-Dirichlet
  boundary block; positive into the domain, from the last residual evaluation).  Interior
  exchanges and the limited corrections cancel in the global sum, so this is dt times the sum
  of the low-order Newton residual: the fluid the step creates or destroys.
* ``flow_recovery``: the fluid lost or gained when the pressure head is recovered from the
  corrected mass, ``sum_n sum_i ML_i (m_i^{n+1,old} - m_i^{n,accepted})``: the next step's
  old-time mass (re-evaluated from psi^{n+1} with the same lagged density) minus this step's
  accepted mass.  Non-zero at nodes whose predictor is saturated (they keep psi^L).
* ``flow_balance_error``: ``|flow_residual + flow_recovery|`` over the final fluid mass
  ``sum_i ML_i m_i``; also reported over the time-integrated |boundary flux|.
* ``drift``: ``max_n |sum_i ML_i (M_i - m_i^{n+1})| / sum_i ML_i m_i^{n+1}``, carried transport
  storage against the flow model's accepted mass (TADR's [TADR-EDGE] sum).  Equal to the
  accumulated ``flow_recovery`` up to the last step.
* ``c_min``, ``c_max``: over all owned nodes and accepted steps.
* ``steps``, ``newton_flow``, ``newton_transport`` (Jacobian evaluations, rejected steps
  included), ``wall_s``, ``ranks``.
"""
import json
import os
import time

import numpy as np

_OUT = os.environ.get("VDFT_METRICS_OUT", "vdft_metrics.json")


def _comm():
    from proteus import Comm
    return Comm.get()


def _gsum(x):
    c = _comm()
    if c.size() > 1:
        from mpi4py import MPI
        return c.comm.tompi4py().allreduce(float(x), op=MPI.SUM)
    return float(x)


def _gmin(x):
    c = _comm()
    if c.size() > 1:
        from mpi4py import MPI
        return c.comm.tompi4py().allreduce(float(x), op=MPI.MIN)
    return float(x)


def _gmax(x):
    return -_gmin(-x)


def _count_jacobians(model, st, key):
    # proteus calls attachModels more than once; wrap each model only once
    if getattr(model, '_vdft_counted', False):
        return
    model._vdft_counted = True
    orig = model.getJacobian

    def getJacobian(*args, **kwargs):
        st[key] += 1
        return orig(*args, **kwargs)
    model.getJacobian = getJacobian


def install(coefficients):
    st = dict(steps=0, newton_flow=0, newton_transport=0,
              flow_residual=0.0, flow_recovery=0.0, inflow_abs=0.0, inflow_net=0.0,
              drift=0.0, c_min=np.inf, c_max=-np.inf, mb_error=None,
              mb_flux=None, mb_history=None, fluid_mass=None, fluid_mass_0=None,
              t=None, t_wall0=time.time(), ranks=None, edge_flux=None,
              mask0_rows=None, notes=[])
    prev = {'m_new': None, 't': None}
    # Swap the instance's class rather than binding closures to it: proteus deep-copies the
    # p-module's coefficients, and a closure would keep acting on the original object.
    base = type(coefficients)

    class Metered(base):
        def attachModels(self, modelList):
            r = base.attachModels(self, modelList)
            try:
                _count_jacobians(self.model, self._vdft, 'newton_transport')
                _count_jacobians(self.vModel, self._vdft, 'newton_flow')
            except Exception as e:   # never break a run for a diagnostic
                self._vdft['notes'].append("jacobian counters not installed: %r" % (e,))
            return r

        def postStep(self, t, firstStep=False):
            r = base.postStep(self, t, firstStep=firstStep)
            try:
                _update(self, self._vdft, self._vdft_prev)
            except Exception as e:
                self._vdft['notes'].append("t=%r: %r" % (t, e))
            return r

    Metered.__name__ = Metered.__qualname__ = base.__name__
    coefficients.__class__ = Metered
    coefficients._vdft = st
    coefficients._vdft_prev = prev
    return st


def _update(coefficients, st, prev):
    m = coefficients.model
    v = coefficients.vModel
    ti = m.timeIntegration
    t, dt = float(ti.t), float(ti.dt)
    if dt <= 0.0 or (prev['t'] is not None and t <= prev['t']):
        return
    prev['t'] = t
    st['steps'] += 1
    st['t'] = t
    st['ranks'] = _comm().size()
    st['edge_flux'] = int(getattr(m, '_use_edge_flux', 0))
    no = m.mesh.nNodes_owned
    ML = np.asarray(m.ML)[:no]
    # flow model: accepted mass, old-time mass, physical boundary flux
    fct = bool(getattr(v, 'fctCorrectionApplied', False))
    m_new = np.asarray(v.mAccepted if fct else v.mLow)[:no]
    mn = np.asarray(v.mn)[:no]
    # physical boundary inflow per node: the face integrals (boundaryFluxDOF) plus the
    # weak-Dirichlet boundary block, which Richards routes through its edge loop
    # (edgeFluxLow) but not through the interior exchange edgeFluxInterior; summed over
    # owned rows the interior exchanges cancel, so the row sums of edgeFluxLow add
    # exactly that block
    rowptr = v.MC_global.getCSRrepresentation()[0]
    rows_low = np.add.reduceat(np.asarray(v.edgeFluxLow), rowptr[:-1])
    bflux = (np.asarray(v.boundaryFluxDOF)[:len(rows_low)] + rows_low)[:no]
    mask = np.asarray(getattr(v, 'bc_mask', np.ones_like(mn)))[:no] != 0.0
    st['mask0_rows'] = int(_gsum((~mask).sum()))
    store = _gsum((ML*(m_new - mn))[mask].sum())
    b = _gsum(bflux[mask].sum())
    st['flow_residual'] += store - dt*b
    st['inflow_net'] += dt*b
    st['inflow_abs'] += dt*abs(b)
    if prev['m_new'] is not None and np.shape(prev['m_new']) == np.shape(mn):
        st['flow_recovery'] += _gsum((ML*(mn - prev['m_new']))[mask].sum())
    prev['m_new'] = m_new.copy()
    fluid = _gsum((ML*m_new).sum())
    st['fluid_mass'] = fluid
    if st['fluid_mass_0'] is None:
        st['fluid_mass_0'] = _gsum((ML*mn).sum())
    # transport: carried storage vs accepted flow mass, bounds, solute MB error
    M = np.asarray(m.M_dof)[:no]
    st['drift'] = max(st['drift'], abs(_gsum((ML*(M - m_new)).sum()))/max(fluid, 1e-300))
    u = np.asarray(m.u[0].dof)[:no]
    st['c_min'] = min(st['c_min'], _gmin(u.min()))
    st['c_max'] = max(st['c_max'], _gmax(u.max()))
    cs = getattr(m, 'continuityStats', None)
    if cs:
        st['mb_error'] = cs.get('total_frac')
        solute = abs(cs.get('solute', 0.0)) or 1e-300
        st['mb_flux'] = cs.get('spurious_int', 0.0)/solute
        st['mb_history'] = cs.get('jump_int', 0.0)/solute
    st['wall_s'] = time.time() - st['t_wall0']
    err = st['flow_residual'] + st['flow_recovery']
    st['flow_balance_error'] = abs(err)/max(abs(fluid), 1e-300)
    st['flow_balance_error_vs_inflow'] = abs(err)/max(st['inflow_abs'], 1e-300)
    if _comm().rank() == 0:
        out = {k: v_ for k, v_ in st.items() if k != 't_wall0'}
        tmp = _OUT + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=1, default=float)
        os.replace(tmp, _OUT)
