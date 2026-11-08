"""TADR's recorded boundary flux on the edge-flux branches (USE_EDGE_FLUX) is the flux the update applied.

With the flow model's edge fluxes the solute's advective boundary exchange is nodal, B_i c_i + max(B_i, 0)
(c_in - c_i) into node i (B = nodalBoundaryFlux), and the dispersive one on a weak Dirichlet face is a lumped
penalty. ebqe_flux (and the "boundary_flux" diagnostic that integrates it) recorded n.f(c_h) with TADR's own
velocity plus the IIPG dispersive flux instead. On the levee deck below (fresh water, cin = 0.7 at leftTop) the first
step's recorded flux was -3.49 against an applied -68.8 (-2.65 against -68.0 for the advective part), still 46% off
at t = 0.029 and 10% at 0.063, within 3% after 0.09. Now the advective exchange of each boundary node is spread over its boundary faces by the lumped
boundary mass, and with no solute source the closed budget is the check: int ebqe_flux dS = -(S^{n+1} - S^n)/dt,
S = sum_i ML_i M_i c_i.
"""
import importlib.util
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("vdft_compatible_helpers", os.path.join(HERE, "test_vdft_compatible.py"))
tvc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tvc)


@pytest.fixture
def levee_deck(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    modules = tvc._load(tvc.CASE_D, tvc.CASE_D_MODULES)
    yield modules
    tvc._unload(tvc.CASE_D_MODULES)


@pytest.mark.parametrize("stab,rtol", [(5, 1e-10), (4, 1e-4)])
def test_edge_flux_branches_record_the_applied_boundary_flux(levee_deck, monkeypatch, stab, rtol):
    """ImplicitEV (5): measured 6.6e-13 relative (the Newton tolerance). Explicit EV (4): measured 4.0e-6 (the
    recorded flux is evaluated at c^n with the step's B, the budget over the step)."""
    from proteus.mprans import TADR
    rows = []
    post = TADR.Coefficients.postStep

    def postStep(self, t, firstStep=False):
        r = post(self, t, firstStep=firstStep)
        m = self.model
        rows.append((t, float(np.sum(m.ML*m.M_dof*m.u[0].dof)),
                     float(np.sum(m.ebqe[('advectiveFlux', 0)]*m.ebqe['dS']))))
        return r

    monkeypatch.setattr(TADR.Coefficients, "postStep", postStep)
    domain, rp, rn, tp, tn, so = levee_deck
    tvc._edge_flux_levee_transport(tp, tn, stab, 1, c0=0.0, cin=0.7)
    tvc._direct_solvers(rn, tn)
    so.tnList = [0.0, 1.0e-8, 2.0e-8, 1.0e-2]
    ns = tvc._run(so, [rp, tp], [rn, tn], 'tadr_edge_flux_recording_%d' % stab)
    assert ns.modelList[1].levelModelList[-1]._use_edge_flux == 1
    steps = [(t, -(S - S0)/(t - t0), rec) for (t0, S0, _), (t, S, rec) in zip(rows[:-1], rows[1:]) if t > t0]
    assert len(steps) > 5
    applied = np.array([a for _, a, _ in steps])
    recorded = np.array([r for _, _, r in steps])
    assert applied.max() < -10.0, applied          # solute enters: an exchange worth checking
    np.testing.assert_allclose(recorded, applied, rtol=rtol)
