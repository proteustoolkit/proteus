"""RDLS.PsiTC, the pseudo-transient step controller of the redistancing model, on the vortex2D deck.

redist_vortex_2d_n.py takes the Newton branch (``redist_Newton`` True, the default) unless the deck sets it False; no test did, so the controller
(``RDLS.PsiTC``: updateSubstep, stepExact_model, initialize_dt_model, choose_dt_model) never ran. Run in-process (as test_mcorr_bisection.py), two steps of the
level set transport, redistancing by pseudo-time continuation each. Checked: the model's controller is PsiTC, it substepped (updateSubstep and choose_dt_model called at least once per step), the redistanced field is finite, and has the sign of the level set it came from, up to the corrector's shift of it (< 0.1 he).
"""
from importlib import reload

import numpy as np
import pytest

from proteus import NumericalSolution
from . import (vortex2D, ls_vortex_2d_p, redist_vortex_2d_p, vof_vortex_2d_p, ls_consrv_vortex_2d_p, ls_vortex_2d_n,
               redist_vortex_2d_n, vof_vortex_2d_n, ls_consrv_vortex_2d_n, ls_vortex_2d_so)


def _decks(newton):
    reload(vortex2D)
    vortex2D.redist_Newton = newton
    reload(redist_vortex_2d_n)


@pytest.fixture
def psitc_decks():
    _decks(False)
    yield
    _decks(True)      # the other vortex2D tests import these modules: put the defaults back


def test_the_pseudo_transient_controller_redistances_the_level_set(opts, tmp_path, monkeypatch, psitc_decks):
    from proteus import default_s
    from proteus.kernels import RDLS
    monkeypatch.chdir(tmp_path)
    calls = {"updateSubstep": 0, "choose_dt_model": 0}
    for name in calls:
        def counted(self, _f=getattr(RDLS.PsiTC, name), _n=name):
            calls[_n] += 1
            return _f(self)
        monkeypatch.setattr(RDLS.PsiTC, name, counted)
    reload(default_s)
    so = ls_vortex_2d_so
    tnList = so.tnList
    so.tnList = tnList[:3]
    try:
        ns = NumericalSolution.NS_base(so, [ls_vortex_2d_p, redist_vortex_2d_p, vof_vortex_2d_p, ls_consrv_vortex_2d_p],
                                       [ls_vortex_2d_n, redist_vortex_2d_n, vof_vortex_2d_n, ls_consrv_vortex_2d_n],
                                       [default_s] * len(so.pnList), opts)
        ns.calculateSolution(so.name + "_psitc")
    finally:
        so.tnList = tnList
    rd = [m for m in ns.modelList if type(m.levelModelList[-1]).__module__.endswith("RDLS")][0]
    ls = [m for m in ns.modelList if type(m.levelModelList[-1]).__module__.endswith("NCLS")][0]
    assert type(rd.stepController) is RDLS.PsiTC
    assert calls["updateSubstep"] >= 2 and calls["choose_dt_model"] >= 2
    d = rd.levelModelList[-1].u[0].dof
    phi = ls.levelModelList[-1].u[0].dof
    assert np.isfinite(d).all()
    he = float(vortex2D.he)
    flipped = np.sign(d) != np.sign(phi)
    # the mass corrector shifts the level set by a constant after the redistancing: only nodes within that shift of the interface may differ in sign
    assert np.abs(phi[flipped]).max(initial=0.0) < 0.1 * he, "redistancing changed the sign of the level set away from the interface"
    assert np.abs(d - phi).max() < 0.1 * he
