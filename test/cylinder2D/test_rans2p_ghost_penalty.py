"""The ghost penalty of RANS2P/RANS2P2D (jump of the normal derivative of pressure and velocity across the faces
of the cut elements), assembled through proteus::unfitted::jumpPenalty.

Only a run with ``useExact`` has cut elements, and the stock cylinder decks have none (``useExact`` False), so the
penalty was reached by no test. This one runs the immersed-boundary cylinder decks with ``useExact`` on, evaluates the
residual and the Jacobian ONCE, at the initial state of the first step plus a fixed random perturbation of the
unknowns (the same in every build), and compares norms and checksums with the values the pre-refactor
per-dof ``std::map`` assembly gave (recorded before it was replaced, to the digits printed). The penalty is
O(1) in those numbers: switching it off moves single Jacobian entries by 1.4 (2D) and 2.7 (3D). Serial, one step.
"""
import os
import sys
import importlib

import numpy as np
import pytest

import proteus
from proteus import Context, default_s

HERE = os.path.dirname(os.path.abspath(__file__))

# (deck dir, context options, ball radius or None for the deck's own); stats = 2-norm, sum, sum_i i x_i
CASES = {
    '2d': ('ibm_rans2p', 'T=0.02 onlySaveFinalSolution=True', 0.12),
    '3d': ('ibm_rans2p_3D', 'T=0.02 onlySaveFinalSolution=True genMesh=False Refinement=7', None),
}
# the values of the per-dof std::map assembly (b9d98f5d), residual then Jacobian; the jumpPenalty build agrees to 1e-15
PINNED = {
    '2d': ([2.9100900972465658, -1.2650302853880007, -138.63059745666038],
           [14.501532583231377, 42.4842380157602, 116820.16600572805]),
    '3d': ([2.3790038689807425, -0.04181976366686011, 211.69160735654899],
           [9.831608956152154, 0.5273275260467332, 19937.9591214808]),
}


class _Stop(Exception):
    pass


def _stats(a):
    a = np.asarray(a, dtype=float).ravel()
    return np.array([np.linalg.norm(a), a.sum(), np.dot(np.arange(a.size), a)])


@pytest.mark.parametrize('case', sorted(CASES))
def test_ghost_penalty_through_jumpPenalty(case, opts, monkeypatch):
    deck, options, radius = CASES[case]
    deckdir = os.path.join(HERE, deck)
    before = set(os.listdir(deckdir))
    monkeypatch.chdir(deckdir)
    monkeypatch.syspath_prepend(deckdir)
    from proteus.mprans import RANS2P
    LevelModel = RANS2P.LevelModel
    getResidual, getJacobian = LevelModel.getResidual, LevelModel.getJacobian
    seen = {}

    def residual(self, u, r):
        if radius is not None:
            self.coefficients.ball_radius[:] = radius
        self.coefficients.useExact = True
        out = getResidual(self, u, r)
        if 'res' not in seen:
            uf = np.array(u) + 0.2*np.random.default_rng(1).standard_normal(len(u))
            rf = np.zeros(len(u))
            getResidual(self, uf, rf)
            jm = self.jacobian
            getJacobian(self, jm)
            seen['res'] = _stats(rf)
            seen['jac'] = _stats(jm.getCSRrepresentation()[2])
            raise _Stop()
        return out

    monkeypatch.setattr(LevelModel, 'getResidual', residual)
    Context.contextOptionsString = options
    package = 'test.cylinder2D.' + deck
    try:
        so = importlib.import_module(package + '.cylinder_so')
        pList, nList = [], []
        for (pModule, nModule) in so.pnList:
            pList.append(importlib.import_module(package + '.' + pModule))
            nList.append(importlib.import_module(package + '.' + nModule))
            if pList[-1].name is None:
                pList[-1].name = pModule
        sList = so.sList if so.sList != [] else [default_s for _ in so.pnList]
        so.name += '_ghost_penalty'
        opts.gatherArchive = True
        with pytest.raises(_Stop):
            ns = proteus.NumericalSolution.NS_base(so, pList, nList, sList, opts)
            ns.calculateSolution(so.name)
    finally:
        for f in set(os.listdir(deckdir)) - before:
            path = os.path.join(deckdir, f)
            if os.path.isfile(path):
                os.remove(path)
    if case not in PINNED:
        print('PINNED', case, repr(seen['res'].tolist()), repr(seen['jac'].tolist()))
        pytest.fail('no pinned values for ' + case)
    res, jac = PINNED[case]
    np.testing.assert_allclose(seen['res'], res, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(seen['jac'], jac, rtol=1e-12, atol=1e-12)
