"""RANS2P's cut-cell code (the ``icase_s == 0`` branches, the surface weights ``gf_s.D``, the ghost penalty) reached from a deck.

The stock cylinder decks never reach it: with ``useExact`` False ``GeneralizedFunctions_mix::calculate`` returns icase 1 for
every element (found by C0 round 13). This runs the immersed-boundary cylinder deck (2D, P1) with ``useExact`` True, ball of
radius 0.12 at (0.2, 0.2), evaluates the residual at the initial state plus a fixed random perturbation, changes the SOLID
velocity from 0 to d in x, and looks at which rows of the residual respond:

* the response is real (the solid velocity enters through the particle's surface terms) and it is confined to the nodes of
  the cut elements and of the elements across their faces (the ghost-penalty neighbours): nothing else can feel the solid
  velocity when the interface is sharp;
* the response norm is pinned (2D deck, serial, macOS arm64 and linux-64 agree: no node is within 1e-3 of the surface).

This is a new file next to ``test_rans2p_ghost_penalty.py``, which pins the ghost penalty itself.
"""
import importlib
import os

import numpy as np
import pytest

import proteus
from proteus import Context, default_s

HERE = os.path.dirname(os.path.abspath(__file__))
DECK = 'ibm_rans2p'
OPTIONS = 'T=0.02 onlySaveFinalSolution=True'
RADIUS = 0.12
# |R(solid velocity (1, 0)) - R(0)| with useExact True (recorded 2026-11-18, serial)
PINNED_RESPONSE = {True: 0.304176296839593}
# (A fresh process with useExact False gives 95 responding rows and the norm 0.20248057823663768: the regularized interface
# smears the response over a band. A second NS_base in the SAME process with useExact False reproduces the sharp response,
# 42 rows: state of the first run survives, so this file runs the sharp case only, once.)


class _Stop(Exception):
    pass


def _probe(opts, monkeypatch, useExact):
    deckdir = os.path.join(HERE, DECK)
    before = set(os.listdir(deckdir))
    monkeypatch.chdir(deckdir)
    monkeypatch.syspath_prepend(deckdir)
    from proteus.kernels import RANS2P
    LevelModel = RANS2P.LevelModel
    getResidual = LevelModel.getResidual
    seen = {}

    def residual(self, u, r):
        c = self.coefficients
        c.ball_radius[:] = RADIUS
        c.useExact = useExact
        out = getResidual(self, u, r)
        if 'dR' not in seen:
            uf = np.array(u) + 0.2 * np.random.default_rng(1).standard_normal(len(u))

            def R():
                rr = np.zeros(len(uf))
                getResidual(self, uf.copy(), rr)
                return rr
            c.ball_velocity[:] = 0.0
            r0 = R()
            c.ball_velocity[0, 0] = 1.0
            seen['dR'] = R() - r0
            c.ball_velocity[:] = 0.0
            mesh = self.mesh
            X = np.asarray(mesh.nodeArray)
            phi = np.linalg.norm(X[:, :2] - c.ball_center[0][:2], axis=1) - RADIUS
            en = np.asarray(mesh.elementNodesArray)
            cut = np.array([phi[e].min() * phi[e].max() < 0 for e in en])
            near = cut.copy()
            nb = np.asarray(mesh.elementNeighborsArray)         # (element, local face) -> neighbouring element or -1
            for e in np.nonzero(cut)[0]:
                for k in nb[e]:
                    if k >= 0:
                        near[k] = True
            seen['cut'], seen['near'] = int(cut.sum()), np.unique(en[near])
            seen['margin'] = np.abs(phi).min()
            seen['active'] = int(np.asarray(self.isActiveElement).sum())
            raise _Stop()
        return out

    monkeypatch.setattr(LevelModel, 'getResidual', residual)
    Context.contextOptionsString = OPTIONS
    package = 'test.cylinder2D.' + DECK
    try:
        so = importlib.import_module(package + '.cylinder_so')
        pList, nList = [], []
        for (pModule, nModule) in so.pnList:
            pList.append(importlib.import_module(package + '.' + pModule))
            nList.append(importlib.import_module(package + '.' + nModule))
            if pList[-1].name is None:
                pList[-1].name = pModule
        sList = so.sList if so.sList != [] else [default_s for _ in so.pnList]
        so.name += '_cut_cells'
        opts.gatherArchive = True
        with pytest.raises(_Stop):
            ns = proteus.NumericalSolution.NS_base(so, pList, nList, sList, opts)
            ns.calculateSolution(so.name)
    finally:
        for f in set(os.listdir(deckdir)) - before:
            path = os.path.join(deckdir, f)
            if os.path.isfile(path):
                os.remove(path)
    return seen


@pytest.fixture
def exact(opts, monkeypatch):
    return _probe(opts, monkeypatch, True)


def test_the_solid_velocity_reaches_the_residual_only_through_the_cut_cells(exact):
    dR = exact['dR']
    assert exact['cut'] > 0 and exact['margin'] > 1.0e-3
    rows = np.nonzero(dR)[0]
    assert rows.size > 0
    nodes = np.unique(rows // 3)                                   # (p, u, v) interleaved
    assert set(nodes) <= set(exact['near']), sorted(set(nodes) - set(exact['near']))
    np.testing.assert_allclose(np.linalg.norm(dR), PINNED_RESPONSE[True], rtol=1.0e-6)
