"""Smoke runs of the manuscript decks in vdft_reproduce/.

CI does not collect that directory (it holds parun decks and plotting scripts, not tests), so this
module runs each figure's deck for its first output time: copied into a temporary directory as its
run.sh does, serial, direct solvers. A run must reach its end time with a finite, bounded
concentration and, for the coupled decks, advect with the coupling the deck asks for.
"""
import glob
import importlib
import os
import shutil
import sys

import numpy as np
import pytest

from proteus import Context, LinearSolvers, NumericalSolution, default_s, default_so
from proteus.iproteus import opts as _iproteus_opts



@pytest.fixture
def opts():
    """1.9.x: the one RunOptions object proteus.iproteus creates (test_vdft_compatible does the same)."""
    return _iproteus_opts


HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.join(HERE, 'vdft_reproduce')

# simmons_uhd/run_level0.sh: UHD on the level-0 mesh
SIMMONS_L0 = dict(SIMMONS_CASE='UHD', SIMMONS_LEVEL='0', SIMMONS_LEVEL_Z='0', SIMMONS_THETA='4',
                  SIMMONS_SEED='0', SIMMONS_STAB='5', SIMMONS_COURANT='0.5', SIMMONS_FCT='1',
                  SIMMONS_DRAIN='corners', SIMMONS_DRAIN_FACES='1', SIMMONS_VISC='1',
                  SIMMONS_RICHARDS_FCT='1', SIMMONS_Q_FILE='data/q_UHD_ramp.csv',
                  SIMMONS_K0_PER_CELL='1',
                  # 0.25 min of the 48 min experiment: a few steps
                  SIMMONS_T_MINUTES='0.25')

# (deck directory, split-operator module, environment, end time or None to keep the deck's
# (shortened) tnList, expected (edge flux, edge velocity))
CASES = {
    # first output of each deck (T/nDTout)
    'henry_fct': ('henry', 'ls_CCS_so', {}, 30000.0 / 201, (1, 1)),
    'henry_lo': ('henry/low_order', 'ls_CCS_so', {}, 30000.0 / 201, (1, 1)),
    'bioswale': ('bioswale', 'ls_CCS_so', {'BIOSWALE_DENSITY': '1'}, 0.5 / 201, (1, 1)),
    # the levee to t = 1 (its first output is 20000/801 ~ 25): its 13,429-node mesh (nnx = 161, fixed in
    # domain_rg.py) and small start-up steps still cost ~40 s serial, against ~60 s to the first output
    'levee_density': ('levee_density/density', 'ls_CCS_so', {}, 1.0, (1, 1)),
    'levee_non_density': ('levee_density/non_density', 'ls_CCS_so', {}, 1.0, (1, 1)),
    # Elder on a 121 x 31 mesh (the paper's is 601 x 151) for one day
    'elder': ('elder', 'ls_CCS_so', {'ELDER_NNX': '121', 'ELDER_NNY': '31'}, 1.0, (1, 1)),
    # MMS T3 at N = 10, first time step
    'mms': ('mms', 'mms_so', {}, None, (1, 1)),
    'simmons_compatible': ('simmons_uhd', 'simmons_so', SIMMONS_L0, None, (1, 1)),
    'simmons_lagged': ('simmons_uhd', 'simmons_so', dict(SIMMONS_L0, SIMMONS_VELOCITY_COUPLING='lagged'),
                       None, (0, 0)),
    'simmons_eos': ('simmons_uhd', 'simmons_so', dict(SIMMONS_L0, SIMMONS_VELOCITY_COUPLING='eos'),
                    None, (0, 0)),
    'simmons_conservative_lagged': ('simmons_uhd', 'simmons_so',
                                    dict(SIMMONS_L0, SIMMONS_VELOCITY_COUPLING='lagged',
                                         SIMMONS_CONSERVATIVE_FORM='1', SIMMONS_FCT='0'), None, (0, 0)),
}

# concentrations of every case lie in [0, 1]; the MMS solution is only checked for finiteness.
# 1e-8 allows for the nonlinear solver tolerance on the bounds.
BOUND_TOL = 1.0e-8


def _copy_deck(deck, tmp_path):
    """Copy the deck's Python files (and its data directory, by link) into tmp_path; the module names."""
    src = os.path.join(BUNDLE, deck)
    names = []
    for f in sorted(glob.glob(os.path.join(src, '*.py'))):
        shutil.copy(f, tmp_path)
        names.append(os.path.splitext(os.path.basename(f))[0])
    if os.path.isdir(os.path.join(src, 'data')):
        os.symlink(os.path.join(src, 'data'), os.path.join(tmp_path, 'data'))
    return names


def _import_fresh(directory, names, modules):
    for name in names:
        sys.modules.pop(name, None)
    sys.path.insert(0, directory)
    try:
        return [importlib.import_module(m) for m in modules]
    finally:
        sys.path.remove(directory)


def _run(so, pList, nList, name, opts):
    so.name = name
    for p in pList:
        p.name = p.__name__
        # the decks inherit genMesh from proteus.default_p, which other test modules set to False
        p.genMesh = True
    for n in nList:
        n.multilevelLinearSolver = LinearSolvers.LU
        n.levelLinearSolver = LinearSolvers.LU
    so.sList = [default_s for p in pList]
    ns = NumericalSolution.NS_base(so, pList, nList, so.sList, opts)
    failed = ns.calculateSolution(name)
    return ns, failed


@pytest.mark.parametrize('case', sorted(CASES))
def test_coupled_deck(case, tmp_path, monkeypatch, opts, request):
    deck, so_module, env, t_end, coupling = CASES[case]
    if case == 'levee_non_density':
        # 1.9.x only: run serially (as here), the passive-tracer levee's flow Newton iteration diverges at
        # t = 0.043 and the step is not recovered; on 4 ranks (the manuscript's configuration) the same deck runs
        # through t = 0.054 and beyond, and cekees/main recovers the step. See the bundle README (On 1.9.x).
        request.applymarker(pytest.mark.xfail(
            reason='1.9.x: serial flow Newton divergence at t = 0.043 in the passive-tracer levee deck', strict=False))
    monkeypatch.chdir(tmp_path)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    names = _copy_deck(deck, str(tmp_path))
    try:
        so, = _import_fresh(str(tmp_path), names, [so_module])
        # parun imports the pnList modules by name; mms_so.py does not import them itself
        pList = _import_fresh(str(tmp_path), [], [p for p, n in so.pnList])
        nList = _import_fresh(str(tmp_path), [], [n for p, n in so.pnList])
        # the three start-up entries, then the shortened end time
        so.tnList = so.tnList[:3] + [t_end] if t_end is not None else so.tnList[:4]
        ns, failed = _run(so, pList, nList, 'smoke_' + case, opts)
        assert not failed
        transport = ns.modelList[-1].levelModelList[-1]
        assert ns.systemStepController.t_system_last == pytest.approx(so.tnList[-1])
        c = transport.u[0].dof
        assert np.isfinite(c).all()
        if case != 'mms':
            assert c.min() >= -BOUND_TOL and c.max() <= 1.0 + BOUND_TOL
        assert (transport._use_edge_flux, transport._use_edge_velocity) == coupling
    finally:
        for name in names:
            sys.modules.pop(name, None)


@pytest.mark.parametrize('fct', [False, True])
def test_zalesak_disk(fct, tmp_path, monkeypatch, opts):
    """zalesak/: TADR alone, STABILIZATION_TYPE 5, a short rotation of the disk on refinement 2."""
    monkeypatch.chdir(tmp_path)
    # thelper_tadr reads its options from the context string when it is imported
    monkeypatch.setattr(Context, 'contextOptionsString',
                        'problem=1 nd=2 T=0.02 nDTout=1 refinement=2 STABILIZATION_TYPE=5 cfl=0.5 '
                        'SSPOrder=2 FCT=%s' % fct)
    names = _copy_deck('zalesak', str(tmp_path))
    try:
        p, n = _import_fresh(str(tmp_path), names, ['thelper_tadr_p', 'thelper_tadr_n'])
        default_so.tnList = n.tnList
        ns, failed = _run(default_so, [p], [n], 'smoke_zalesak_fct%d' % fct, opts)
        assert not failed
        transport = ns.modelList[0].levelModelList[-1]
        assert ns.systemStepController.t_system_last == pytest.approx(n.tnList[-1])
        c = transport.u[0].dof
        assert np.isfinite(c).all()
        assert c.min() >= -BOUND_TOL and c.max() <= 1.0 + BOUND_TOL
    finally:
        for name in names:
            sys.modules.pop(name, None)
