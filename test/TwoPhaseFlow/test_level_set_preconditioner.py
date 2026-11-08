"""TwoPhaseFlow's VOF, NCLS and MCorr preconditioner setting (C19 round 4), the same rule as RANS2P's.

``SystemPhysics.modelDict[name].preconditioner`` (name in vof, ncls, mcorr) picks the preconditioner of the model's
iterative solve (``useSuperlu`` off): ``'auto'`` (the default: ``'boomeramg'`` when PETSc has hypre, ``'asm'`` with a
log line when it does not), ``'boomeramg'`` or ``'asm'`` (ASM with an LU per subdomain). Before, these three asked for
``pc_type hypre`` whatever the PETSc, so a PETSc without hypre failed at the first linear solve. ``'gamg'`` (PETSc's own
AMG) is there on request. An explicit
``'boomeramg'`` without hypre, or an unknown name, makes the problem refuse to initialize.
"""
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PERF = os.path.join(HERE, os.pardir, 'perf')
MODELS = ('vof', 'ncls', 'mcorr')

# load damBreakLevelSet with superlu=0 through the perf harness (it applies params= before initializeAll),
# optionally with a PETSc that reports no hypre
LOAD = r"""
import sys
sys.path.insert(0, {perf!r})
import proteus.TwoPhaseFlow.utils.Parameters as P
if {no_hypre}:
    P.petsc_has_hypre = lambda: False
from proteus import Comm, Profiling
Comm.init()
logged = []
log = Profiling.logEvent
def logEvent(msg, *args, **kwargs):
    if 'no hypre' in str(msg):
        logged.append(str(msg))
    return log(msg, *args, **kwargs)
P.logEvent = logEvent
import model_baseline_mpi as mb
from petsc4py import PETSc
try:
    so, pList, nList, sList, tp = mb.load_case({case!r}, 'final_time=0.01 dt_output=0.01 he=0.1', '0', {params!r})
except ValueError as e:
    print('REFUSED', e)
    sys.exit(0)
db = PETSc.Options()
print('LOADED ' + ' '.join('%s=%s/%s/%s' % (m, db.getString(m + '_ksp_type'), db.getString(m + '_pc_type'),
                                           db.getString(m + '_sub_pc_type', '-')) for m in {models!r}))
print(' '.join(['LOGGED'] + sorted(m for m in {models!r} if any(m + "'s" in line for line in logged))))
"""


def load(tmp_path, params, no_hypre=False):
    for name in os.listdir(HERE):
        if name.startswith('meshDambreak'):
            os.symlink(os.path.join(HERE, name), tmp_path / name)
    code = LOAD.format(perf=PERF, no_hypre=no_hypre, case=os.path.join(HERE, 'damBreakLevelSet.py'), params=params,
                       models=MODELS)
    proc = subprocess.run([sys.executable, '-c', code], cwd=str(tmp_path), capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return [line for line in proc.stdout.splitlines() if line.startswith(('LOADED', 'REFUSED', 'LOGGED'))][-2:]


def test_default_is_boomeramg_with_hypre(tmp_path):
    from petsc4py import PETSc
    if not PETSc.Sys.hasExternalPackage('hypre'):
        pytest.skip('this PETSc has no hypre')
    loaded, logged = load(tmp_path, '')
    assert loaded == 'LOADED vof=gmres/hypre/- ncls=gmres/hypre/- mcorr=fgmres/hypre/-', loaded
    assert logged == 'LOGGED', logged


def test_default_without_hypre_is_asm_and_logged(tmp_path):
    loaded, logged = load(tmp_path, '', no_hypre=True)
    assert loaded == 'LOADED vof=gmres/asm/lu ncls=gmres/asm/lu mcorr=fgmres/asm/lu', loaded
    assert logged == 'LOGGED mcorr ncls vof', logged


def test_asm_on_request(tmp_path):
    loaded, logged = load(tmp_path, 'vof.preconditioner=asm ncls.preconditioner=asm mcorr.preconditioner=asm')
    assert loaded == 'LOADED vof=gmres/asm/lu ncls=gmres/asm/lu mcorr=fgmres/asm/lu', loaded
    assert logged == 'LOGGED', logged


def test_gamg_on_request_without_hypre(tmp_path):
    loaded, logged = load(tmp_path, 'vof.preconditioner=gamg ncls.preconditioner=gamg mcorr.preconditioner=gamg',
                          no_hypre=True)
    assert loaded == 'LOADED vof=gmres/gamg/- ncls=gmres/gamg/- mcorr=fgmres/gamg/-', loaded
    assert logged == 'LOGGED', logged


@pytest.mark.parametrize('model', MODELS)
def test_boomeramg_without_hypre_is_refused_at_setup(tmp_path, model):
    line = load(tmp_path, model + '.preconditioner=boomeramg', no_hypre=True)[-1]
    assert line.startswith('REFUSED') and 'hypre' in line and model + '.preconditioner' in line, line


def test_an_unknown_preconditioner_is_refused(tmp_path):
    line = load(tmp_path, 'ncls.preconditioner=ilu')[-1]
    assert line.startswith('REFUSED') and "'auto', 'asm', 'boomeramg' or 'gamg'" in line, line
