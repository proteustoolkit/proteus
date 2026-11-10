"""TwoPhaseFlow's RANS2P preconditioner setting (cekees/proteus#198, C19 round 3).

``SystemPhysics.modelDict['flow'].preconditioner`` picks the preconditioner of RANS2P's iterative solve
(``useSuperlu`` off): ``'auto'`` (the default: ``'schur_boomeramg'`` when PETSc has hypre, ``'schur_gamg'`` when it
does not), ``'schur_boomeramg'`` (a Schur complement fieldsplit with BoomerAMG blocks), ``'schur_gamg'`` (the same
with GAMG blocks, velocity aggressive coarsening off) or ``'asm'`` (GMRES + ASM with an LU per subdomain). An explicit ``'schur_boomeramg'`` without hypre, or an unknown name, makes the
problem refuse to initialize instead of failing at the first linear solve.
"""
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PERF = os.path.join(HERE, os.pardir, 'perf')

# load damBreakLevelSet through the perf harness (it applies params= before initializeAll),
# optionally with a PETSc that reports no hypre
LOAD = r"""
import sys
sys.path.insert(0, {perf!r})
import proteus.TwoPhaseFlow.utils.Parameters as P
if {no_hypre}:
    P.petsc_has_hypre = lambda: False
from proteus import Comm
Comm.init()
logged = []
_log = P.logEvent
def logEvent(msg, *args, **kwargs):
    if "RANS2P's" in str(msg) and 'no hypre' in str(msg):
        logged.append(str(msg))
    return _log(msg, *args, **kwargs)
P.logEvent = logEvent
import model_baseline_mpi as mb
from petsc4py import PETSc
try:
    so, pList, nList, sList, tp = mb.load_case({case!r}, 'final_time=0.01 dt_output=0.01 he=0.1', '0', {params!r})
except ValueError as e:
    print('REFUSED', e)
    sys.exit(0)
db = PETSc.Options()
print('LOADED pc_type=%s vel=%s pres=%s agg=%s side=%s lag=%s ctest=%s' % (db.getString('rans2p_pc_type'), db.getString('rans2p_fieldsplit_velocity_pc_type', '-'), db.getString('rans2p_fieldsplit_pressure_pc_type', '-'), db.getString('rans2p_fieldsplit_velocity_pc_gamg_aggressive_coarsening', '-'), db.getString('rans2p_ksp_pc_side'),
      db.getString('rans2p_pc_lag_newton'), [n.linearSolverConvergenceTest for p, n in zip(pList, nList) if p.name == 'rans2p'][0]))
print('LOGGED %d' % len(logged))
"""


def load(tmp_path, params, no_hypre=False, with_log=False):
    for name in os.listdir(HERE):
        if name.startswith('meshDambreak'):
            os.symlink(os.path.join(HERE, name), tmp_path / name)
    code = LOAD.format(perf=PERF, no_hypre=no_hypre, case=os.path.join(HERE, 'damBreakLevelSet.py'), params=params)
    proc = subprocess.run([sys.executable, '-c', code], cwd=str(tmp_path), capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    lines = [line for line in proc.stdout.splitlines() if line.startswith(('LOADED', 'REFUSED', 'LOGGED'))]
    return lines if with_log else lines[0]



def test_default_is_schur_boomeramg_with_hypre(tmp_path):
    from petsc4py import PETSc
    if not PETSc.Sys.hasExternalPackage('hypre'):
        pytest.skip('this PETSc has no hypre')
    line = load(tmp_path, '')
    assert line == 'LOADED pc_type=fieldsplit vel=hypre pres=hypre agg=- side=right lag=true ctest=r', line


GAMG = 'LOADED pc_type=fieldsplit vel=gamg pres=gamg agg=0 side=right lag=true ctest=r'


def test_default_without_hypre_is_schur_gamg_and_logged(tmp_path):
    line, logged = load(tmp_path, '', no_hypre=True, with_log=True)
    assert line == GAMG and logged == 'LOGGED 1', (line, logged)


def test_schur_gamg_on_request(tmp_path):
    line = load(tmp_path, 'flow.preconditioner=schur_gamg')
    assert line == GAMG, line


def test_asm_on_request(tmp_path):
    line = load(tmp_path, 'flow.preconditioner=asm')
    assert line == 'LOADED pc_type=asm vel=- pres=- agg=- side=right lag=true ctest=r', line


def test_schur_boomeramg_without_hypre_is_refused_at_setup(tmp_path):
    line = load(tmp_path, 'flow.preconditioner=schur_boomeramg', no_hypre=True)
    assert line.startswith('REFUSED') and 'hypre' in line, line


def test_an_unknown_preconditioner_is_refused(tmp_path):
    line = load(tmp_path, 'flow.preconditioner=ilu')
    assert line.startswith('REFUSED') and "'auto', 'asm', 'schur_boomeramg' or 'schur_gamg'" in line, line
