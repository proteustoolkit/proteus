#!/usr/bin/env python
"""Multigrid for Q1 on regular quadrilateral and hexahedral grids, in parallel.

The same cross-checks the simplex hierarchies get (test_petsc_pcmg.py): with
Jacobi smoothing and exact transfers, proteus's MGM must give the serial answer
on any number of ranks; PETSc's PCMG over the same hierarchy must reproduce it;
and PCMG-preconditioned GMRES must reach the direct solution. Tensor-product
elements add one thing the simplices did not exercise -- transfer operators
with up to eight parents per fine node, partitioned across ranks.
"""
import os
import shutil
import subprocess
import sys

import pytest


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ('mpiexec', 'mpirun'):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which('mpiexec') or shutil.which('mpirun')


def _run(kind, n_levels, mode, n_ranks, nn=3):
    mpiexec = _mpiexec()
    if mpiexec is None:
        pytest.skip('no MPI launcher available')
    helper = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cube_mg_mpi.py')
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get('PYTHONPATH', '')]).strip(os.pathsep)
    result = subprocess.run([mpiexec, '-n', str(n_ranks), sys.executable, helper,
                             kind, str(nn), str(n_levels), mode],
                            capture_output=True, text=True, env=env, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    for line in result.stdout.splitlines():
        if line.startswith('RESULT'):
            return float(line.split('checksum=')[1].split()[0])
    raise AssertionError(result.stdout + result.stderr)


@pytest.mark.parametrize('kind', ['quad', 'hex'])
@pytest.mark.parametrize('n_levels', [2, 3])
@pytest.mark.parametrize('n_ranks', [2, 3])
def test_parallel_cycle_is_the_serial_cycle(kind, n_levels, n_ranks):
    serial = _run(kind, n_levels, 'mgm', 1)
    parallel = _run(kind, n_levels, 'mgm', n_ranks)
    assert abs(parallel - serial) < 1.0e-11 * abs(serial), (
        "%s MGM on %i ranks gave %.14e against %.14e in serial" % (kind, n_ranks, parallel, serial))


@pytest.mark.parametrize('kind', ['quad', 'hex'])
@pytest.mark.parametrize('n_ranks', [2, 3])
def test_pcmg_cycle_is_the_mgm_cycle(kind, n_ranks):
    mgm = _run(kind, 3, 'mgm', 1)
    pcmg = _run(kind, 3, 'pcmg', n_ranks)
    assert abs(pcmg - mgm) < 1.0e-11 * abs(mgm), (kind, n_ranks, pcmg, mgm)


@pytest.mark.parametrize('kind', ['quad', 'hex'])
def test_pcmg_preconditioned_gmres_reaches_the_direct_solution(kind):
    direct = _run(kind, 3, 'direct', 1)
    converged = _run(kind, 3, 'pcmg_its', 3)
    assert abs(converged - direct) < 1.0e-9 * abs(direct), (kind, converged, direct)
