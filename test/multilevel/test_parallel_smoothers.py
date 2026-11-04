#!/usr/bin/env python
"""proteus's own smoothers, running in parallel.

They update the unknowns a rank owns and subtract the correction from the
whole local vector, ghosts included -- but a ghost entry's correction belongs
to the rank that owns it, so the values left behind are wrong.  The next
residual reads them, because the subdomain matrix couples owned rows to ghost
columns.  Refreshing the ghosts from their owners after each update is what
makes the smoothers parallel; it is also all that is needed, given node-based
partitioning makes the subdomain matrix correct for the rows a rank owns.

Two further things had to change for this to run at all: ``par_du`` was built
un-ghosted (KSP only ever reads its owned range, so it never needed them), and
Transport asserted outright that the linear solver must be ``KSP_petsc4py`` in
parallel.

A Schwarz smoother needs two further things, because unlike Jacobi and
Gauss-Seidel it reads the residual across a whole star rather than at the one
unknown it is updating:

* its sweep is restricted to stars centred on nodes the rank owns -- those are
  correct square systems, since an owned row carries all of its columns, while
  a star centred on a ghost node is built from a row the subdomain holds only
  partially;
* the residual's ghost entries are given their owners' values before the local
  solves read them, or the iteration is driven by noise and stalls.

The convergence test also has to use a real parallel norm. self.norm runs over
the whole local vector, ghost rows included, where the residual is meaningless;
counting them makes the reported norm wrong and rank-dependent. Reduced over
owned entries it is the same number every decomposition computes.
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


def _run(smoother, n_ranks):
    mpiexec = _mpiexec()
    if mpiexec is None:
        pytest.skip('no MPI launcher available')
    helper = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'parallel_smoother_mpi.py')
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get('PYTHONPATH', '')]).strip(os.pathsep)
    result = subprocess.run([mpiexec, '-n', str(n_ranks), sys.executable, helper, smoother],
                            capture_output=True, text=True, env=env, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    for line in result.stdout.splitlines():
        if line.startswith('RESULT'):
            return float(line.split('checksum=')[1].split()[0])
    raise AssertionError(result.stdout + result.stderr)


@pytest.mark.parametrize('smoother', ['Jacobi', 'GaussSeidel', 'StarILU'])
@pytest.mark.parametrize('n_ranks', [2, 3])
def test_smoother_reaches_the_serial_solution(smoother, n_ranks):
    serial = _run(smoother, 1)
    parallel = _run(smoother, n_ranks)
    assert abs(parallel - serial) < 1.0e-8 * abs(serial), (
        "%s on %i ranks gave %.12e against %.12e in serial"
        % (smoother, n_ranks, parallel, serial))


@pytest.mark.parametrize('n_ranks', [2, 3])
def test_jacobi_is_bit_for_bit_the_serial_iteration(n_ranks):
    """Jacobi's update is pointwise, so the decomposition cannot change it.

    Anything weaker than exact agreement here would mean a ghost value is
    still being read stale somewhere.
    """
    assert _run('Jacobi', n_ranks) == _run('Jacobi', 1)


def test_starbilu_is_refused_in_parallel():
    """Excluded for a different reason than it used to be.

    StarBILU returns a zero solution even in serial on this problem, so there
    is nothing to preserve in parallel until that is understood.
    """
    mpiexec = _mpiexec()
    if mpiexec is None:
        pytest.skip('no MPI launcher available')
    helper = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'parallel_smoother_mpi.py')
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get('PYTHONPATH', '')]).strip(os.pathsep)
    result = subprocess.run([mpiexec, '-n', '2', sys.executable, helper, 'StarBILU'],
                            capture_output=True, text=True, env=env, timeout=300)
    assert result.returncode != 0
    assert 'cannot be used in parallel' in result.stdout + result.stderr
