#!/usr/bin/env python
"""Proteus's own parallel multigrid with petsc4py made unimportable.

Proteus's solvers need from PETSc only a ghosted vector and, for multigrid,
the transfer operators applied to owned rows; without petsc4py those come from
mpi4py (LinearAlgebraTools.ParVec_mpi4py, GhostExchange, DistMat) and the
coarse level is solved by a redundant SuperLU factorization (RedundantLU). The
PETSc C library is still used by the graph partitioner, unless the partition is
prescribed (test_partition_arrays.py runs multigrid with no PETSc at all).

* The building blocks agree with PETSc's, on uneven layouts with empty ranks:
  ParVec_mpi4py's forward insert and reverse add exactly, DistMat's mult and
  multTranspose to rounding. They agree exactly on the conda builds; the fleet's
  pip PETSc on aarch64 (spark2) differs by 4.4e-16-8.9e-16, a fused
  multiply-add, so the products are held to the rounding bound distmat_mpi.py
  prints (about 1e-14 here) instead of to zero.
* MGM run with petsc4py blocked (without_petsc4py.py) reproduces the PETSc path
  at every rank count, for nodal P1, P2, discontinuous P1 and hexahedral Q1. The
  PETSc path is in turn the serial cycle (test_parallel_mgm.py and friends), so
  this is the serial answer too.
* Nothing imports petsc4py behind the block's back: without_petsc4py.py fails
  the run if it finds petsc4py in sys.modules at the end.
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


def _launch(args, n_ranks, blocked, timeout=300):
    mpiexec = _mpiexec()
    if mpiexec is None:
        pytest.skip('no MPI launcher available')
    here = os.path.dirname(os.path.abspath(__file__))
    script = [os.path.join(here, 'without_petsc4py.py')] if blocked else []
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get('PYTHONPATH', '')]).strip(os.pathsep)
    # CI runners have two cores; let Open MPI (4.x and 5.x/PRRTE) start more
    # ranks than that. MPICH oversubscribes by default and ignores these.
    env.setdefault('OMPI_MCA_rmaps_base_oversubscribe', '1')
    env.setdefault('PRTE_MCA_rmaps_default_mapping_policy', ':oversubscribe')
    result = subprocess.run([mpiexec, '-n', str(n_ranks), sys.executable] + script +
                            [os.path.join(here, args[0])] + list(args[1:]),
                            capture_output=True, text=True, env=env, timeout=timeout)
    assert result.returncode == 0, result.stdout + result.stderr
    for line in result.stdout.splitlines():
        if line.startswith('RESULT'):
            return dict(item.split('=', 1) for item in line.split()[1:])
    raise AssertionError(result.stdout + result.stderr)


@pytest.mark.parametrize('n_ranks', [1, 2, 3, 5])
def test_building_blocks_agree_with_petsc(n_ranks):
    pytest.importorskip('petsc4py')
    got = _launch(['distmat_mpi.py', '40', '23'], n_ranks, blocked=False)
    for key in ('forward', 'forward_truth', 'reverse'):
        assert float(got[key]) == 0.0, got
    for key in ('mult', 'multTranspose'):
        assert float(got[key]) <= float(got[key + '_bound']), got


CASES = {
    'p1': ['mgm_parallel_mpi.py', '3', 'Jacobi'],
    'p1x2': ['mgm_parallel_mpi.py', '3', 'Jacobi', '3', '3', '2'],
    'p2': ['space_mg_mpi.py', 'p2', '3', 'mgm'],
    'dgp1': ['space_mg_mpi.py', 'dgp1', '3', 'mgm'],
    'hex': ['cube_mg_mpi.py', 'hex', '3', '3', 'mgm'],
}


@pytest.mark.parametrize('case', sorted(CASES))
@pytest.mark.parametrize('n_ranks', [1, 2, 3, 4])
def test_mgm_without_petsc4py_is_the_petsc_path(case, n_ranks):
    pytest.importorskip('petsc4py')
    petsc = _launch(CASES[case], n_ranks, blocked=False)
    mpi = _launch(CASES[case], n_ranks, blocked=True)
    key = 'checksums' if 'checksums' in petsc else 'checksum'
    for a, b in zip(petsc[key].split(','), mpi[key].split(',')):
        a, b = float(a), float(b)
        assert abs(a - b) <= 1.0e-11 * abs(a), (case, n_ranks, petsc, mpi)
