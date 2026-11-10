#!/usr/bin/env python
"""Multigrid for spaces whose degrees of freedom are not mesh vertices, in parallel.

Quadratic Lagrange, non-conforming P1 and discontinuous P1 cannot use the nodal
transfer operators, which name a fine dof by a pair of coarse vertices. Their
operators come from evaluating coarse basis functions at the fine interpolation
points, run on each rank's *subdomain* spaces over the subdomain parent/child
relations (MultilevelMesh.subdomainHierarchy). A finer level inherits the coarser
one's partition, so the result is complete on the rows a rank owns -- those are
translated to global dof numbers and kept, and nothing is gathered
(test_level_consistent_partitioning.py pins the partitioning side).

* The operators reproduce a function of the coarse space exactly, on any number
  of ranks, for every space. This is independent of how the discretization is
  assembled, so it covers the discontinuous space too.
* With Jacobi smoothing and an exact coarse solve, proteus's MGM cycle is the
  serial cycle at any rank count; PETSc's PCMG over the same hierarchy is the
  same cycle; and PCMG-preconditioned GMRES reaches the direct solution.
* A rank with no ghost entries must still take part in the ghost refreshes,
  which are collective. They were skipped on such a rank, so every other rank
  waited for it forever. A discontinuous problem on two ranks has exactly that
  rank.

* A discontinuous space gets the same discretization at any rank count, which
  needs element partitioning (every face neighbour of an owned element on the
  rank). Legacy p/n problems used to lose the numerics file's partitioning
  options on the way to the mesh and always got node partitioning.
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


def _launch(helper, args, n_ranks, timeout):
    mpiexec = _mpiexec()
    if mpiexec is None:
        pytest.skip('no MPI launcher available')
    helper = os.path.join(os.path.dirname(os.path.abspath(__file__)), helper)
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get('PYTHONPATH', '')]).strip(os.pathsep)
    # a rank must own elements on the coarse mesh (a discontinuous space needs it), which the partition
    # ParMETIS makes does not promise in every build (a fleet node left rank 2 empty): pin the initial one
    env['PETSC_OPTIONS'] = (env.get('PETSC_OPTIONS', '') + ' -mat_partitioning_type current').strip()
    result = subprocess.run([mpiexec, '-n', str(n_ranks), sys.executable, helper] + args,
                            capture_output=True, text=True, env=env, timeout=timeout)
    assert result.returncode == 0, result.stdout + result.stderr
    for line in result.stdout.splitlines():
        if line.startswith('RESULT'):
            return line
    raise AssertionError(result.stdout + result.stderr)


def _checksum(space, mode, n_ranks, n_levels=3, timeout=300):
    line = _launch('space_mg_mpi.py', [space, str(n_levels), mode], n_ranks, timeout)
    return float(line.split('checksum=')[1].split()[0])


@pytest.mark.parametrize('space,degree', [('p2', 1), ('p2', 2), ('ncp1', 1), ('dgp1', 1)])
@pytest.mark.parametrize('n_ranks', [1, 2, 3])
def test_transfers_reproduce_a_coarse_space_function(space, degree, n_ranks):
    line = _launch('space_transfer_mpi.py', [space, '3', str(degree)], n_ranks, 300)
    err = float(line.split('maxerr=')[1].split()[0])
    scale = float(line.split('scale=')[1].split()[0])
    assert scale > 0.5, "the interpolated function is trivially small: " + line
    assert err < 1.0e-12 * scale, line


@pytest.mark.parametrize('space', ['p2', 'ncp1', 'dgp1'])
@pytest.mark.parametrize('n_ranks', [2, 3])
def test_parallel_cycle_is_the_serial_cycle(space, n_ranks):
    serial = _checksum(space, 'mgm', 1)
    parallel = _checksum(space, 'mgm', n_ranks)
    assert abs(parallel - serial) < 1.0e-11 * abs(serial), (space, n_ranks, parallel, serial)


@pytest.mark.parametrize('space', ['p2', 'ncp1', 'dgp1'])
@pytest.mark.parametrize('n_ranks', [2, 3])
def test_pcmg_cycle_is_the_mgm_cycle(space, n_ranks):
    mgm = _checksum(space, 'mgm', 1)
    pcmg = _checksum(space, 'pcmg', n_ranks)
    assert abs(pcmg - mgm) < 1.0e-11 * abs(mgm), (space, n_ranks, pcmg, mgm)


@pytest.mark.parametrize('space', ['p2', 'ncp1', 'dgp1'])
def test_pcmg_preconditioned_gmres_reaches_the_direct_solution(space):
    direct = _checksum(space, 'direct', 1)
    converged = _checksum(space, 'pcmg_its', 3)
    assert abs(converged - direct) < 1.0e-9 * abs(direct), (space, converged, direct)


def test_a_rank_without_ghosts_still_joins_the_collective_refreshes():
    # completing is the assertion: the failure was a hang
    _checksum('dgp1', 'mgm', 2, timeout=240)


def test_a_rank_that_owns_no_coarse_dofs_still_solves():
    # the coarse NC-P1 level has 16 dofs; on 6 ranks one rank owns none of them,
    # which the owned-rows CSR extraction used to reject. Coarse levels with idle
    # ranks are normal for multigrid at scale.
    serial = _checksum('ncp1', 'mgm', 1)
    parallel = _checksum('ncp1', 'mgm', 6)
    assert abs(parallel - serial) < 1.0e-11 * abs(serial), (parallel, serial)


@pytest.mark.parametrize('n_ranks', [2, 3, 4])
def test_discontinuous_direct_solve_does_not_depend_on_rank_count(n_ranks):
    # A discontinuous space needs every face neighbour of an owned element on the
    # rank, which only element partitioning provides. The numerics file asks for
    # it, but legacy p/n problems used to drop n.parallelPartitioningType and
    # n.nLayersOfOverlapForParallel on the way to the mesh, so they always got
    # node partitioning: the missing neighbours' faces were assembled as domain
    # boundary and the answer changed with the rank count (16.76 / 17.77 / 12.91
    # / 14.86 at 1-4 ranks).
    serial = _checksum('dgp1', 'ksplu', 1, n_levels=1)
    parallel = _checksum('dgp1', 'ksplu', n_ranks, n_levels=1)
    assert abs(parallel - serial) < 1.0e-11 * abs(serial), (n_ranks, parallel, serial)
