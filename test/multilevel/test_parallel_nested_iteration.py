#!/usr/bin/env python
"""Nonlinear nested iteration in parallel, and the operators it is handed.

NLNI.solveMultilevel solves a coarse level, prolongs the result to the next
level through ``meshTransfers.prolong_bcListDict`` and solves again. In
parallel those operators were still the *global* matrices -- the distributed
ones had been installed only for ``prolongList``/``restrictList``, the free
unknowns the linear multigrid solvers use. A 25x9 matrix applied to a 7-entry
coarse vector and a 17-entry fine one wrote past the end of the fine array;
the loop is compiled without bounds checks, so nothing complained until the
heap was damaged. That was reachable with ``nLevels > 1`` whatever the linear
solver was, because NLNI is the multilevel *nonlinear* solver.

The strong-Dirichlet variant, where Dirichlet dofs are eliminated from the
unknowns, is not parallel and cannot be reached: OneLevelTransport asserts weak
Dirichlet conditions for parallel runs, so every dof stays an unknown and the
free-unknown and full-dof operators coincide.
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


def _launch(n_levels, n_ranks, space='p1'):
    mpiexec = _mpiexec()
    if mpiexec is None:
        pytest.skip('no MPI launcher available')
    helper = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'bc_transfers_mpi.py')
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get('PYTHONPATH', '')]).strip(os.pathsep)
    result = subprocess.run([mpiexec, '-n', str(n_ranks), sys.executable, helper,
                             str(n_levels), space],
                            capture_output=True, text=True, env=env, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def _run(n_levels, n_ranks, space='p1'):
    result = _launch(n_levels, n_ranks, space)
    for line in result.stdout.splitlines():
        if line.startswith('RESULT'):
            return float(line.split('checksum=')[1].split()[0])
    raise AssertionError(result.stdout + result.stderr)


@pytest.mark.parametrize('n_levels', [2, 3])
@pytest.mark.parametrize('n_ranks', [2, 3])
def test_nested_iteration_reaches_the_serial_solution(n_levels, n_ranks):
    serial = _run(n_levels, 1)
    parallel = _run(n_levels, n_ranks)
    assert abs(parallel - serial) < 1.0e-5 * abs(serial), (
        "%i levels on %i ranks gave %.12e against %.12e in serial"
        % (n_levels, n_ranks, parallel, serial))


def test_one_level_needs_no_transfers_for_any_space():
    """Quadratic Lagrange on two ranks, one level, must still run.

    Parallel runs used to build a global space per level for the transfer
    operators unconditionally. Quadratic Lagrange's dof map is built from the
    partitioned mesh, so a global space cannot be constructed for it and every
    P2 run on more than one rank died at construction -- including the single
    level ones, which have no transfers at all and ran fine before.
    """
    serial = _run(1, 1, 'p2')
    parallel = _run(1, 2, 'p2')
    assert abs(parallel - serial) < 1.0e-5 * abs(serial), (parallel, serial)


def test_several_levels_of_a_non_nodal_space_now_run():
    """Quadratic dofs are not vertices, so the nodal operators cannot describe
    them; the general route (replicated global operators, cut to owned rows)
    must build distributed ones instead of refusing. Its numerics are pinned in
    test_parallel_space_multigrid.py.
    """
    result = _launch(2, 2, 'p2')
    assert result.returncode == 0, result.stdout + result.stderr
    assert not [l for l in result.stdout.splitlines() if l.startswith('REFUSED')], result.stdout
    assert [l for l in result.stdout.splitlines() if l.startswith('RESULT')], result.stdout + result.stderr


def test_components_are_transferred_independently():
    """Two uncoupled copies of one problem: both must come out as the serial one.

    The *_bcListDict operators are per component and act on each component's
    own dof array, so unlike the stacked free-unknown operators they do not
    care how components are interleaved in the system. The helper also checks
    every operator for every component against the subdomain meshes.
    """
    serial = float(_launch(3, 1, 'p1').stdout.split('checksum=')[1].split()[0])
    result = _launch(3, 2, 'p1x2')
    line = [l for l in result.stdout.splitlines() if l.startswith('RESULT')][0]
    values = [float(v) for v in line.split('checksums=')[1].split(',')]
    assert len(values) == 2, line
    for cj, value in enumerate(values):
        assert abs(value - serial) < 1.0e-5 * abs(serial), (cj, value, serial)
