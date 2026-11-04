#!/usr/bin/env python
"""PETSc's multigrid as a cross-check on proteus's.

petsc_MG hands PCMG what MGM uses -- the transfer operators from the multilevel
mesh and one re-discretized matrix per level -- and lets PETSc run the cycle.
Configured to be the same algorithm (one V-cycle, three sweeps of Jacobi damped
by 0.8 either side, an exact coarse solve) the two must agree to roundoff, and
that is a check on everything they share: the transfer operators, the level
matrices, the vector layouts, the ghost handling. The cycle itself is
implemented twice independently, so a difference is a defect in what we feed
PETSc rather than in either multigrid.

PETSc objects exist only on more than one rank (Transport builds the parallel
matrices and vectors only then), so the comparison is a parallel PCMG against
the serial MGM.

Three things this turned up on the way, each of which would have been taken for
a multigrid problem:

* the numerics modules set ``-pc_type asm`` in the global options database, and
  KSP.setFromOptions() honors it over the preconditioner a solver was given, so
  an apparently-installed PCMG was really ASM;
* with a single level PCMG has no coarse solver -- level 0 is configured as a
  smoother under ``mg_levels_0_`` -- so the level-0 solve of a nested iteration
  was three damped Jacobi sweeps instead of an exact solve; and
* PETSc's option lookup falls back from a numbered prefix to the unnumbered
  one, so a hasName test for "did the user set mg_levels_0_..." is true
  whenever mg_levels_ is.
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


def _run(n_levels, mode, n_ranks, dim=2, components=1, all_components=False):
    mpiexec = _mpiexec()
    if mpiexec is None:
        pytest.skip('no MPI launcher available')
    helper = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'pcmg_parallel_mpi.py')
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get('PYTHONPATH', '')]).strip(os.pathsep)
    result = subprocess.run([mpiexec, '-n', str(n_ranks), sys.executable, helper,
                             str(n_levels), mode, str(dim), str(components)],
                            capture_output=True, text=True, env=env, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    for line in result.stdout.splitlines():
        if line.startswith('RESULT'):
            if all_components:
                return [float(v) for v in line.split('checksums=')[1].split(',')]
            return float(line.split('checksum=')[1].split()[0])
    raise AssertionError(result.stdout + result.stderr)


@pytest.mark.parametrize('dim,n_levels', [(2, 1), (2, 2), (2, 3), (2, 4), (3, 2), (3, 3)])
@pytest.mark.parametrize('n_ranks', [2, 3])
def test_pcmg_cycle_is_the_mgm_cycle(dim, n_levels, n_ranks):
    mgm = _run(n_levels, 'mgm', 1, dim)
    pcmg = _run(n_levels, 'pcmg', n_ranks, dim)
    assert abs(pcmg - mgm) < 1.0e-11 * abs(mgm), (
        "PCMG on %i ranks gave %.14e against serial MGM's %.14e (%iD, %i levels): "
        "the same V-cycle over the same hierarchy must agree"
        % (n_ranks, pcmg, mgm, dim, n_levels))


@pytest.mark.parametrize('dim', [2, 3])
def test_pcmg_preconditioned_gmres_reaches_the_direct_solution(dim):
    direct = _run(3, 'direct', 1, dim)
    converged = _run(3, 'pcmg_its', 3, dim)
    assert abs(converged - direct) < 1.0e-9 * abs(direct), (converged, direct)


@pytest.mark.parametrize('components', [2, 3])
def test_pcmg_over_a_stacked_system_matches_the_single_problem(components):
    """PCMG's interpolation for a multi-component system is P (x) I, in the
    interleaved layout the system's matrix and vectors use. Each component of an
    uncoupled system must come out as the single-component V-cycle."""
    single = _run(3, 'mgm', 1)
    values = _run(3, 'pcmg', 3, components=components, all_components=True)
    assert len(values) == components, values
    for c, value in enumerate(values):
        assert abs(value - single) < 1.0e-11 * abs(single), (c, value, single)
