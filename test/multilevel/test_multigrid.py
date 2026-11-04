#!/usr/bin/env python
"""Regression tests for proteus's own multilevel linear solvers.

MGM (geometric multigrid), NI (nested iteration) and the native smoothers in
``proteus.csmoothers`` had no test coverage at all, and every one of them was
broken:

* ``csmoothers`` read ``nr``/``nc`` off ``superluWrappers.cSparseMatrix``, which
  has stored its dimensions in ``dim[2]`` since 2019 (PR #881) -- so all four
  smoothers raised ``AttributeError`` on first use, both as MG smoothers and as
  the PETSc shell preconditioners ``KSP_petsc4py`` builds from the same classes.
* ``gauss_seidel_NR_preare`` was a misspelled Cython export, so
  ``GaussSeidel.prepare`` could never resolve it.
* ``Newton`` calls ``linearSolver.prepare(b=..., newton_its=...)`` but only
  ``LU`` and ``KSP_petsc4py`` ever grew the ``newton_its`` argument.
* ``MultilevelLinearSolver``/``NI`` had no ``failed()``, which ``Newton`` calls.
* A misspelled local (``printLevelLinearSolverInfo``) made ``NI`` + a
  ``KSP_petsc4py`` level solver raise ``UnboundLocalError`` in the chooser.

These tests pin the behaviour so the whole layer cannot silently rot again.
"""
import inspect
import os

import numpy as np
import pytest

from proteus.iproteus import opts
from proteus import (LinearSolvers, NumericalSolution, defaults, default_n,
                     default_s, default_so)

modulepath = os.path.dirname(os.path.abspath(__file__))
poissondir = os.path.join(os.path.dirname(modulepath), 'poisson_2d')

SMOOTHERS = {
    'Jacobi': LinearSolvers.Jacobi,
    'GaussSeidel': LinearSolvers.GaussSeidel,
    'StarILU': LinearSolvers.StarILU,
}


def _solve(multilevel, level, smoother, nLevels, tag, return_solution=True):
    """Solve the 2D heterogeneous Poisson problem.

    Returns ``(dofs, analytical)``, or the NumericalSolution itself when
    ``return_solution`` is False and a test needs the solver objects.
    """
    p = defaults.load_physics('poisson_het_2d_p', poissondir)
    n = defaults.load_numerics('poisson_het_2d_c0pk_n', poissondir)
    p.meshfile = os.path.join(poissondir, 'square4x4')
    p.x0 = (0.0, 0.0, 0.0)
    p.L = (1.0, 1.0, 1.0)
    n.nLevels = nLevels
    n.femSpaces[0] = default_n.C0_AffineLinearOnSimplexWithNodalBasis
    n.numericalFluxType = default_n.Advection_DiagonalUpwind_Diffusion_SIPG_exterior
    n.multilevelLinearSolver = multilevel
    n.levelLinearSolver = level
    n.linearSmoother = smoother
    so = default_so
    so.name = p.name = tag
    so.sList = [default_s]
    opts.logLevel = 1
    opts.verbose = False
    ns = NumericalSolution.NS_base(so, [p], [n], so.sList, opts)
    ns.calculateSolution(tag)
    if not return_solution:
        return ns
    fine = ns.modelList[0].levelModelList[-1]
    return np.array(fine.u[0].dof, copy=True), np.array(fine.u_analytical[0][0], copy=True)


@pytest.fixture(scope='module')
def direct_solution():
    """Reference solve of the same discrete problem with a direct solver."""
    return _solve(LinearSolvers.LU, LinearSolvers.LU, None, 3, 'mg_reference_lu')


@pytest.mark.parametrize('name', sorted(SMOOTHERS))
def test_mgm_matches_direct_solve(name, direct_solution):
    """MGM must converge to the same discrete solution a direct solver finds.

    Running without raising is not enough -- before the csmoothers fix every
    smoother raised, and a silently non-convergent MG cycle would still return.
    """
    reference, exact = direct_solution
    u, _ = _solve(LinearSolvers.MGM, LinearSolvers.MGM, SMOOTHERS[name], 3,
                  'mgm_' + name)

    discretization_error = np.abs(reference - exact).max()
    solver_error = np.abs(u - reference).max()

    # The MG solve must agree with the direct solve to well inside the
    # discretization error, otherwise it is not solving the same problem.
    assert solver_error < 0.1 * discretization_error, (
        "MGM/%s differs from the direct solve by %.3e, which is not small "
        "compared with the discretization error %.3e"
        % (name, solver_error, discretization_error))


def test_nested_iteration_with_mgm_levels(direct_solution):
    """NI driving MGM level solvers converges to the direct solution."""
    reference, exact = direct_solution
    u, _ = _solve(LinearSolvers.NI, LinearSolvers.MGM, LinearSolvers.Jacobi, 3,
                  'ni_mgm')
    assert np.abs(u - reference).max() < 0.1 * np.abs(reference - exact).max()


def test_mgm_without_smoother_is_rejected():
    """A missing smoother must fail loudly in the chooser.

    It used to be logged and ignored, leaving preSmootherList short so that
    MGM.__init__ died with an IndexError that named neither the option nor the
    cause.
    """
    with pytest.raises(ValueError, match='MGM requires a smoother'):
        _solve(LinearSolvers.MGM, LinearSolvers.MGM, None, 2, 'mgm_no_smoother')


def test_every_linear_solver_prepare_accepts_newton_its():
    """Newton calls prepare(b=..., newton_its=...) on whatever it is handed.

    Only LU and KSP_petsc4py had been updated, so every other solver -- the
    smoothers, TwoLevel, MGM and NI -- raised TypeError the first time Newton
    drove them.
    """
    missing = []
    for name in dir(LinearSolvers):
        cls = getattr(LinearSolvers, name)
        if not inspect.isclass(cls) or 'prepare' not in vars(cls):
            continue
        params = inspect.signature(cls.prepare).parameters
        if 'newton_its' not in params and not any(
                pr.kind is inspect.Parameter.VAR_KEYWORD for pr in params.values()):
            missing.append(name)
    assert not missing, (
        "these solvers' prepare() cannot be called by Newton: %r" % (missing,))


@pytest.mark.parametrize('cls', [LinearSolvers.MultilevelLinearSolver,
                                 LinearSolvers.MGM,
                                 LinearSolvers.NI])
def test_multilevel_solvers_expose_failed(cls):
    """Newton calls failed() on its linear solver right after prepare()."""
    assert callable(getattr(cls, 'failed', None))


def test_nested_iteration_with_ksp_level_solvers(direct_solution):
    """NI must be able to drive PETSc level solvers.

    Every KSP_petsc4py solve needs a parallel vector, including the ones NI
    runs on its coarser levels.  NI's coarse-level work vectors are its own,
    not the model's, so they had no ParVec and NI raised AttributeError on
    par_b the moment a KSP level solver was used -- in serial as well as in
    parallel, because KSP_petsc4py always takes the parallel path.
    """
    reference, exact = direct_solution
    u, _ = _solve(LinearSolvers.NI, LinearSolvers.KSP_petsc4py, None, 3, 'ni_ksp')
    assert np.abs(u - reference).max() < 0.1 * np.abs(reference - exact).max()


def test_ksp_info_returns_a_string():
    """info() is concatenated into an infoString by every multilevel solver.

    KSP_petsc4py.info() used to call ksp.view(), which returns None and dumps
    the whole PETSc object to stdout, so NI raised TypeError as soon as it
    logged a KSP level solver.
    """
    ns = _solve(LinearSolvers.NI, LinearSolvers.KSP_petsc4py, None, 2,
                'ksp_info', return_solution=False)
    level_solver = ns.lsList[0].levelSolverList[-1]
    assert isinstance(level_solver, LinearSolvers.KSP_petsc4py)
    assert isinstance(level_solver.info(), str)


@pytest.mark.parametrize('n_ranks', [2])
def test_parvec_duplicate_keeps_ghost_layout_under_mpi(n_ranks):
    """duplicate_with_array must produce a genuinely ghosted vector.

    Serial runs cannot catch a broken ghost layout, because serial ParVecs are
    not ghosted at all, so this one goes through mpiexec.
    """
    import os
    import shutil
    import subprocess
    import sys

    # Prefer the launcher that ships with this interpreter's environment: a
    # different MPI on PATH (e.g. Homebrew's) cannot launch a petsc4py built
    # against the environment's own MPI, and silently gives every rank size 1.
    bindir = os.path.dirname(sys.executable)
    mpiexec = None
    for name in ('mpiexec', 'mpirun'):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            mpiexec = candidate
            break
    if mpiexec is None:
        mpiexec = shutil.which('mpiexec') or shutil.which('mpirun')
    if mpiexec is None:
        pytest.skip('no MPI launcher available')

    helper = os.path.join(modulepath, 'ghost_duplicate_mpi.py')
    env = dict(os.environ)
    # Make sure the subprocess imports the same proteus this test did.
    env['PYTHONPATH'] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get('PYTHONPATH', '')]).strip(os.pathsep)

    result = subprocess.run([mpiexec, '-n', str(n_ranks), sys.executable, helper],
                            capture_output=True, text=True, env=env, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count('GHOST_DUPLICATE_OK') == n_ranks, (
        result.stdout + result.stderr)


def test_set_unknowns_rejects_a_short_vector():
    """setUnknowns must not be handed a vector shorter than the level needs.

    cfemIntegrals.copyBetweenFreeUnknownsAndGlobalUnknowns indexes free_u from
    C with no bounds checking, so a short vector corrupts adjacent memory --
    typically the dof arrays' own headers, surfacing much later as a
    nonsensical numpy shape rather than an error at the call.  Three call sites
    in NumericalSolution passed uList[0] for *every* level model, which is
    harmless only when nLevels == 1.
    """
    import numpy as np

    ns = _solve(LinearSolvers.LU, LinearSolvers.LU, None, 2, 'short_vec',
                return_solution=False)
    fine = ns.modelList[0].levelModelList[-1]
    coarse_sized = np.zeros(ns.modelList[0].levelModelList[0].dim, 'd')
    assert coarse_sized.size < fine.dim, "levels are not actually different sizes"

    with pytest.raises(ValueError, match='free unknowns'):
        fine.setUnknowns(coarse_sized)


def test_numerical_solution_saves_dofs_per_level():
    """The save_dof blocks must index uList by level, not take uList[0]."""
    import inspect

    from proteus import NumericalSolution as _ns
    source = inspect.getsource(_ns.NS_base.calculateSolution)
    assert 'setUnknowns(m.uList[0])' not in source, (
        "a save_dof block still passes the coarsest level's vector to every "
        "level model")
