"""LU.prepare does not hand a matrix with a NaN or an Inf to SuperLU: the solve is reported as failed.

SuperLU does not look at its input: a NaN matrix gave a NaN pivot and a SIGSEGV in dgstrf (dpivotL) in the
Richards implicit-FCT levee run, which ended the process without a traceback.
"""
import numpy as np
import pytest

from proteus import LinearSolvers, superluWrappers


def _tridiagonal(n=6):
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 4.0
        if i:
            A[i, i - 1] = -1.0
        if i < n - 1:
            A[i, i + 1] = -1.0
    return A


def _sparse(dense, poison=None):
    rowptr, colind, nzval = [0], [], []
    for row in dense:
        for j in np.flatnonzero(row):
            colind.append(j)
            nzval.append(row[j])
        rowptr.append(len(colind))
    nzval = np.array(nzval, 'd')
    if poison is not None:
        index, value = poison
        nzval[index] = value
    n = dense.shape[0]
    return superluWrappers.SparseMatrix(n, n, len(nzval), nzval,
                                        np.array(colind, 'i'), np.array(rowptr, 'i'))


def test_a_finite_sparse_matrix_is_factored_and_solved():
    A = _tridiagonal()
    lu = LinearSolvers.LU(_sparse(A))
    lu.prepare()
    b = np.arange(1.0, 7.0)
    u = b.copy()
    superluWrappers.sparseFactorSolve(lu.sparseFactor, u)
    np.testing.assert_allclose(A @ u, b, atol=1.0e-12)


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_a_sparse_matrix_with_a_non_finite_entry_is_not_factored_and_the_solve_fails(value, monkeypatch):
    called = []
    monkeypatch.setattr(superluWrappers, 'sparseFactorPrepare', lambda *a: called.append(a))
    messages = []
    monkeypatch.setattr(LinearSolvers, 'logEvent', lambda message, level=0: messages.append(message))
    lu = LinearSolvers.LU(_sparse(_tridiagonal(), poison=(7, value)))
    lu.prepare()
    assert not called                      # SuperLU never saw the matrix
    assert lu.failed() is True             # what Newton reads as a failed linear solve
    assert len(messages) == 1
    assert 'LU.prepare' in messages[0] and '6 x 6' in messages[0]
    assert '1 of 16 entries' in messages[0] and 'entry 7' in messages[0]
    u = np.zeros(6)
    lu.solve(u, b=np.ones(6))
    assert np.isnan(u).all()               # a caller that ignores failed() gets NaN, not a stale answer


def test_the_next_finite_matrix_is_factored_again():
    A = _sparse(_tridiagonal())
    lu = LinearSolvers.LU(A)
    A.nzvals[0] = np.nan
    lu.prepare()
    assert lu.failed() is True
    A.nzvals[0] = 4.0
    lu.prepare()
    b = np.arange(1.0, 7.0)
    u = b.copy()
    superluWrappers.sparseFactorSolve(lu.sparseFactor, u)
    np.testing.assert_allclose(_tridiagonal() @ u, b, atol=1.0e-12)


def test_every_entry_nan_is_reported_as_in_the_levee_run():
    A = _sparse(_tridiagonal())
    A.nzvals[:] = np.nan
    assert '16 of 16 entries' in LinearSolvers.nonFiniteMatrixMessage(A.nzvals, 'x')


def test_plus_and_minus_inf_do_not_cancel_in_the_sum():
    A = _sparse(_tridiagonal())
    A.nzvals[0], A.nzvals[1] = np.inf, -np.inf
    assert '2 of 16 entries' in LinearSolvers.nonFiniteMatrixMessage(A.nzvals, 'x')


def test_large_but_finite_entries_are_not_refused():
    A = _sparse(_tridiagonal())
    A.nzvals[:] *= 1.0e308 / 4.0    # the sum overflows, no entry does
    assert LinearSolvers.nonFiniteMatrixMessage(A.nzvals, 'overflowing sum') is None


def test_a_dense_matrix_with_a_nan_is_not_factored():
    A = _tridiagonal()
    A[2, 3] = np.nan
    lu = LinearSolvers.LU(A)
    lu.prepare()
    assert lu.failed() is True
    assert 'entry (2, 3)' in LinearSolvers.nonFiniteMatrixMessage(A, 'dense', A.shape)
