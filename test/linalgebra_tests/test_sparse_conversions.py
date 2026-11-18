#!/usr/bin/env python
"""
Test module for some linear algebra tools functions.
"""
from proteus import superluWrappers
from proteus.csparsity import CSR_INDEX_DTYPE
from proteus import LinearAlgebraTools
from petsc4py import PETSc as p4pyPETSc
import numpy
import os, sys
import pytest

@pytest.fixture(scope="module")
def small_superlu():
    vals    = numpy.array([10.,-2.,3.,9.,3.,7.,8.,7.,3.,
                           8.,7.,5.,8.,9.,9.,13.,4.,2.,-1.])
    col_idx = numpy.array([0, 4 ,0 ,1 ,5 ,1 ,2 ,3 ,0 ,2 ,
                           3 ,4 ,1 ,3 ,4 ,5  ,1 ,4 , 5 ],
                          dtype=CSR_INDEX_DTYPE)
    row_idx = numpy.array([0, 2, 5, 8, 12, 16, 19],
                          dtype=CSR_INDEX_DTYPE)
    size_n  = len(row_idx)-1

    superlu_mat = LinearAlgebraTools.SparseMat(size_n,
                                               size_n,
                                               len(vals),
                                               vals,
                                               col_idx,
                                               row_idx)
    yield superlu_mat

@pytest.fixture(scope="module")
def small_petsc4py():
    vals    = [10.,-2.,3.,9.,3.,7.,8.,7.,3.,8.,7.,5.,8.,9.,9.,13.,4.,2.,-1.]
    col_idx = [0  , 4 ,0 ,1 ,5 ,1 ,2 ,3 ,0 ,2 ,3 ,4 ,1 ,3 ,4 ,5  ,1 ,4 , 5 ]
    row_idx = [0, 2, 5, 8, 12, 16, 19]
    size_n  = len(row_idx)-1
    petsc_mat = p4pyPETSc.Mat().createAIJ(size = (size_n,size_n), \
                                          csr  = (row_idx, col_idx, vals))
    yield petsc_mat

@pytest.fixture(scope="module")
def med_petsc_with_const_pressure():
    petsc_mat = LinearAlgebraTools.petsc_load_matrix(os.path.join(os.path.dirname(__file__),
                                                                  'jac.bin'))
    yield petsc_mat

@pytest.mark.LinearAlgebraTools
def test_superlu_2_dense(small_superlu):
    """ Tests the superlu wrapper to dense matrix function. """
    superlu_mat = small_superlu

    dense_mat = LinearAlgebraTools.superlu_sparse_2_dense(superlu_mat)
    comparison_mat = numpy.loadtxt(os.path.join(os.path.dirname(__file__),
                                                'sparse_mat_1.txt'))
    assert numpy.allclose(dense_mat,comparison_mat)

@pytest.mark.LinearAlgebraTools
def test_superlu_get_rank_1(small_superlu):
    superlu_mat = small_superlu

    rank = LinearAlgebraTools.superlu_get_rank(superlu_mat)
    assert rank == 6

@pytest.mark.LinearAlgebraTools
def test_superlu_has_pressure_null_space(small_superlu):
    superlu_mat = small_superlu

    has_null = LinearAlgebraTools.superlu_has_pressure_null_space(superlu_mat)
    assert has_null == False

@pytest.mark.LinearAlgebraTools
def test_sparse_2_dense(small_petsc4py):
    '''
    This function tests the petsc4py_sparse_2_dense function in the
    LinearAlgebraTools module.
    '''
    petsc_mat = small_petsc4py
    dense_mat = LinearAlgebraTools.petsc4py_sparse_2_dense \
                (petsc_mat)
    comparison_mat = numpy.loadtxt(os.path.join(os.path.dirname(__file__),
                                                'sparse_mat_1.txt'))
    assert numpy.allclose(dense_mat,comparison_mat)

@pytest.mark.LinearAlgebraTools
def test_superlu_2_petsc(small_superlu):
    """ Tests the function superlu_2_petsc4py """
    superlu_mat = small_superlu
    petsc4py_mat = LinearAlgebraTools.superlu_2_petsc4py(superlu_mat)

    dense_mat = LinearAlgebraTools.petsc4py_sparse_2_dense(petsc4py_mat)
    comparison_mat = numpy.loadtxt(os.path.join(os.path.dirname(__file__),
                                                'sparse_mat_1.txt'))
    assert numpy.allclose(dense_mat,comparison_mat)

@pytest.mark.LinearAlgebraTools
def test_petsc4py_get_rank_1(small_petsc4py):
    matr = small_petsc4py

    rank = LinearAlgebraTools.petsc4py_get_rank(small_petsc4py)
    assert rank == 6

@pytest.mark.LinearAlgebraTools
def test_petsc4py_get_rank_2(med_petsc_with_const_pressure):
    matr = med_petsc_with_const_pressure

    rank = LinearAlgebraTools.petsc4py_get_rank(matr)
    assert rank == 44

@pytest.mark.LinearAlgebraTools
def test_petsc4py_has_pressure_null_space_1(small_petsc4py):
    superlu_mat = small_petsc4py

    has_null = LinearAlgebraTools.petsc4py_mat_has_pressure_null_space(superlu_mat)
    assert has_null == False

@pytest.mark.LinearAlgebraTools
def test_petsc4py_has_pressure_null_space_2(med_petsc_with_const_pressure):
    superlu_mat = med_petsc_with_const_pressure

    has_null = LinearAlgebraTools.petsc4py_mat_has_pressure_null_space(superlu_mat)
    assert has_null == True

if __name__ == '__main__':
    test_sparse_2_dense()
    test_superlu_2_dense()
    test_superlu_2_petsc()