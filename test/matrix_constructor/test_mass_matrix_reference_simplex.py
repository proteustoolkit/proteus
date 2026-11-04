#!/usr/bin/env python
"""
Test module for 2D Quadrilateral Meshes
"""
import proteus.test_utils.TestTools
from proteus import LinearAlgebraTools, LinearSolvers
from proteus import Comm

import os
import sys
import inspect
import numpy
import pytest

proteus.test_utils.TestTools.addSubFolders( inspect.currentframe() )
import mass_matrix_reference_C0P1_2D as mm_2d_C0P1
import mass_matrix_reference_TH_2D as mm_2d_TH

@pytest.mark.LinearSolvers
class TestMassConstruction2D(proteus.test_utils.TestTools.SimulationTest):
    """ Verify construction of 2D Mass Matrix using transport coefficients """

    def setup_method(self):
        """ Initialize the test problem """
        self._setRelativePath()
        
    def teardown_method(self):
        """ Tear down function """
        FileList = ['Mass_matrix_test.xmf', 'Mass_matrix_test.ymf',
                    'Mass_matrix_test.h5',
                    'reference_triangle_2d.ele',
                    'reference_triangle_2d.node',
                    'reference_triangle_2d.poly']
        self.remove_files(FileList)

    @classmethod
    def _setRelativePath(self):
        self.scriptdir = os.path.dirname(__file__)

    def test_1(self):
        """ An initial test of the coefficient class. """
        self.mass_matrix_object = mm_2d_C0P1.ns
        self.mass_matrix_object.modelList[0].levelModelList[0].calculateCoefficients()
        rowptr, colind, nzval = self.mass_matrix_object.modelList[0].levelModelList[0].jacobian.getCSRrepresentation()
        self.mass_matrix_object.modelList[0].levelModelList[0].scale_dt = False
        self.Asys_rowptr = rowptr.copy()
        self.Asys_colptr = colind.copy()
        self.Asys_nzval = nzval.copy()
        nn = len(self.Asys_rowptr)-1
        self.Asys = LinearAlgebraTools.SparseMatrix(nn,nn,
                                                    self.Asys_nzval.shape[0],
                                                    self.Asys_nzval,
                                                    self.Asys_colptr,
                                                    self.Asys_rowptr)
        self.petsc4py_A = self.mass_matrix_object.modelList[0].levelModelList[0].getMassJacobian(self.Asys)
        mass_mat = LinearAlgebraTools.superlu_sparse_2_dense(self.petsc4py_A)
        rel_path = "comparison_files/mass_reference_c0p1_2D.txt"
        comparison_mat = numpy.loadtxt(os.path.join(self.scriptdir,rel_path))
        assert numpy.allclose(mass_mat,comparison_mat)

    def test_2(self):
        """ Tests the attachMassOperator function in one-level-transport """
        self.mass_matrix_object = mm_2d_C0P1.ns
        mm = self.mass_matrix_object.modelList[0].levelModelList[0]
        op_constructor = LinearSolvers.OperatorConstructor_oneLevel(mm)
        op_constructor.attachMassOperator()
        op_constructor.updateMassOperator()
        mass_mat = LinearAlgebraTools.superlu_sparse_2_dense(op_constructor.MassOperator)
        rel_path = "comparison_files/mass_reference_c0p1_2D.txt"
        comparison_mat = numpy.loadtxt(os.path.join(self.scriptdir,rel_path))
        assert numpy.allclose(mass_mat,comparison_mat)

    def test_3(self):
        """ Tests mass matrix construction for TH elements. """
        self.mass_matrix_object = mm_2d_TH.ns
        self.mass_matrix_object.modelList[0].levelModelList[0].calculateCoefficients()
        rowptr, colind, nzval = self.mass_matrix_object.modelList[0].levelModelList[0].jacobian.getCSRrepresentation()
        self.mass_matrix_object.modelList[0].levelModelList[0].scale_dt = False
        self.Asys_rowptr = rowptr.copy()
        self.Asys_colptr = colind.copy()
        self.Asys_nzval = nzval.copy()
        nn = len(self.Asys_rowptr)-1
        self.Asys = LinearAlgebraTools.SparseMatrix(nn,nn,
                                                    self.Asys_nzval.shape[0],
                                                    self.Asys_nzval,
                                                    self.Asys_colptr,
                                                    self.Asys_rowptr)
        self.petsc4py_A = self.mass_matrix_object.modelList[0].levelModelList[0].getMassJacobian(self.Asys)
        mass_mat = LinearAlgebraTools.superlu_sparse_2_dense(self.petsc4py_A)
        rel_path = "comparison_files/mass_reference_TH_2D.npy"
        comparison_mat = numpy.load(os.path.join(self.scriptdir,rel_path))
        assert numpy.allclose(mass_mat,comparison_mat)

if __name__ == '__main__':
    pass