#!/usr/bin/env python
"""
Test module for 2D Quadrilateral Meshes
"""
import proteus.test_utils.TestTools
from proteus import LinearAlgebraTools
from proteus import Comm

import os
import sys
import inspect
import numpy
import pytest

proteus.test_utils.TestTools.addSubFolders( inspect.currentframe() )
import singlephase_mass_matrix_THQuad_2D_4 as sp_mm_2d_THQuad

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
        """ Tests mass matrix construction for TH elements. """
        self.mass_matrix_object = sp_mm_2d_THQuad.ns
        self.mass_matrix_object.modelList[0].levelModelList[0].calculateCoefficients()
        rowptr, colind, nzval = self.mass_matrix_object.modelList[0].levelModelList[0].jacobian.getCSRrepresentation()
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
        rel_path = "comparison_files/single_phase_THQuad_4_expected.data"
        comparison_mat = numpy.load(os.path.join(self.scriptdir,rel_path))
        assert numpy.allclose(mass_mat,comparison_mat)

if __name__ == '__main__':
    pass