#!/usr/bin/env python
"""
Test module for NCLS with EV
"""
import os
import proteus
from importlib import reload
from proteus import Profiling, default_s, default_so
from proteus import Comm
comm = Comm.get()
Profiling.logLevel=2
Profiling.verbose=True
import numpy as np
import h5py
import sys
sys.path.append('import_files')
from . import ncls
from . import ncls_p
from . import ncls_n

class TestNCLS(object):

    @classmethod
    def setup_class(cls):
        pass
    
    @classmethod
    def teardown_class(cls):
        pass
    
    def setup_method(self,method):
        """Initialize the test problem. """
        reload(ncls)
        self.pList = [ncls_p]
        self.nList = [ncls_n]        
        self.sList = [default_s]
        self.so = default_so
        self.so.tnList = self.nList[0].tnList
        self._scriptdir = os.path.dirname(__file__)                
        self.sim_names = []
        self.aux_names = []
               
    def teardown_method(self,method):
        pass
    
    def test_pure_advection_supg(self, opts):
        ncls.ct.level_set_function = 0 # distance function
        ncls.ct.STABILIZATION_TYPE = 0 # SUPG
        ncls.ct.COUPEZ = False
        ncls.ct.DO_REDISTANCING = False
        reload(ncls_n)
        reload(ncls_p)
        self.so.name = self.pList[0].name+"_pureAdvection_SUPG"        
        # NUMERICAL SOLUTION #
        ns = proteus.NumericalSolution.NS_base(self.so,
                                               self.pList,
                                               self.nList,
                                               self.sList,
                                               opts)
        self.sim_names.append(ns.modelList[0].name)
        ns.calculateSolution('ncls')
        # COMPARE VS SAVED FILES #
        actual = h5py.File('ncls_level_0_pureAdvection_SUPG.h5','r')
        expected_path = 'comparison_files/' + 'comparison_' + 'ncls_level_0_pureAdvection_SUPG_' + '_u_t2.csv'
        #write comparison file
        #np.array(actual['u_t2']).tofile(os.path.join(self._scriptdir, expected_path),sep=",")
        np.testing.assert_almost_equal(np.fromfile(os.path.join(self._scriptdir, expected_path),sep=","),np.array(actual['u_t2']).flatten(),decimal=10)
        actual.close()
        
    def test_pure_advection_ev1(self, opts):
        ncls.ct.level_set_function = 0 # distance function
        ncls.ct.STABILIZATION_TYPE = 1 # EV1
        ncls.ct.SATURATED_LEVEL_SET = False
        ncls.ct.ENTROPY_TYPE = 1 # quadratic entropy
        ncls.ct.COUPEZ = False
        ncls.ct.DO_REDISTANCING = False
        reload(ncls_n)
        reload(ncls_p)
        self.so.name = self.pList[0].name+"_pureAdvection_EV1"
        # NUMERICAL SOLUTION #
        ns = proteus.NumericalSolution.NS_base(self.so,
                                               self.pList,
                                               self.nList,
                                               self.sList,
                                               opts)
        self.sim_names.append(ns.modelList[0].name)
        ns.calculateSolution('ncls')
        # COMPARE VS SAVED FILES #
        actual = h5py.File('ncls_level_0_pureAdvection_EV1.h5','r')
        expected_path = 'comparison_files/' + 'comparison_' + 'ncls_level_0_pureAdvection_EV1_' + '_u_t2.csv'
        #write comparison file
        #np.array(actual['u_t2']).tofile(os.path.join(self._scriptdir, expected_path),sep=",")
        np.testing.assert_almost_equal(np.fromfile(os.path.join(self._scriptdir, expected_path),sep=","),np.array(actual['u_t2']).flatten(),decimal=10)
        actual.close()

    def test_coupez_with_redistancing_non_saturated(self, opts):
        ncls.ct.level_set_function = 0 # distance function
        ncls.ct.STABILIZATION_TYPE = 1 # EV1
        ncls.ct.SATURATED_LEVEL_SET = False
        ncls.ct.ENTROPY_TYPE = 1
        ncls.ct.COUPEZ = True
        ncls.ct.DO_REDISTANCING = True
        reload(ncls_n)
        reload(ncls_p)
        self.so.name = self.pList[0].name+"_non_saturated_ls"
        # NUMERICAL SOLUTION #
        ns = proteus.NumericalSolution.NS_base(self.so,
                                               self.pList,
                                               self.nList,
                                               self.sList,
                                               opts)
        self.sim_names.append(ns.modelList[0].name)
        ns.calculateSolution('ncls')
        # COMPARE VS SAVED FILES #
        actual = h5py.File('ncls_level_0_non_saturated_ls.h5','r')
        expected_path = 'comparison_files/' + 'comparison_' + 'ncls_level_0_non_saturated_ls_' + '_u_t2.csv'
        #write comparison file
        #np.array(actual['u_t2']).tofile(os.path.join(self._scriptdir, expected_path),sep=",")
        np.testing.assert_almost_equal(np.fromfile(os.path.join(self._scriptdir, expected_path),sep=","),np.array(actual['u_t2']).flatten(),decimal=10)

        actual.close()
        
    def test_coupez_with_redistancing_saturated(self, opts):
        ncls.ct.level_set_function = 1 # saturated distance function
        ncls.ct.STABILIZATION_TYPE = 1 # EV2
        ncls.ct.SATURATED_LEVEL_SET = True
        ncls.ct.ENTROPY_TYPE = 2
        ncls.ct.COUPEZ = True
        ncls.ct.DO_REDISTANCING = True
        reload(ncls_n)
        reload(ncls_p)
        self.so.name = self.pList[0].name+"_saturated_ls"
        # NUMERICAL SOLUTION #
        ns = proteus.NumericalSolution.NS_base(self.so,
                                               self.pList,
                                               self.nList,
                                               self.sList,
                                               opts)
        self.sim_names.append(ns.modelList[0].name)
        ns.calculateSolution('ncls')                
        # COMPARE VS SAVED FILES #
        actual = h5py.File('ncls_level_0_saturated_ls.h5','r')

        expected_path = 'comparison_files/' + 'comparison_' + 'ncls_level_0_saturated_ls_' + '_u_t2.csv'
        #write comparison file
        #np.array(actual['u_t2']).tofile(os.path.join(self._scriptdir, expected_path),sep=",")
        np.testing.assert_almost_equal(np.fromfile(os.path.join(self._scriptdir, expected_path),sep=","),np.array(actual['u_t2']).flatten(),decimal=10)

        actual.close()
        
