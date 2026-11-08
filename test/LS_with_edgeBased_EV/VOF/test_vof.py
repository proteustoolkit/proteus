#!/usr/bin/env python
"""
Test module for VOF with EV
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
from . import thelper_vof
from . import thelper_vof_p
from . import thelper_vof_n

class TestVOF(object):

    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def setup_method(self,method):
        """Initialize the test problem. """
        reload(thelper_vof)
        self.pList = [thelper_vof_p]
        self.nList = [thelper_vof_n]
        self.sList = [default_s]
        self.so = default_so
        self.so.tnList = self.nList[0].tnList
        self._scriptdir = os.path.dirname(__file__)
        self.sim_names = []
        self.aux_names = []

    def teardown_method(self,method):
        pass

    def test_supg(self, opts):
        ########
        # SUPG #
        ########
        thelper_vof.ct.STABILIZATION_TYPE = 0 # SUPG
        thelper_vof.ct.FCT = False
        reload(thelper_vof_p)
        reload(thelper_vof_n)
        self.so.name = self.pList[0].name+"_SUPG"
        # NUMERICAL SOLUTION #
        ns = proteus.NumericalSolution.NS_base(self.so,
                                               self.pList,
                                               self.nList,
                                               self.sList,
                                               opts)
        self.sim_names.append(ns.modelList[0].name)
        ns.calculateSolution('vof')
        # COMPARE VS SAVED FILES #
        actual = h5py.File('vof_level_0_SUPG.h5','r')
        expected_path = 'comparison_files/' + 'comparison_' + 'vof_level_0_SUPG_' + '_u_t2.csv'
        #write comparison file
        #np.array(actual['u_t2']).tofile(os.path.join(self._scriptdir, expected_path),sep=",")
        np.testing.assert_almost_equal(np.fromfile(os.path.join(self._scriptdir, expected_path),sep=","),np.array(actual['u_t2']).flatten(),decimal=10)

        actual.close()

    def test_TaylorGalerkin(self, opts):
        ##################
        # TaylorGalerkin #
        ##################
        thelper_vof.ct.STABILIZATION_TYPE = 1 # Taylor Galerkin
        thelper_vof.ct.FCT = False
        reload(thelper_vof_p)
        reload(thelper_vof_n)
        self.so.name = self.pList[0].name+"_TaylorGalerkin"
        # NUMERICAL SOLUTION #
        ns = proteus.NumericalSolution.NS_base(self.so,
                                               self.pList,
                                               self.nList,
                                               self.sList,
                                               opts)
        self.sim_names.append(ns.modelList[0].name)
        ns.calculateSolution('vof')
        # COMPARE VS SAVED FILES #
        actual = h5py.File('vof_level_0_TaylorGalerkin.h5','r')
        expected_path = 'comparison_files/' + 'comparison_' + 'vof_level_0_TaylorGalerkin_' + '_u_t2.csv'
        #write comparison file
        #np.array(actual['u_t2']).tofile(os.path.join(self._scriptdir, expected_path),sep=",")
        np.testing.assert_almost_equal(np.fromfile(os.path.join(self._scriptdir, expected_path),sep=","),np.array(actual['u_t2']).flatten(),decimal=10)



        actual.close()

    def test_EV1(self, opts):
        #######################
        # ENTROPY VISCOSITY 1 # Polynomial entropy
        #######################
        thelper_vof.ct.STABILIZATION_TYPE = 2 # EV
        thelper_vof.ct.ENTROPY_TYPE = 1 #polynomial
        thelper_vof.ct.cE = 1.0
        thelper_vof.ct.FCT = True
        reload(thelper_vof_p)
        reload(thelper_vof_n)
        self.so.name = self.pList[0].name+"_EV1"
        # NUMERICAL SOLUTION #
        ns = proteus.NumericalSolution.NS_base(self.so,
                                               self.pList,
                                               self.nList,
                                               self.sList,
                                               opts)
        self.sim_names.append(ns.modelList[0].name)
        ns.calculateSolution('vof')
        # COMPARE VS SAVED FILES #
        actual = h5py.File('vof_level_0_EV1.h5','r')
        expected_path = 'comparison_files/' + 'comparison_' + 'vof_level_0_EV1_' + '_u_t2.csv'
        #write comparison file
        #np.array(actual['u_t2']).tofile(os.path.join(self._scriptdir, expected_path),sep=",")
        np.testing.assert_almost_equal(np.fromfile(os.path.join(self._scriptdir, expected_path),sep=","),np.array(actual['u_t2']).flatten(),decimal=1)

        actual.close()

    def test_EV2(self, opts):
        thelper_vof.ct.STABILIZATION_TYPE = 2 # EV
        thelper_vof.ct.ENTROPY_TYPE = 1 #logarithmic
        thelper_vof.ct.cE = 0.1
        thelper_vof.ct.FCT = True
        reload(thelper_vof_p)
        reload(thelper_vof_n)
        self.so.name = self.pList[0].name+"_EV2"
        # NUMERICAL SOLUTION #
        ns = proteus.NumericalSolution.NS_base(self.so,
                                               self.pList,
                                               self.nList,
                                               self.sList,
                                               opts)
        self.sim_names.append(ns.modelList[0].name)
        ns.calculateSolution('vof')
        # COMPARE VS SAVED FILES #
        actual = h5py.File('vof_level_0_EV2.h5','r')
        expected_path = 'comparison_files/' + 'comparison_' + 'vof_level_0_EV2_' + '_u_t2.csv'
        #write comparison file
        #np.array(actual['u_t2']).tofile(os.path.join(self._scriptdir, expected_path),sep=",")
        np.testing.assert_almost_equal(np.fromfile(os.path.join(self._scriptdir, expected_path),sep=","),np.array(actual['u_t2']).flatten(),decimal=2)

        actual.close()

    def test_SmoothnessBased(self, opts):
        thelper_vof.ct.STABILIZATION_TYPE = 3 # Smoothness based
        thelper_vof.ct.FCT = True
        reload(thelper_vof_p)
        reload(thelper_vof_n)
        self.so.name = self.pList[0].name+"_SmoothnessBased"
        # NUMERICAL SOLUTION #
        ns = proteus.NumericalSolution.NS_base(self.so,
                                               self.pList,
                                               self.nList,
                                               self.sList,
                                               opts)
        self.sim_names.append(ns.modelList[0].name)
        ns.calculateSolution('vof')
        # COMPARE VS SAVED FILES #
        actual = h5py.File('vof_level_0_SmoothnessBased.h5','r')
        expected_path = 'comparison_files/' + 'comparison_' + 'vof_level_0_SmoothnessBased_' + '_u_t2.csv'
        #write comparison file
        #np.array(actual['u_t2']).tofile(os.path.join(self._scriptdir, expected_path),sep=",")
        np.testing.assert_almost_equal(np.fromfile(os.path.join(self._scriptdir, expected_path),sep=","),np.array(actual['u_t2']).flatten(),decimal=3)



        actual.close()

    def test_stab4(self, opts):
        thelper_vof.ct.STABILIZATION_TYPE = 4 # Proposed by D.Kuzmin
        thelper_vof.ct.FCT = True
        reload(thelper_vof_p)
        reload(thelper_vof_n)
        self.so.name = self.pList[0].name+"_stab4"
        # NUMERICAL SOLUTION #
        ns = proteus.NumericalSolution.NS_base(self.so,
                                               self.pList,
                                               self.nList,
                                               self.sList,
                                               opts)
        self.sim_names.append(ns.modelList[0].name)
        ns.calculateSolution('vof')
        # COMPARE VS SAVED FILES #
        actual = h5py.File('vof_level_0_stab4.h5','r')
        expected_path = 'comparison_files/' + 'comparison_' + 'vof_level_0_stab4_' + '_u_t2.csv'
        #write comparison file
        #np.array(actual['u_t2']).tofile(os.path.join(self._scriptdir, expected_path),sep=",")
        np.testing.assert_almost_equal(np.fromfile(os.path.join(self._scriptdir, expected_path),sep=","),np.array(actual['u_t2']).flatten(),decimal=8)
        actual.close()
