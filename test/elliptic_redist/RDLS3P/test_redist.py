#!/usr/bin/env python
"""
Test module for Elliptic Re-distancing
"""
import proteus
from importlib import reload
from proteus import Profiling, default_s
from proteus import Comm
comm = Comm.get()
Profiling.logLevel=7
Profiling.verbose=True
import os
import numpy as np
import h5py
import pytest
from petsc4py import PETSc
from . import (vortex2D, vortex2D_so,
               ncls_p, ncls_n,
               rdls_p, rdls_n)

class TestEllipticRedistancing(object):

    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def setup_method(self,method):
        """Initialize the test problem. """
        self._scriptdir = os.path.dirname(__file__)

    def teardown_method(self,method):
        pass

    def test_ELLIPTIC_REDISTANCING_0(self, opts):
        # Set parameters for test #
        vortex2D.ct.ELLIPTIC_REDISTANCING = 0
        reload(rdls_p)
        reload(vortex2D_so)        
        pnList = [(ncls_p,ncls_n),
                  (rdls_p,rdls_n)]
        self.so = vortex2D_so
        pList=[]
        nList=[]
        sList=[]
        for (pModule,nModule) in pnList:
            pList.append(pModule)
            if pList[-1].name == None:
                pList[-1].name = pModule.__name__
            nList.append(nModule)
        for i in range(len(pnList)):
            sList.append(default_s)
        self.so.name += "_ELLIPTIC_REDIST_0"
        # NUMERICAL SOLUTION #
        petsc_options = PETSc.Options()
        petsc_options.setValue('ncls_pc_type','lu')
        petsc_options.setValue('ncls_ksp_type','preonly')
        petsc_options.setValue('ncls_pc_factor_mat_solver_package','superlu')
        petsc_options.setValue('rdls_pc_type','lu')
        petsc_options.setValue('rdls_ksp_type','preonly')
        petsc_options.setValue('rdls_pc_factor_mat_solver_package','superlu')
        ns = proteus.NumericalSolution.NS_base(self.so,
                                               pList,
                                               nList,
                                               sList,
                                               opts)
        ns.calculateSolution('rdls')
        actual = h5py.File('vortex_c0p1_level_1_ELLIPTIC_REDIST_0.h5','r')
        assert np.isclose(np.amax(actual['u_t1']),0.13600213609175285,atol=1e-10)
        actual.close()

    def test_ELLIPTIC_REDISTANCING_1(self, opts):
        # Set parameters for test #
        vortex2D.ct.ELLIPTIC_REDISTANCING = 1
        reload(rdls_p)
        reload(vortex2D_so)
        pnList = [(ncls_p,ncls_n),
                  (rdls_p,rdls_n)]
        self.so = vortex2D_so
        pList=[]
        nList=[]
        sList=[]
        for (pModule,nModule) in pnList:
            pList.append(pModule)
            if pList[-1].name == None:
                pList[-1].name = pModule.__name__
            nList.append(nModule)
        for i in range(len(pnList)):
            sList.append(default_s)
        self.so.name += "_ELLIPTIC_REDIST_1"
        # NUMERICAL SOLUTION #
        petsc_options = PETSc.Options()
        petsc_options.setValue('ncls_pc_type','lu')
        petsc_options.setValue('ncls_ksp_type','preonly')
        petsc_options.setValue('ncls_pc_factor_mat_solver_package','superlu')
        petsc_options.setValue('rdls_pc_type','lu')
        petsc_options.setValue('rdls_ksp_type','preonly')
        petsc_options.setValue('rdls_pc_factor_mat_solver_package','superlu')
        ns = proteus.NumericalSolution.NS_base(self.so,
                                               pList,
                                               nList,
                                               sList,
                                               opts)
        ns.calculateSolution('rdls')
        actual = h5py.File('vortex_c0p1_level_1_ELLIPTIC_REDIST_1.h5','r')
        assert np.isclose(np.amax(actual['u_t1']),0.13084321505201912,atol=1e-10)
        actual.close()

    def test_ELLIPTIC_REDISTANCING_2(self, opts):
        # Set parameters for test #
        vortex2D.ct.ELLIPTIC_REDISTANCING = 2
        reload(rdls_p)
        reload(vortex2D_so)
        pnList = [(ncls_p,ncls_n),
                  (rdls_p,rdls_n)]
        self.so = vortex2D_so
        pList=[]
        nList=[]
        sList=[]
        for (pModule,nModule) in pnList:
            pList.append(pModule)
            if pList[-1].name == None:
                pList[-1].name = pModule.__name__
            nList.append(nModule)
        for i in range(len(pnList)):
            sList.append(default_s)
        self.so.name += "_ELLIPTIC_REDIST_2"
        # NUMERICAL SOLUTION #
        petsc_options = PETSc.Options()
        petsc_options.setValue('ncls_pc_type','lu')
        petsc_options.setValue('ncls_ksp_type','preonly')
        petsc_options.setValue('ncls_pc_factor_mat_solver_package','superlu')
        petsc_options.setValue('rdls_pc_type','lu')
        petsc_options.setValue('rdls_ksp_type','preonly')
        petsc_options.setValue('rdls_pc_factor_mat_solver_package','superlu')
        ns = proteus.NumericalSolution.NS_base(self.so,
                                               pList,
                                               nList,
                                               sList,
                                               opts)
        ns.calculateSolution('rdls')
        actual = h5py.File('vortex_c0p1_level_1_ELLIPTIC_REDIST_2.h5','r')
        Profiling.logEvent("max u"+repr(np.amax(actual['u_t1'])))
        assert np.isclose(np.amax(actual['u_t1']),0.1060107277952287,atol=1e-10)
        actual.close()

    def test_ELLIPTIC_REDISTANCING_3(self, opts):
        # Set parameters for test #
        vortex2D.ct.ELLIPTIC_REDISTANCING = 3
        reload(rdls_p)
        reload(vortex2D_so)
        pnList = [(ncls_p,ncls_n),
                  (rdls_p,rdls_n)]
        self.so = vortex2D_so
        pList=[]
        nList=[]
        sList=[]
        for (pModule,nModule) in pnList:
            pList.append(pModule)
            if pList[-1].name == None:
                pList[-1].name = pModule.__name__
            nList.append(nModule)
        for i in range(len(pnList)):
            sList.append(default_s)
        self.so.name += "_ELLIPTIC_REDIST_3"
        # NUMERICAL SOLUTION #
        petsc_options = PETSc.Options()
        petsc_options.setValue('ncls_pc_type','lu')
        petsc_options.setValue('ncls_ksp_type','preonly')
        petsc_options.setValue('ncls_pc_factor_mat_solver_package','superlu')
        petsc_options.setValue('rdls_pc_type','lu')
        petsc_options.setValue('rdls_ksp_type','preonly')
        petsc_options.setValue('rdls_pc_factor_mat_solver_package','superlu')
        ns = proteus.NumericalSolution.NS_base(self.so,
                                               pList,
                                               nList,
                                               sList,
                                               opts)
        ns.calculateSolution('rdls')
        actual = h5py.File('vortex_c0p1_level_1_ELLIPTIC_REDIST_3.h5','r')
        assert np.isclose(np.amax(actual['u_t1']),0.10593090830115062,atol=1e-10)
        actual.close()
