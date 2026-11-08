#!/usr/bin/env python
"""
Test module for surface tension
"""
import proteus
from importlib import reload
from proteus import Profiling, default_s
from proteus import Comm
comm = Comm.get()
Profiling.logLevel=1
Profiling.verbose=True
import os
import numpy as np
import h5py
import pytest
from . import (parameters,
               risingBubble_so, risingBubble,
               vof_p,
               vof_n,
               ls_p,
               ls_n,
               redist_p,
               redist_n,               
               ls_consrv_p,
               ls_consrv_n,
               twp_navier_stokes_p,
               twp_navier_stokes_n,
               pressureincrement_p,
               pressureincrement_n,
               pressure_p,
               pressure_n,
               pressureInitial_p,
               pressureInitial_n)


class TestSurfaceTension(object):

    @classmethod
    def setup_class(cls):
        pass
    
    @classmethod
    def teardown_class(cls):
        pass
    
    def reload_modules(self):
        reload(risingBubble)
        reload(risingBubble_so)
        reload(vof_p)
        reload(vof_n)
        reload(ls_p)
        reload(ls_n)
        reload(redist_p)
        reload(redist_n)               
        reload(ls_consrv_p)
        reload(ls_consrv_n)
        reload(twp_navier_stokes_p)
        reload(twp_navier_stokes_n)
        reload(pressureincrement_p)
        reload(pressureincrement_n)
        reload(pressure_p)
        reload(pressure_n)
        reload(pressureInitial_p)
        reload(pressureInitial_n)
        
    def setup_method(self,method):
        self._scriptdir = os.path.dirname(__file__)
        
    def teardown_method(self,method):
        pass
    
    def compare_files(self,path,name):
        actual = h5py.File(name+'.h5','r')
        expected_path = 'comparison_files/' + 'comparison_' + name + '_phi_t2.csv'
        #write comparison file
        #np.array(actual.root.phi_t2).tofile(os.path.join(self._scriptdir, expected_path),sep=",")
        np.testing.assert_allclose(np.fromfile(os.path.join(self._scriptdir, expected_path),sep=","),np.array(actual['phi_t2']).flatten(),atol=0.05,rtol=0.05)

        expected_path = 'comparison_files/' + 'comparison_' + name + '_p_t2.csv'
        #np.array(actual.root.p_t2).tofile(os.path.join(self._scriptdir, expected_path),sep=",")
        np.testing.assert_allclose(np.fromfile(os.path.join(self._scriptdir, expected_path),sep=","),np.array(actual['p_t2']).flatten(),atol=0.05,rtol=0.05)

        expected_path = 'comparison_files/' + 'comparison_' + name + '_velocity_t2.csv'
        #np.array(actual.root.velocity_t2).tofile(os.path.join(self._scriptdir, expected_path),sep=",")
        np.testing.assert_allclose(np.fromfile(os.path.join(self._scriptdir, expected_path),sep=","),np.array(actual['velocity_t2']).flatten(),atol=0.05,rtol=0.05)

        actual.close()


    def test_2D_with_supg(self, opts):
        # Set parameters for this test
        parameters.ct.USE_SUPG_NS=1
        parameters.ct.ARTIFICIAL_VISCOSITY_NS=1
        parameters.ct.nd=2
        # RELOAD MODULES
        self.reload_modules()
        pnList = [(vof_p,               vof_n),
                  (ls_p,                ls_n),
                  (redist_p,            redist_n),
                  (ls_consrv_p,         ls_consrv_n),
                  (twp_navier_stokes_p, twp_navier_stokes_n),
                  (pressureincrement_p, pressureincrement_n),
                  (pressure_p,          pressure_n),
                  (pressureInitial_p,   pressureInitial_n)]
        self.so = risingBubble_so
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
        self.so.name += "_2D_supg"
        # NUMERICAL SOLUTION #
        ns = proteus.NumericalSolution.NS_base(self.so,
                                               pList,
                                               nList,
                                               sList,
                                               opts)
        ns.calculateSolution('2D_supg')
        # COMPARE VS SAVED FILES #
        self.compare_files('comparison_files',self.so.name)        

    def test_2D_with_EV(self, opts):
        # Set parameters for this test
        parameters.ct.USE_SUPG_NS=0
        parameters.ct.ARTIFICIAL_VISCOSITY_NS=2
        parameters.ct.nd=2 
        # RELOAD MODULES
        self.reload_modules()
        pnList = [(vof_p,               vof_n),
                  (ls_p,                ls_n),
                  (redist_p,            redist_n),
                  (ls_consrv_p,         ls_consrv_n),
                  (twp_navier_stokes_p, twp_navier_stokes_n),
                  (pressureincrement_p, pressureincrement_n),
                  (pressure_p,          pressure_n),
                  (pressureInitial_p,   pressureInitial_n)]
        self.so = risingBubble_so
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
        self.so.name += "_2D_ev"
        # NUMERICAL SOLUTION #
        ns = proteus.NumericalSolution.NS_base(self.so,
                                               pList,
                                               nList,
                                               sList,
                                               opts)
        ns.calculateSolution('2D_ev')
        # COMPARE VS SAVED FILES #
        self.compare_files('comparison_files',self.so.name)        

    def test_3D_with_SUPG(self, opts):
        # Set parameters for this test
        parameters.ct.USE_SUPG_NS=1
        parameters.ct.ARTIFICIAL_VISCOSITY_NS=1
        parameters.ct.nd=3
        # RELOAD MODULES
        self.reload_modules()
        pnList = [(vof_p,               vof_n),
                  (ls_p,                ls_n),
                  (redist_p,            redist_n),
                  (ls_consrv_p,         ls_consrv_n),
                  (twp_navier_stokes_p, twp_navier_stokes_n),
                  (pressureincrement_p, pressureincrement_n),
                  (pressure_p,          pressure_n),
                  (pressureInitial_p,   pressureInitial_n)]
        self.so = risingBubble_so
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
        self.so.name += "_3D_supg" 
        # NUMERICAL SOLUTION #
        ns = proteus.NumericalSolution.NS_base(self.so,
                                               pList,
                                               nList,
                                               sList,
                                               opts)
        ns.calculateSolution('3D_supg')
        # COMPARE VS SAVED FILES #
        self.compare_files('comparison_files',self.so.name)        

    def test_3D_with_EV(self, opts):
        # Set parameters for this test
        parameters.ct.USE_SUPG_NS=0
        parameters.ct.ARTIFICIAL_VISCOSITY_NS=2
        parameters.ct.nd=3
        # RELOAD MODULES
        self.reload_modules()
        pnList = [(vof_p,               vof_n),
                  (ls_p,                ls_n),
                  (redist_p,            redist_n),
                  (ls_consrv_p,         ls_consrv_n),
                  (twp_navier_stokes_p, twp_navier_stokes_n),
                  (pressureincrement_p, pressureincrement_n),
                  (pressure_p,          pressure_n),
                  (pressureInitial_p,   pressureInitial_n)]
        self.so = risingBubble_so
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
        self.so.name += "_3D_ev" 
        # NUMERICAL SOLUTION #
        ns = proteus.NumericalSolution.NS_base(self.so,
                                               pList,
                                               nList,
                                               sList,
                                               opts)
        failed = ns.calculateSolution('3D_ev')
        assert(not failed)
        # COMPARE VS SAVED FILES #
        self.compare_files('comparison_files',self.so.name)        
