#!/usr/bin/env python
"""
Test module for level set transport
"""
from importlib import reload
from proteus import NumericalSolution
import os
import numpy as np
import h5py
from . import (vortex2D,
               ls_vortex_2d_p,
               redist_vortex_2d_p,
               vof_vortex_2d_p,
               ls_consrv_vortex_2d_p,
               ls_vortex_2d_n,
               redist_vortex_2d_n,
               vof_vortex_2d_n,
               ls_consrv_vortex_2d_n,
               ls_vortex_2d_so)

class TestVortex2D(object):

    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def setup_method(self,method):
        self.aux_names = []
        self.meshdir = os.path.dirname(os.path.abspath(__file__))
        self._scriptdir = os.path.dirname(os.path.abspath(__file__))
        
    def teardown_method(self,method):
        #pass
         filenames = []
         for aux_name in self.aux_names:
             filenames.extend([aux_name+'.'+ext for ext in ['h5','xmf','ymf']])
         filenames.append('proteus.log')
         for f in filenames:
             if os.path.exists(f):
                 try:
                     os.remove(f)
                 except OSError as e:
                     print ("Error: %s - %s" %(e.filename,e.strerror))
             else:
                 pass
            
    def test_vortex2D(self,opts,use_strong_constraints=False):
        from proteus import default_s
        opts.gatherArchive=True
        sList=[]
        if ls_vortex_2d_so.sList == []:
            for i in range(len(ls_vortex_2d_so.pnList)):
                s = default_s
                sList.append(s)
        else:
            sList = ls_vortex_2d_so.sList

        ns = NumericalSolution.NS_base(ls_vortex_2d_so,
                                       [ls_vortex_2d_p,
                                        redist_vortex_2d_p,
                                        vof_vortex_2d_p,
                                        ls_consrv_vortex_2d_p],
                                       [ls_vortex_2d_n,
                                        redist_vortex_2d_n,
                                        vof_vortex_2d_n,
                                        ls_consrv_vortex_2d_n],
                                       sList,
                                       opts)
        ns.calculateSolution(ls_vortex_2d_so.name)
        self.aux_names.append(ls_vortex_2d_so.name)
        # COMPARE VS SAVED FILES #
        actual = h5py.File(ls_vortex_2d_so.name+'.h5','r')
        expected_path = 'comparison_files/' + 'comparison_' + ls_vortex_2d_so.name + '_u_t80.csv'
        #write comparison file
        #np.array(actual['u_t80']).tofile(os.path.join(self._scriptdir, expected_path),sep=",")
        np.testing.assert_almost_equal(np.fromfile(os.path.join(self._scriptdir, expected_path),sep=","),np.array(actual['u_t80']).flatten(),decimal=10)

        expected_path = 'comparison_files/' + 'comparison_' + ls_vortex_2d_so.name + '_phid_t80.csv'
        #write comparison file
        #np.array(actual['phid_t80']).tofile(os.path.join(self._scriptdir, expected_path),sep=",")
        np.testing.assert_almost_equal(np.fromfile(os.path.join(self._scriptdir, expected_path),sep=","),np.array(actual['phid_t80']).flatten(),decimal=10)

        expected_path = 'comparison_files/' + 'comparison_' + ls_vortex_2d_so.name + '_vof_t80.csv'
        #write comparison file
        #np.array(actual['vof_t80']).tofile(os.path.join(self._scriptdir, expected_path),sep=",")
        np.testing.assert_almost_equal(np.fromfile(os.path.join(self._scriptdir, expected_path),sep=","),np.array(actual['vof_t80']).flatten(),decimal=10)

        actual.close()
        del ns
        
    def test_vortex2D_exactHeaviside(self,opts,use_strong_constraints=False):
        from proteus import default_s
        reload(vortex2D)
        vortex2D.useExact=True
        reload(ls_consrv_vortex_2d_p)
        opts.gatherArchive=True
        sList=[]
        if ls_vortex_2d_so.sList == []:
            for i in range(len(ls_vortex_2d_so.pnList)):
                s = default_s
                sList.append(s)
        else:
            sList = ls_vortex_2d_so.sList
        ls_vortex_2d_so.name +='_exactHeaviside'
        ns = NumericalSolution.NS_base(ls_vortex_2d_so,
                                       [ls_vortex_2d_p,
                                        redist_vortex_2d_p,
                                        vof_vortex_2d_p,
                                        ls_consrv_vortex_2d_p],
                                       [ls_vortex_2d_n,
                                        redist_vortex_2d_n,
                                        vof_vortex_2d_n,
                                        ls_consrv_vortex_2d_n],
                                       sList,
                                       opts)
        ns.calculateSolution(ls_vortex_2d_so.name)
        self.aux_names.append(ls_vortex_2d_so.name)
        # COMPARE VS SAVED FILES #
        actual = h5py.File(ls_vortex_2d_so.name+'.h5','r')
        expected_path = 'comparison_files/' + 'comparison_' + ls_vortex_2d_so.name + '_u_t80.csv'
        #write comparison file
        #np.array(actual['u_t80']).tofile(os.path.join(self._scriptdir, expected_path),sep=",")
        np.testing.assert_almost_equal(np.fromfile(os.path.join(self._scriptdir, expected_path),sep=","),np.array(actual['u_t80']).flatten(),decimal=10)

        expected_path = 'comparison_files/' + 'comparison_' + ls_vortex_2d_so.name + '_phid_t80.csv'
        #write comparison file
        #np.array(actual['phid_t80']).tofile(os.path.join(self._scriptdir, expected_path),sep=",")
        np.testing.assert_almost_equal(np.fromfile(os.path.join(self._scriptdir, expected_path),sep=","),np.array(actual['phid_t80']).flatten(),decimal=10)

        expected_path = 'comparison_files/' + 'comparison_' + ls_vortex_2d_so.name + '_vof_t80.csv'
        #write comparison file
        #np.array(actual['vof_t80']).tofile(os.path.join(self._scriptdir, expected_path),sep=",")
        np.testing.assert_almost_equal(np.fromfile(os.path.join(self._scriptdir, expected_path),sep=","),np.array(actual['vof_t80']).flatten(),decimal=10)


        actual.close()
        del ns
        
if __name__ == '__main__':
    pass
