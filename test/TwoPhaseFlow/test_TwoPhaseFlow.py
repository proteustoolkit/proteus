#!/usr/bin/env python
"""
Test module for TwoPhaseFlow
"""
import h5py
import pytest
import numpy as np
import proteus.defaults
from proteus import Context
from proteus import default_so
from proteus.iproteus import *
import os
import sys
Profiling.logLevel=1
Profiling.verbose=True

class TestTwoPhaseFlow(object):

    def setup_method(self,method):
        self._scriptdir = os.path.dirname(__file__)
        self.path = self._scriptdir

    def teardown_method(self, method):
        """ Tear down function """
        FileList = ['marin.h5','marin.xmf'
                    'moses.h5','moses.xmf'
                    'damBreak.h5','damBreak.xmf'
                    'TwoDimBucklingFlow.h5','TwoDimBucklingFlow.xmf'
                    'filling.h5','filling.xmf'
                    ]
        for file in FileList:
            if os.path.isfile(file):
                os.remove(file)
            else:
                pass

    def compare_vs_saved_files(self,name,write=False,decimal=6):
        # `decimal` is per-caller because these baselines were generated on one
        # machine and decimal=6 (agreement below ~5e-7) is tighter than
        # cross-platform floating-point reproducibility allows for some of them.
        # Loosen only the cases that need it, rather than the default for all.
        actual = h5py.File(name+'.h5','r')

        expected_path = 'comparison_files/' + 'comparison_' + name + '_phi_t2.csv'
        #write comparison file
        if(write):
            np.array(actual['phi_t2']).tofile(os.path.join(self._scriptdir, expected_path),sep=",")
        np.testing.assert_almost_equal(np.fromfile(os.path.join(self._scriptdir, expected_path),sep=","),np.array(actual['phi_t2'][:]).flatten(),decimal=decimal)

        expected_path = 'comparison_files/' + 'comparison_' + name + '_velocity_t2.csv'
        #write comparison file
        if(write):
            np.array(actual['velocity_t2']).tofile(os.path.join(self._scriptdir, expected_path),sep=",")
        np.testing.assert_almost_equal(np.fromfile(os.path.join(self._scriptdir, expected_path),sep=","),np.array(actual['velocity_t2']).flatten(),decimal=decimal)

        actual.close()

    # *** 2D tests *** #
    def test_risingBubble(self): #uses structured triangle mesh
        os.system("parun --TwoPhaseFlow --path " + self.path + " "
                  "risingBubble.py -l5 -v -C 'final_time=0.1 dt_output=0.1 refinement=1'")
        self.compare_vs_saved_files("risingBubble")

    def test_damBreak(self):
        os.system("parun --TwoPhaseFlow --path " + self.path + " "
                  "damBreak.py -l5 -v -C 'final_time=0.1 dt_output=0.1 he=0.1'")
        self.compare_vs_saved_files("damBreak")

    def test_damBreak_hotstart(self):
        os.system("parun --TwoPhaseFlow --path " + self.path + " "
                  "damBreak.py -l5 -v -H -C 'final_time=0.1 dt_output=0.1 he=0.1 hotstart=True'")

    def test_TwoDimBucklingFlow(self):
        os.system("parun --TwoPhaseFlow --path " + self.path + " "
                  "TwoDimBucklingFlow.py -l5 -v -C 'final_time=0.1 dt_output=0.1 he=0.09'")
        self.compare_vs_saved_files("TwoDimBucklingFlow")

    def test_fillingTank(self):
        os.system("parun --TwoPhaseFlow --path " + self.path + " "
                  "fillingTank.py -l5 -v -C 'final_time=0.02 dt_output=0.02 he=0.01'")
        self.compare_vs_saved_files("fillingTank")

    # *** 3D tests *** #
    def test_marin(self):
        os.system("parun --TwoPhaseFlow --path " + self.path + " "
                  "marin.py -l5 -v -C 'final_time=0.1 dt_output=0.1 he=0.5'")
        self.compare_vs_saved_files("marin")

    def test_moses(self):
        os.system("parun --TwoPhaseFlow --path " + self.path + " "
                  "moses.py -l5 -v -C 'final_time=0.1 dt_output=0.1 he=0.5'")
        # decimal=5, not the default 6: this baseline was generated against one
        # BLAS build and does not reproduce to 6 decimals elsewhere. Every
        # aarch64 environment tested (cekees-spark2 in all 8 install pathways, and
        # GitHub CI's ubuntu-24.04-arm) disagrees on 2 of 3129 elements by
        # 2.1e-6 -- about 4x the decimal=6 tolerance of ~5e-7 -- and does so
        # deterministically: the two machines agree to all 17 printed digits.
        # linux-x86_64 also fails it under Spack, whose OpenBLAS is built from
        # source with different kernel selection, while passing under conda and
        # pip. So the variable is the numeric library, not the architecture.
        # decimal=5 leaves an order of magnitude of headroom and still catches
        # anything real. See the audit task on baseline tolerances suite-wide.
        self.compare_vs_saved_files("moses", decimal=5)

    # The PUMI run reads the mesh the --genPUMI run writes (Reconstructed0.smb,
    # finalMesh0.smb), so the two are one test: it does not depend on another
    # test having run. os.system() never raises on nonzero exit, so the
    # compare_vs_saved_files() call is the real assertion; it passes against the
    # existing baseline.
    def test_damBreak_genPUMI_runPUMI(self):
        os.system("parun --TwoPhaseFlow --genPUMI --path " + self.path + " "
                  "damBreak.py -l5 -v -C 'final_time=0.1 dt_output=0.1 he=0.1'")
        os.system("parun --TwoPhaseFlow --path " + self.path + " "
                  "damBreak_PUMI.py -l5 -v -C 'final_time=0.1 dt_output=0.1 he=0.1 adapt=0'")
        self.compare_vs_saved_files("damBreak_PUMI")
