"""Tests for 2d flow around an immersed boundary cylinder with rans2p"""
import os
import sys
from proteus import Profiling
from proteus import Comm
from proteus import Context
from proteus import MeshTools
from proteus import Domain
from proteus.MeshAdaptPUMI import MeshAdapt
import h5py
import importlib
import subprocess
import pytest


comm = Comm.get()
Profiling.logLevel = 7
Profiling.verbose = False
import numpy as np


class Test_Adapt_ibm():

    @classmethod
    def setup_class(cls):
        cls._scriptdir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0,cls._scriptdir)
    @classmethod
    def teardown_class(cls):
        sys.path.remove(cls._scriptdir)
        pass

    def setup_method(self, method):
        """Initialize the test problem. """
        self.aux_names = []

    def teardown_method(self, method):
        """ Tear down function """
        FileList = ['mesh.ele',
                    'mesh.edge',
                    'mesh.node',
                    'mesh.neigh',
                    'mesh.face',
                    'mesh.poly',
                    'mesh.1.neigh',
                    'mesh.1.poly',
                    'mesh.ply',
                    'mesh.asy',
                    'cylinder_so.log',
                    #'Reconstructed.dmg', #can't remove since teardown is called after each test
                    #'Reconstructed0.smb',
                    'proteus.log',
                    'cylinder.xmf', 'cylinder.ymf',
                    'cylinder.h5',
                    'finalMesh0.smb',
                    'particle_forceHistory.txt',
                    'particle_momentHistory.txt',
                    'particle_vforceHistory.txt',
                    'particle_pforceHistory.txt',
                    'momentHistory.txt',
                    'TimeList.txt',
                    'timeHistory.txt',
                    'forceHistory_p.txt',
                    'forceHistory_v.txt',
                    'wettedAreaHistory.txt'
                    ]
        for file in FileList:
            if os.path.isfile(file):
                os.remove(file)
            else:
                pass

    def test_adaptIBM_adaptMesh(self, case_copy):
        # in a copy of this directory, so the runs' files stay out of the checkout
        currentPath = str(case_copy(os.path.dirname(os.path.abspath(__file__))))
        # the adapt run reads the Reconstructed.* model and mesh this one writes
        runCommand = "cd "+currentPath+"; parun -l5 cylinder_so.py -C 'T=0.01 onlySaveFinalSolution=True genMesh=True usePUMI=True';"
        subprocess.run(runCommand,shell=True, check=True, timeout=900)
        runCommand = "cd "+currentPath+"; parun -l5 cylinder_so.py -C 'T=0.01 onlySaveFinalSolution=True genMesh=False usePUMI=True';"
        subprocess.run(runCommand,shell=True, check=True, timeout=900)

        #load initial mesh and extract element count
        domain = Domain.PUMIDomain(manager=MeshAdapt.AdaptManager()) #initialize the domain
        filePath=bytes(currentPath+'/','utf-8')
        domain.AdaptManager.PUMIAdapter.loadModelAndMesh(filePath+b"Reconstructed.dmg", filePath+b"Reconstructed.smb")
        mesh = MeshTools.TetrahedralMesh()
        mesh.convertFromPUMI(domain,domain.AdaptManager.PUMIAdapter,
                     [1],
                     [1],
                     parallel = comm.size() > 1,
                     dim=2)
        nElements_initial = mesh.nElements_global

        #load final mesh and extract element count
        domain.AdaptManager.PUMIAdapter.loadModelAndMesh(filePath+b"Reconstructed.dmg", filePath+b"finalMesh.smb")
        mesh2 = MeshTools.TetrahedralMesh()
        mesh2.convertFromPUMI(domain,domain.AdaptManager.PUMIAdapter,
                     [1],
                     [1],
                     parallel = comm.size() > 1,
                     dim=2)
        nElements_final = mesh2.nElements_global

        #adapted mesh should have less elements
        assert nElements_final < nElements_initial
