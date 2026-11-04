#!/usr/bin/env python
""" Test modules for Driven Cavity Stokes preconditioners. """

import proteus.test_utils.TestTools as TestTools
from proteus import Profiling, NumericalSolution
Profiling.logLevel = 7
Profiling.verbose = True

import os
import sys
import inspect
import numpy as np
import h5py
import pickle
import pytest

from petsc4py import PETSc

TestTools.addSubFolders( inspect.currentframe() )
from proteus import defaults
import_modules = os.path.join(os.path.dirname(os.path.abspath(__file__)),'import_modules')
twp_navier_stokes_cavity_2d_so = defaults.load_system('twp_navier_stokes_cavity_2d_so',import_modules)
twp_navier_stokes_cavity_2d_p = defaults.load_physics('twp_navier_stokes_cavity_2d_p',import_modules)
twp_navier_stokes_cavity_2d_n = defaults.load_numerics('twp_navier_stokes_cavity_2d_n',import_modules)

def clean_up_directory():
    FileList = ['forceHistory_p',
                'forceHistory_v',
                #'cavity2D',
                #'mesh',
                'momentHistory',
                'proteus',
                'wettedAreaHistory',
                'twp_navier_stokes_cavity_2d']
    mesh_ext = ['asy', 'edge', 'ele', 'neig', 'node',
                'ply', 'poly', 'txt', 'xmf','ymf', 'h5', 'log']
    TestTools.removeFiles(prefix_ext_tuple=(FileList,mesh_ext))

@pytest.fixture()
def initialize_tp_pcd_options(request):
    petsc_options = PETSc.Options()
    petsc_options.clear()#doesn't really clear all options 
    for k in petsc_options.getAll(): petsc_options.delValue(k)
    petsc_options.setValue('rans2p_ksp_type','gmres')
    petsc_options.setValue('rans2p_ksp_gmres_restart',500)
    petsc_options.setValue('rans2p_ksp_atol',1e-12)
    petsc_options.setValue('rans2p_ksp_rtol',1e-12)
    petsc_options.setValue('rans2p_ksp_gmres_modifiedgramschmidt','')
    petsc_options.setValue('rans2p_pc_type','fieldsplit')
    petsc_options.setValue('rans2p_pc_fieldsplit_type','schur')
    petsc_options.setValue('rans2p_pc_fieldsplit_schur_fact_type','upper')
    petsc_options.setValue('rans2p_pc_fieldsplit_schur_precondition','user')
    petsc_options.setValue('rans2p_fieldsplit_velocity_ksp_type','preonly')
    petsc_options.setValue('rans2p_fieldsplit_velocity_pc_type','lu')
    petsc_options.setValue('rans2p_fieldsplit_pressure_ksp_type','preonly')
    petsc_options.setValue('innerTPPCDsolver_Qp_visc_ksp_type','preonly')
    petsc_options.setValue('innerTPPCDsolver_Qp_visc_pc_type','lu')
    petsc_options.setValue('innerTPPCDsolver_Qp_dens_ksp_type','preonly')
    petsc_options.setValue('innerTPPCDsolver_Qp_dens_pc_type','lu')
    petsc_options.setValue('innerTPPCDsolver_Ap_rho_ksp_type','preonly')
    petsc_options.setValue('innerTPPCDsolver_Ap_rho_ksp_constant_null_space','')
    petsc_options.setValue('innerTPPCDsolver_Ap_rho_pc_type','hypre')
    petsc_options.setValue('innerTPPCDsolver_Ap_rho_pc_hypre_type','boomeramg')

@pytest.fixture()
def load_cavity_problem(request):
    pList = [twp_navier_stokes_cavity_2d_p]
    nList = [twp_navier_stokes_cavity_2d_n]
    so = twp_navier_stokes_cavity_2d_so
    from proteus import default_s
    so.sList = [default_s]
    yield pList, nList, so
    
@pytest.mark.Stabilization
@pytest.mark.LinearSolvers
def test_bochev_pressure_cavity(opts,load_cavity_problem,
                                initialize_tp_pcd_options):
    initialize_tp_pcd_options
    ns =NumericalSolution.NS_base(load_cavity_problem[2],
                                  load_cavity_problem[0],
                                  load_cavity_problem[1],
                                  load_cavity_problem[2].sList,
                                  opts)

    assert ns.modelList[0].solver.solverList[0].linearSolver.null_space.get_name() == 'constant_pressure'
    ns.calculateSolution('bochev_pressure')
    script_dir = os.path.dirname(__file__)
#    relpath = 'comparison_files/twp_navier_stokes_cavity_2d.h5'
#    expected = tables.open_file(os.path.join(script_dir,relpath))
#    actual = tables.open_file('twp_navier_stokes_cavity_2d.h5','r')
##    assert numpy.allclose(expected.root.p_t1,actual.root.p_t1)
#    expected.close()
    actual = h5py.File('twp_navier_stokes_cavity_2d.h5','r')

    expected_path = 'comparison_files/' + 'comparison_' + 'twp_navier_stokes_cavity_2d' + '_p_t1.csv'
    #write comparison file
    #np.array(actual.root.p_t1).tofile(os.path.join(script_dir, expected_path),sep=",")
    np.testing.assert_almost_equal(np.fromfile(os.path.join(script_dir, expected_path),sep=","),np.array(actual['p_t1']).flatten(),decimal=8)
    actual.close()
    clean_up_directory()