#!/usr/bin/env python
"""
Test module for TADR with EV
"""
import os
import proteus
from importlib import reload
from proteus import Profiling, default_n, default_s, default_so
from proteus import Comm
comm = Comm.get()
Profiling.logLevel=2
Profiling.verbose=True
import numpy as np
import h5py
from . import thelper_tadr
from . import thelper_tadr_p
from . import thelper_tadr_n

class TestTADR(object):

    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def setup_method(self,method):
        """Initialize the test problem. """
        reload(default_n)
        reload(thelper_tadr)
        self.pList = [thelper_tadr_p]
        self.nList = [thelper_tadr_n]
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
        thelper_tadr.ct.STABILIZATION_TYPE = 0 # SUPG
        thelper_tadr.ct.FCT = False
        reload(default_n)
        reload(thelper_tadr_p)
        reload(thelper_tadr_n)
        tname = "_SUPG"
        self.so.name = self.pList[0].name+tname
        # NUMERICAL SOLUTION #
        ns = proteus.NumericalSolution.NS_base(self.so,
                                               self.pList,
                                               self.nList,
                                               self.sList,
                                               opts)
        self.sim_names.append(ns.modelList[0].name)
        ns.calculateSolution('tadr')
        # COMPARE VS SAVED FILES #
        
        with h5py.File("tadr_level_0"+tname+".h5",'r') as actual:
            expected_path = os.path.join(self._scriptdir,"comparison_files","tadr_level_0"+tname+ "_u_t2.csv")
            #write comparison file
            #(actual['u_t2'][:]).tofile(expected_path,sep=",")
            np.testing.assert_almost_equal(np.fromfile(expected_path,sep=","),actual['u_t2'][:],decimal=10)

    def test_TaylorGalerkin(self, opts):
        ##################
        # TaylorGalerkin #
        ##################
        thelper_tadr.ct.STABILIZATION_TYPE = 1 # Taylor Galerkin
        thelper_tadr.ct.FCT = False
        reload(default_n)
        reload(thelper_tadr_p)
        reload(thelper_tadr_n)
        tname = "_TaylorGalerkin"
        self.so.name = self.pList[0].name+tname
        # NUMERICAL SOLUTION #
        ns = proteus.NumericalSolution.NS_base(self.so,
                                               self.pList,
                                               self.nList,
                                               self.sList,
                                               opts)
        self.sim_names.append(ns.modelList[0].name)
        ns.calculateSolution('tadr')
        # COMPARE VS SAVED FILES #
        with h5py.File("tadr_level_0"+tname+".h5",'r') as actual:
            expected_path = os.path.join(self._scriptdir,"comparison_files","tadr_level_0"+tname+ "_u_t2.csv")
            #write comparison file
            #(actual['u_t2'][:]).tofile(expected_path,sep=",")
            np.testing.assert_almost_equal(np.fromfile(expected_path,sep=","),actual['u_t2'][:],decimal=10)

    def test_EV1(self, opts):
        #######################
        # ENTROPY VISCOSITY 1 # Polynomial entropy
        #######################
        thelper_tadr.ct.STABILIZATION_TYPE = 2 # EV
        thelper_tadr.ct.ENTROPY_TYPE = 1 #polynomial
        thelper_tadr.ct.cE = 1.0
        thelper_tadr.ct.FCT = True
        reload(default_n)
        reload(thelper_tadr_p)
        reload(thelper_tadr_n)
        tname = "_EV1"
        self.so.name = self.pList[0].name+tname
        # NUMERICAL SOLUTION #
        ns = proteus.NumericalSolution.NS_base(self.so,
                                               self.pList,
                                               self.nList,
                                               self.sList,
                                               opts)
        self.sim_names.append(ns.modelList[0].name)
        ns.calculateSolution('tadr')
        # COMPARE VS SAVED FILES #
        with h5py.File("tadr_level_0"+tname+".h5",'r') as actual:
            expected_path = os.path.join(self._scriptdir,"comparison_files", "tadr_level_0"+tname+ "_u_t2.csv")
            #write comparison file
            #(actual['u_t2'][:]).tofile(expected_path,sep=",")
            np.testing.assert_almost_equal(np.fromfile(expected_path,sep=","),actual['u_t2'][:],decimal=1)

    def test_EV2(self, opts):
        thelper_tadr.ct.STABILIZATION_TYPE = 2 # EV
        thelper_tadr.ct.ENTROPY_TYPE = 1 #logarithmic
        thelper_tadr.ct.cE = 0.1
        thelper_tadr.ct.FCT = True
        reload(default_n)
        reload(thelper_tadr_p)
        reload(thelper_tadr_n)
        tname = "_EV2"
        self.so.name = self.pList[0].name+tname
        # NUMERICAL SOLUTION #
        ns = proteus.NumericalSolution.NS_base(self.so,
                                               self.pList,
                                               self.nList,
                                               self.sList,
                                               opts)
        self.sim_names.append(ns.modelList[0].name)
        ns.calculateSolution('tadr')
        # COMPARE VS SAVED FILES #
        with h5py.File("tadr_level_0"+tname+".h5",'r') as actual:
            expected_path = os.path.join(self._scriptdir,"comparison_files","tadr_level_0"+tname+ "_u_t2.csv")
            #write comparison file
            #(actual['u_t2'][:]).tofile(expected_path,sep=",")
            np.testing.assert_almost_equal(np.fromfile(expected_path,sep=","),actual['u_t2'][:],decimal=2)

    def test_SmoothnessBased(self, opts):
        thelper_tadr.ct.STABILIZATION_TYPE = 3 # Smoothness based
        thelper_tadr.ct.FCT = True
        reload(default_n)
        reload(thelper_tadr_p)
        reload(thelper_tadr_n)
        tname = "_SmoothnessBased"
        self.so.name = self.pList[0].name+tname
        # NUMERICAL SOLUTION #
        ns = proteus.NumericalSolution.NS_base(self.so,
                                               self.pList,
                                               self.nList,
                                               self.sList,
                                               opts)
        self.sim_names.append(ns.modelList[0].name)
        ns.calculateSolution('tadr')
        # COMPARE VS SAVED FILES #
        with h5py.File("tadr_level_0"+tname+".h5",'r') as actual:
            expected_path = os.path.join(self._scriptdir,"comparison_files","tadr_level_0"+tname+ "_u_t2.csv")
            #write comparison file
            #(actual['u_t2'][:]).tofile(expected_path,sep=",")
            np.testing.assert_almost_equal(np.fromfile(expected_path,sep=","),actual['u_t2'][:],decimal=3)

    def test_stab4(self, opts):
        thelper_tadr.ct.STABILIZATION_TYPE = 4 # Proposed by D.Kuzmin
        thelper_tadr.ct.FCT = True
        reload(default_n)
        reload(thelper_tadr_p)
        reload(thelper_tadr_n)
        tname="_stab4"
        self.so.name = self.pList[0].name+tname
        # NUMERICAL SOLUTION #
        ns = proteus.NumericalSolution.NS_base(self.so,
                                               self.pList,
                                               self.nList,
                                               self.sList,
                                               opts)
        self.sim_names.append(ns.modelList[0].name)
        ns.calculateSolution('tadr')
        # COMPARE VS SAVED FILES #
        with h5py.File("tadr_level_0"+tname+".h5",'r') as actual:
            expected_path = os.path.join(self._scriptdir,"comparison_files","tadr_level_0"+tname+ "_u_t2.csv")
            #write comparison file
            #(actual['u_t2'][:]).tofile(expected_path,sep=",")
            # STABILIZATION_TYPE=4 (Kuzmin's FCT limiter) clips values against a
            # bound; a handful of nodes sitting almost exactly on that bound can
            # flip which side they land on from a tiny (~1e-8 absolute, ~1.5e-7
            # relative) cross-build BLAS/Hypre/compiler difference, unlike this
            # file's other (unlimited) test cases, which stay reproducible to
            # decimal=10. Confirmed real: conda-forge, pip, and both PETSc
            # BuildSystem builds agree bit-for-bit with each other but differ
            # from an independently-toolchained Spack build at exactly this
            # tolerance. assert_almost_equal's fixed decimal count can't express
            # "tight everywhere except at a limiter threshold", so use an
            # explicit relative+absolute tolerance sized to what's actually been
            # observed across builds (with a comfortable margin), rather than
            # enshrining one build's exact bit pattern as the only correct one.
            np.testing.assert_allclose(np.fromfile(expected_path,sep=","),actual['u_t2'][:],rtol=1e-6,atol=1e-7)

    def test_stab5_low_order_outflow_is_m_matrix(self, opts, monkeypatch):
        """STABILIZATION_TYPE=5 (ImplicitEV) low order on the Zalesak disk (problem 1, with inflow and outflow
        walls): no positive off-diagonal in the Jacobian, so the low-order solution obeys the discrete maximum
        principle. On an upwinded outflow face the solute flux must be c_i times the fluid flux; the trace flux
        n.f(c_h) coupled c_i to the other face nodes with a positive off-diagonal (Zalesak refinement 4 with FCT
        then reached c^L = -2.9e-9 at the outflow wall node (0, 0.625))."""
        # thelper_tadr builds its mesh from the context when imported, so the problem is set before the reload
        from proteus import Context
        monkeypatch.setattr(Context, 'contextOptionsString',
                            'problem=1 nd=2 T=0.01 nDTout=1 refinement=1 STABILIZATION_TYPE=5 FCT=False cfl=0.5')
        try:
            reload(thelper_tadr)
            reload(default_n)
            p = reload(thelper_tadr_p)
            n = reload(thelper_tadr_n)
            self.so.tnList = n.tnList
            self.so.name = p.name + "_stab5_mmatrix"
            ns = proteus.NumericalSolution.NS_base(self.so, [p], [n], self.sList, opts)
            ns.calculateSolution('tadr')
            model = ns.modelList[0].levelModelList[-1]
            jacobian = model.initializeJacobian()
            model.getResidual(model.u[0].dof, np.zeros_like(model.u[0].dof))
            model.getJacobian(jacobian)
        finally:
            # setup_method takes the next test's tnList from thelper_tadr_n as already imported, so put the
            # default (problem 0) modules back
            monkeypatch.undo()
            reload(thelper_tadr)
            reload(default_n)
            reload(thelper_tadr_p)
            reload(thelper_tadr_n)
        rowptr, colind, nzval = jacobian.getCSRrepresentation()
        rows = np.repeat(np.arange(len(rowptr) - 1), np.diff(rowptr))
        off = colind != rows
        diagonal = np.zeros(len(rowptr) - 1)
        diagonal[rows[~off]] = nzval[~off]
        # the low-order operator's off-diagonals are -d_ij + c_ij.v <= 0 by construction; 1e-12 of the row's
        # diagonal only allows for round-off
        assert (nzval[off] <= 1.0e-12 * np.abs(diagonal[rows[off]])).all()
