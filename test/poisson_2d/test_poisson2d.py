#!/usr/bin/env python
"""
Test module for linear boundary value problems (serial)

This module solves equations of the form

.. math::

  \nabla \cdot \left( a(x) \nabla u \right) = f(x)

"""
from importlib import reload
from proteus import Comm, NumericalSolution, default_n, default_s, default_so
import os
import numpy as np
import pytest
from petsc4py import PETSc
from . import poisson_het_2d_p
from . import poisson_het_2d_c0pk_n
from . import poisson_het_2d_dgpk_n

femSpaces=['c0p1','c0p2','dgp1','dgp2']
bcTypes=['weak','strong']
solverTypes=['direct','iterative']
cases =[]
for f in femSpaces:
    for b in bcTypes:
        for s in solverTypes:
            cases.append((f,b,s))

class TestPoisson2D(object):

    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def setup_method(self,method):
        self.aux_names = []
        reload(poisson_het_2d_p)
        self.meshdir = os.path.dirname(os.path.abspath(__file__))
        
    def teardown_method(self,method):
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
            
    @pytest.mark.parametrize("femSpace,bcType,solverType", cases)
    def test_poisson(self,opts,femSpace,bcType,solverType):
        reload(poisson_het_2d_p)
        pList = [poisson_het_2d_p]
        if femSpace in ['c0p1','c0p2']:
            reload(poisson_het_2d_c0pk_n)
            nList = [poisson_het_2d_c0pk_n]
            if femSpace == 'c0p1':
                nList[0].femSpaces[0]  = default_n.C0_AffineLinearOnSimplexWithNodalBasis
            else:
                nList[0].femSpaces[0]  = default_n.C0_AffineQuadraticOnSimplexWithNodalBasis
            if bcType == 'strong':
                nList[0].numericalFluxType = nList[0].Exterior_StrongFlux
        if femSpace in ['dgp1','dgp2']:
            reload(poisson_het_2d_dgpk_n)
            nList = [poisson_het_2d_dgpk_n]
            if femSpace == 'dgp1':
                nList[0].femSpaces[0]  = default_n.DG_AffineLinearOnSimplexWithNodalBasis
            else:
                nList[0].femSpaces[0]  = default_n.DG_AffineQuadraticOnSimplexWithNodalBasis
            if bcType == 'strong':
                nList[0].numericalFluxType.useStrongDirichletConstraints=True
        nList[0].nnx=nList[0].nny=nList[0].nn=51
        so = default_so
        so.name = pList[0].name = "poisson_2d_{0}_{1}_{2}_pe{3}".format(femSpace, bcType,solverType,repr(Comm.get().size()))
        so.sList=[default_s]
        opts.gatherArchive=True
        soln_name = so.name
        if solverType == 'direct':
            nList[0].linearSolver=default_n.LU
            nList[0].multilevelLinearSolver=default_n.LU
        else:
            nList[0].linearSolver=default_n.KSP_petsc4py
            nList[0].multilevelLinearSolver=default_n.KSP_petsc4py
            OptDB = nList[0].OptDB
            OptDB.clear()
            OptDB.setValue('ksp_type','cg')
            OptDB.setValue('pc_asm_type','basic')
            OptDB.setValue('pc_asm_overlap',2)
            OptDB.setValue('sub_ksp_type','preonly')
            OptDB.setValue('sub_pc_type','lu')
            OptDB.setValue('sub_pc_factor_type','superlu')
        ns = NumericalSolution.NS_base(so,pList,nList,so.sList,opts)
        ns.calculateSolution(soln_name)
        np.testing.assert_almost_equal(ns.modelList[0].levelModelList[-1].u[0].dof,
                                       ns.modelList[0].levelModelList[-1].u_analytical[0][0],
                                       decimal=4)
        self.aux_names.append(pList[0].name)
        del ns

    def test_2dm(self,opts,use_strong_constraints=False,nLevels=3):
        reload(poisson_het_2d_p)
        reload(poisson_het_2d_c0pk_n)
        # square4x4.ymf/.h5 is square4x4.3dm/.bc converted by ymf-import-sms; proteus reads meshes as ymf
        poisson_het_2d_p.meshfile=os.path.join(self.meshdir,'square4x4')
        assert os.path.isfile(poisson_het_2d_p.meshfile+'.ymf'), "could not find meshfile {0}".format(poisson_het_2d_p.meshfile+'.ymf')
        poisson_het_2d_p.x0 = (0.0,0.0,0.)
        poisson_het_2d_p.L  = (1.,1.,1.)
        poisson_het_2d_c0pk_n.nLevels = nLevels
        pList = [poisson_het_2d_p]
        nList = [poisson_het_2d_c0pk_n]
        so = default_so
        so.name = pList[0].name = "poisson_2d_c0p1_2dm"+"pe"+repr(Comm.get().size())
        so.sList=[default_s]
        opts.gatherArchive=True
        nList[0].femSpaces[0]  = default_n.C0_AffineLinearOnSimplexWithNodalBasis
        nList[0].linearSolver=default_n.KSP_petsc4py
        nList[0].multilevelLinearSolver=default_n.KSP_petsc4py
        nList[0].numericalFluxType = default_n.Advection_DiagonalUpwind_Diffusion_SIPG_exterior
        soln_name = 'poisson_2d_c0p1_2dm'
        if use_strong_constraints == True:
            nList[0].numericalFluxType = nList[0].Exterior_StrongFlux
        soln_name = 'poisson_2d_c0p1_strong_dirichlet'
        #nList[0].linearSolver=default_n.LU
        #nList[0].multilevelLinearSolver=default_n.LU
        ns = NumericalSolution.NS_base(so,pList,nList,so.sList,opts)
        ns.calculateSolution(soln_name)
        self.aux_names.append(pList[0].name)
        del ns

@pytest.mark.parametrize("use_weak_dirichlet", [False,True])
def test_load_vector_use_weak(opts, use_weak_dirichlet):
    from . import poisson_het_2d_p
    reload(poisson_het_2d_p)
    from . import poisson_het_2d_c0pk_n
    # test_poisson's strong cases set numericalFluxType on this module, and
    # on a worker that ran one last, it would still hold Exterior_StrongFlux
    reload(poisson_het_2d_c0pk_n)
    pList = [poisson_het_2d_p]
    nList = [poisson_het_2d_c0pk_n]
    so = default_so
    so.name = pList[0].name = "poisson_2d_c0p1"+"pe"+repr(Comm.get().size())
    so.sList=[default_s]
    opts.gatherArchive=True
    nList[0].femSpaces[0]  = default_n.C0_AffineLinearOnSimplexWithNodalBasis
    nList[0].linearSolver=default_n.LU
    nList[0].multilevelLinearSolver=default_n.LU
    if use_weak_dirichlet:
        nList[0].linearSolver=default_n.LU
        nList[0].multilevelLinearSolver=default_n.LU
        nList[0].numericalFluxType = default_n.Advection_DiagonalUpwind_Diffusion_SIPG_exterior
    OptDB = nList[0].OptDB
    soln_name = so.name
    OptDB.clear()
    OptDB.setValue('ksp_type','bcgsl')
    OptDB.setValue('pc_type','asm')
    OptDB.setValue('pc_asm_type','basic')
    OptDB.setValue('pc_asm_overlap',2)
    OptDB.setValue('sub_ksp_type','preonly')
    OptDB.setValue('sub_pc_type','lu')
    OptDB.setValue('sub_pc_factor_type','superlu')
    ns = NumericalSolution.NS_base(so,pList,nList,so.sList,opts)
    ns.calculateSolution('poisson_2d_c0p1')
    #test load vector calculation in a crude way
    #see if sum over test functions of residual is same as the sum over the test functions of the load vector
    #should be if have strong bc's and no other terms in residual besides stiffness and load
    #transport model on the final grid
    finest_model = ns.modelList[0].levelModelList[-1]
    #cheating a little bit to get the global test space dimension from the global trial space
    nr = finest_model.u[0].femSpace.dim
    r = np.zeros((nr,),'d')
    f = np.zeros((nr,),'d')
    utmp = np.zeros((nr,),'d')
    finest_model.getResidual(utmp,r)
    finest_model.getLoadVector(f)
    del ns
    np.testing.assert_almost_equal(r,f)

if __name__ == '__main__':
    pass


def test_ncp1_p1nc_velocity(opts):
    """poisson_het_2d_ncp1_n: Crouzeix-Raviart P1 with the p1-nc velocity post-processing (Chou-Tang RT0).

    Every element balances its source against the post-processed face fluxes, and the velocity is within its first
    order error of the exact -a grad u (measured: residual 1.8e-16, largest pointwise error 0.044).
    """
    from . import poisson_het_2d_ncp1_n
    reload(poisson_het_2d_p)
    reload(poisson_het_2d_ncp1_n)
    pList = [poisson_het_2d_p]
    nList = [poisson_het_2d_ncp1_n]
    # the deck sets nn = 21 but star-imports nnx and nny from default_n, and the mesh takes nnx over nn: a value an
    # earlier test leaves there (3 on a CI worker) gives 8 elements and an error of 0.33
    nList[0].nnx = nList[0].nny = nList[0].nn
    so = default_so
    so.name = pList[0].name = "poisson_2d_ncp1_pe{0}".format(repr(Comm.get().size()))
    so.sList = [default_s]
    ns = NumericalSolution.NS_base(so, pList, nList, so.sList, opts)
    ns.calculateSolution(so.name)
    model = ns.modelList[0].levelModelList[-1]
    assert type(model.velocityPostProcessor.vpp_algorithms[0]).__name__ == "VPP_P1nc_RT0"
    q, ebq = model.q, model.ebq
    nOwned = model.mesh.nElements_owned
    source = (-q[('r', 0)] * q[('dV_u', 0)]).sum(axis=1)
    outflow = ((ebq[('velocity', 0)] * ebq['n']).sum(axis=-1) * ebq[('dS_u', 0)]).sum(axis=(1, 2))
    residual = np.abs(source[:nOwned] - outflow[:nOwned]).max()
    exact = pList[0].analyticalSolutionVelocity[0]
    error = max(np.abs(q[('velocity', 0)][eN, k] - exact.uOfX(q['x'][eN, k])).max()
                for eN in range(nOwned) for k in range(q['x'].shape[1]))
    assert residual < 1e-12
    assert error < 0.06
    for ext in ('h5', 'xmf', 'ymf'):
        if os.path.exists(so.name + '.' + ext):
            os.remove(so.name + '.' + ext)
    del ns
