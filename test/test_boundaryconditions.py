"""
Testing module for proteus.BoundaryConditions
Work in progress

TO ADD TO TESTS:
setTwoPhaseVelocityInlet()
setHydrostaticPressureOutlet()
setHydrostaticPressureOutletWithDepth()
wallFunctions()
"""
import os, sys
import random
import unittest
import numpy.testing as npt
import numpy as np
from proteus import Comm, Profiling
from proteus.Profiling import logEvent as log
from proteus.BoundaryConditions import BC_Base, BoundaryCondition
from proteus.BoundaryConditions import BC_RANS


comm = Comm.init()
Profiling.procID = comm.rank()
log("Testing BoundaryConditions")


def create_BC(folder=None, b_or=None, b_i=0, nd=2):
    if folder is None:
        return BC_Base(b_or=b_or, b_i=b_i, nd=nd)
    if folder == 'mprans':
        return BC_RANS(b_or=b_or, b_i=b_i, nd=nd)

def get_random_x(start=0., stop=10.):
    x1 = random.uniform(start, stop)
    x2 = random.uniform(start, stop)
    x3 = random.uniform(start, stop)
    return np.array([x1, x2, x3])

def get_time_array(start=0, stop=5, steps=101):
    return np.linspace(0, 5, steps)

class PseudoContext(object):
    def __init__(self):
        from proteus import Domain
        self.ecH = 3.
        self.he = 0.2

def get_context():
    return PseudoContext()
class TestBC(unittest.TestCase):

    def test_bc_base(self):
        BC = create_BC()
        #BC.newGlobalBC('test1', 4)
        #npt.assert_equal(BC.test1, 4)

    def test_constantBC(self):
        t_list = get_time_array()
        constant = 4
        constants = []
        values = []
        BC_func = lambda x, t: constant
        for t in t_list:
            x = get_random_x()
            constants += [constant]
            values += [BC_func(x, t)]
        npt.assert_equal(values, constants)
    def test_linearBC(self):
        t_list = get_time_array()
        a0 = 1
        a = np.ndarray([1,2,3])
        BC = BoundaryCondition()
        BC.setLinearBC(a0,a)
        for t in t_list:
            x = get_random_x()
            b = BC.uOfXT(x,t)
            npt.assert_equal(b, a0+sum(a*x))
    def test_linearRampBC(self):
        t_list = get_time_array()
        t1 = 3.
        value = 5.
        BC = BoundaryCondition()
        BC.setLinearRamp(t1,value)
        for t in t_list:
            x = get_random_x()
            b = BC.uOfXT(x,t)
            npt.assert_almost_equal(b, min(value, value*t/t1))
    def test_non_material(self):
        BC = create_BC(folder='mprans')
        BC.setNonMaterial()
        vof_adv, u_diff, v_diff, w_diff = [], [], [], []
        t_list = get_time_array()
        for t in t_list:
            x = get_random_x()
            vof_adv += [BC.vof_advective.uOfXT(x, t)]
            u_diff += [BC.u_diffusive.uOfXT(x, t)]
            v_diff += [BC.v_diffusive.uOfXT(x, t)]
            w_diff += [BC.w_diffusive.uOfXT(x, t)]
        zeros = np.zeros(len(t_list))
        npt.assert_equal(vof_adv, zeros)
        npt.assert_equal(u_diff, zeros)
        npt.assert_equal(v_diff, zeros)
        npt.assert_equal(w_diff, zeros)

    def test_mprans_no_slip(self):
        BC = create_BC(folder='mprans')
        BC.setNoSlip()
        u_dir, v_dir, w_dir, p_adv, k_dir,d_dir, d_diff, vof_adv,k_diff = [], [], [], [], [], [], [], [],[]
        us_dir, vs_dir, ws_dir, pInc_adv, pInit_adv, vos_adv = [],[],[],[],[],[]
        pInc_diff = []
        t_list = get_time_array()
        for t in t_list:
            x = get_random_x()
            u_dir += [BC.u_dirichlet.uOfXT(x, t)]
            v_dir += [BC.v_dirichlet.uOfXT(x, t)]
            w_dir += [BC.w_dirichlet.uOfXT(x, t)]
            us_dir += [BC.us_dirichlet.uOfXT(x, t)]
            vs_dir += [BC.vs_dirichlet.uOfXT(x, t)]
            ws_dir += [BC.ws_dirichlet.uOfXT(x, t)]
            p_adv += [BC.p_advective.uOfXT(x, t)]
            pInc_adv += [BC.pInc_advective.uOfXT(x, t)]
            pInit_adv += [BC.pInit_advective.uOfXT(x, t)]
            pInc_diff += [BC.pInc_diffusive.uOfXT(x, t)]
            k_dir += [BC.k_dirichlet.uOfXT(x, t)]
            d_dir += [BC.dissipation_dirichlet.uOfXT(x, t)]
            d_diff += [BC.dissipation_diffusive.uOfXT(x, t)]
            vof_adv += [BC.vof_advective.uOfXT(x, t)]
            vos_adv += [BC.vos_advective.uOfXT(x, t)]
            k_diff += [BC.k_diffusive.uOfXT(x, t)]
        zeros = np.zeros(len(t_list))
        small1 = zeros+1e-10
        small2=zeros+1e-20
        npt.assert_equal(BC.p_dirichlet.uOfXT, None)
        npt.assert_equal(BC.pInc_dirichlet.uOfXT, None)
        npt.assert_equal(BC.pInit_dirichlet.uOfXT, None)
        npt.assert_equal(u_dir, zeros)
        npt.assert_equal(v_dir, zeros)
        npt.assert_equal(w_dir, zeros)
        npt.assert_equal(us_dir, zeros)
        npt.assert_equal(vs_dir, zeros)
        npt.assert_equal(ws_dir, zeros)
        npt.assert_equal(BC.vof_dirichlet.uOfXT, None)
        npt.assert_equal(BC.vos_dirichlet.uOfXT, None)
        npt.assert_equal(k_dir, small2)
        npt.assert_equal(d_dir, small1)
        npt.assert_equal(p_adv, zeros)
        npt.assert_equal(pInc_adv, zeros)
        npt.assert_equal(pInit_adv, zeros)
        npt.assert_equal(BC.u_advective.uOfXT, None)
        npt.assert_equal(BC.v_advective.uOfXT, None)
        npt.assert_equal(BC.w_advective.uOfXT, None)
        npt.assert_equal(BC.us_advective.uOfXT, None)
        npt.assert_equal(BC.vs_advective.uOfXT, None)
        npt.assert_equal(BC.ws_advective.uOfXT, None)
        npt.assert_equal(vof_adv, zeros)
        npt.assert_equal(vos_adv, zeros)
        npt.assert_equal(BC.k_advective.uOfXT, None)
        npt.assert_equal(BC.dissipation_advective.uOfXT, None)
        npt.assert_equal(BC.u_diffusive.uOfXT, None)
        npt.assert_equal(BC.v_diffusive.uOfXT, None)
        npt.assert_equal(BC.w_diffusive.uOfXT, None)
        npt.assert_equal(BC.us_diffusive.uOfXT, None)
        npt.assert_equal(BC.vs_diffusive.uOfXT, None)
        npt.assert_equal(BC.ws_diffusive.uOfXT, None)
        npt.assert_equal(k_diff, zeros)
        npt.assert_equal(pInc_diff, zeros)
        npt.assert_equal(d_diff, zeros)

    def test_mprans_free_slip(self):
        BC = create_BC(folder='mprans')
        BC.setFreeSlip()
        u_adv, v_adv, w_adv,p_adv, u_diff, v_diff, w_diff, k_dir, k_diff,d_dir,d_diff, vof_adv = [], [], [], [], [], [], [], [], [], [],[],[]
        us_adv, vs_adv, ws_adv,pInc_adv,pInit_adv, us_diff, vs_diff, ws_diff, vos_adv = [], [], [], [], [], [], [], [], []
        pInc_diff = []
        t_list = get_time_array()
        for t in t_list:
            x = get_random_x()
            u_adv += [BC.u_advective.uOfXT(x, t)]
            v_adv += [BC.v_advective.uOfXT(x, t)]
            w_adv += [BC.w_advective.uOfXT(x, t)]
            us_adv += [BC.u_advective.uOfXT(x, t)]
            vs_adv += [BC.v_advective.uOfXT(x, t)]
            ws_adv += [BC.w_advective.uOfXT(x, t)]
            p_adv += [BC.p_advective.uOfXT(x, t)]
            pInc_adv += [BC.pInc_advective.uOfXT(x, t)]
            pInit_adv += [BC.pInit_advective.uOfXT(x, t)]
            pInc_diff += [BC.pInit_advective.uOfXT(x, t)]
            u_diff += [BC.u_diffusive.uOfXT(x, t)]
            v_diff += [BC.v_diffusive.uOfXT(x, t)]
            w_diff += [BC.w_diffusive.uOfXT(x, t)]
            us_diff += [BC.us_diffusive.uOfXT(x, t)]
            vs_diff += [BC.vs_diffusive.uOfXT(x, t)]
            ws_diff += [BC.ws_diffusive.uOfXT(x, t)]
            k_dir += [BC.k_dirichlet.uOfXT(x, t)]
            d_dir += [BC.dissipation_dirichlet.uOfXT(x, t)]
            k_diff += [BC.k_diffusive.uOfXT(x, t)]
            d_diff += [BC.dissipation_diffusive.uOfXT(x, t)]
            pInc_diff += [BC.pInc_diffusive.uOfXT(x, t)]
            vof_adv += [BC.vof_advective.uOfXT(x, t)]
            vos_adv += [BC.vos_advective.uOfXT(x, t)]
        zeros = np.zeros(len(t_list))
        small1 = zeros+1e-10
        small2 = zeros+1e-20
        npt.assert_equal(BC.p_dirichlet.uOfXT, None)
        npt.assert_equal(BC.pInc_dirichlet.uOfXT, None)
        npt.assert_equal(BC.pInit_dirichlet.uOfXT, None)
        npt.assert_equal(BC.u_dirichlet.uOfXT, None)
        npt.assert_equal(BC.v_dirichlet.uOfXT, None)
        npt.assert_equal(BC.w_dirichlet.uOfXT, None)
        npt.assert_equal(BC.us_dirichlet.uOfXT, None)
        npt.assert_equal(BC.vs_dirichlet.uOfXT, None)
        npt.assert_equal(BC.ws_dirichlet.uOfXT, None)
        npt.assert_equal(BC.vof_dirichlet.uOfXT, None)
        npt.assert_equal(BC.vos_dirichlet.uOfXT, None)
        npt.assert_equal(k_dir, small2)
        npt.assert_equal(d_dir, small1)
        npt.assert_equal(p_adv, zeros)
        npt.assert_equal(pInc_adv, zeros)
        npt.assert_equal(pInit_adv, zeros)
        npt.assert_equal(u_adv, zeros)
        npt.assert_equal(v_adv, zeros)
        npt.assert_equal(w_adv, zeros)
        npt.assert_equal(us_adv, zeros)
        npt.assert_equal(vs_adv, zeros)
        npt.assert_equal(ws_adv, zeros)
        npt.assert_equal(vof_adv, zeros)
        npt.assert_equal(vos_adv, zeros)
        npt.assert_equal(BC.k_advective.uOfXT, None)
        npt.assert_equal(BC.dissipation_advective.uOfXT, None)
        npt.assert_equal(u_diff, zeros)
        npt.assert_equal(v_diff, zeros)
        npt.assert_equal(w_diff, zeros)
        npt.assert_equal(us_diff, zeros)
        npt.assert_equal(vs_diff, zeros)
        npt.assert_equal(ws_diff, zeros)
        npt.assert_equal(k_diff, zeros)
        npt.assert_equal(d_diff, zeros)
        # check if other BC are None
    def test_constant_inlet_velocity(self):
        BC = create_BC(folder='mprans')
        UU = np.array([0.2,1.,0.1])
        ramp = 2.5
        kk = 1e-3
        dd = 1e-4
        b_or = np.array([0., 1., 0.])
        BC.setConstantInletVelocity(UU, ramp, kk,dd,b_or)
        u_dir, v_dir, w_dir, p_adv, k_dir, d_diff, vof_adv,k_diff = [], [], [], [], [], [], [], []
        us_dir, vs_dir, ws_dir, us_diff, vs_diff, ws_diff, pInc_adv, pInit_adv, vos_adv = [],[],[],[],[],[],[],[],[]
        pInc_diff = []
        t_list = get_time_array()
        for t in t_list:
            x = get_random_x()
            u_dir = [BC.u_dirichlet.uOfXT(x, t)]
            v_dir = [BC.v_dirichlet.uOfXT(x, t)]
            w_dir = [BC.w_dirichlet.uOfXT(x, t)]
            us_dir += [BC.us_dirichlet.uOfXT(x, t)]
            vs_dir += [BC.vs_dirichlet.uOfXT(x, t)]
            ws_dir += [BC.ws_dirichlet.uOfXT(x, t)]
            us_diff += [BC.us_diffusive.uOfXT(x, t)]
            vs_diff += [BC.vs_diffusive.uOfXT(x, t)]
            ws_diff += [BC.ws_diffusive.uOfXT(x, t)]
            p_adv = [BC.p_advective.uOfXT(x, t)]
            pInc_adv = [BC.pInc_advective.uOfXT(x, t)]
            pInit_adv += [BC.pInit_advective.uOfXT(x, t)]
            k_dir = [BC.k_dirichlet.uOfXT(x, t)]
            d_dir = [BC.dissipation_dirichlet.uOfXT(x, t)]
            d_diff += [BC.dissipation_diffusive.uOfXT(x, t)]
            k_diff += [BC.k_diffusive.uOfXT(x, t)]
            npt.assert_almost_equal(u_dir, [min(UU[0],UU[0]*t/ramp)])
            npt.assert_almost_equal(v_dir, [min(UU[1],UU[1]*t/ramp)])
            npt.assert_almost_equal(w_dir,[min(t/ramp, 1.)*UU[2]])
            npt.assert_equal(k_dir, [kk])
            npt.assert_equal(d_dir, [dd])
            npt.assert_almost_equal(p_adv, [min(np.dot(UU,b_or),np.dot(UU,b_or)*t/ramp)])
            npt.assert_almost_equal(pInc_adv, [min(np.dot(UU,b_or),np.dot(UU,b_or)*t/ramp)])
        zeros = np.zeros(len(t_list))
        npt.assert_equal(BC.p_dirichlet.uOfXT, None)
        npt.assert_equal(BC.pInc_dirichlet.uOfXT, None)
        npt.assert_equal(BC.pInit_dirichlet.uOfXT, None)
        npt.assert_equal(us_dir, zeros)
        npt.assert_equal(vs_dir, zeros)
        npt.assert_equal(ws_dir, zeros)
        npt.assert_equal(BC.vof_dirichlet.uOfXT, None)
        npt.assert_equal(BC.vos_dirichlet.uOfXT, None)
        npt.assert_equal(BC.k_advective.uOfXT, None)
        npt.assert_equal(BC.dissipation_advective.uOfXT, None)
        npt.assert_equal(BC.u_diffusive.uOfXT, None)
        npt.assert_equal(BC.v_diffusive.uOfXT, None)
        npt.assert_equal(BC.w_diffusive.uOfXT, None)
        npt.assert_equal(us_diff, zeros)
        npt.assert_equal(vs_diff, zeros)
        npt.assert_equal(ws_diff, zeros)
        npt.assert_equal(k_diff, zeros)
        npt.assert_equal(d_diff, zeros)
    def test_constant_outlet_pressure(self):
        BC = create_BC(folder='mprans')
        p = 1012.
        kk = 1e-3
        dd = 1e-4
        b_or = np.array([0., 1., 0.])
        rho = 1000
        g = np.array([-9.81,0,0])
        BC.setConstantOutletPressure(p,rho,g, kk,dd,b_or)
        u_dir, v_dir, w_dir, p_adv, k_dir, d_diff, vof_adv,k_diff = [], [], [], [], [], [], [], []
        us_dir, vs_dir, ws_dir, u_diff, v_diff, w_diff,us_diff, vs_diff, ws_diff, pInc_adv, pInit_adv, vos_adv,pInc_dir = [],[],[],[],[],[],[],[],[],[],[],[],[]
        pInc_diff = []
        t_list = get_time_array()
        for t in t_list:
            x = get_random_x()
            u_dir += [BC.u_dirichlet.uOfXT(x, t)]
            v_dir += [BC.v_dirichlet.uOfXT(x, t)]
            w_dir += [BC.w_dirichlet.uOfXT(x, t)]
            us_dir += [BC.us_dirichlet.uOfXT(x, t)]
            vs_dir += [BC.vs_dirichlet.uOfXT(x, t)]
            ws_dir += [BC.ws_dirichlet.uOfXT(x, t)]
            if b_or[0] == 1. or b_or[0] == -1.:
                u_diff += [BC.us_diffusive.uOfXT(x, t)]
                v_diff = BC.us_diffusive.uOfXT
                w_diff = BC.us_diffusive.uOfXT
            if b_or[1] == 1. or b_or[1] == -1.:
                v_diff += [BC.vs_diffusive.uOfXT(x, t)]
                u_diff = BC.us_diffusive.uOfXT
                w_diff = BC.us_diffusive.uOfXT
            if b_or[2] == 1. or b_or[2] == -1.:
                w_diff += [BC.ws_diffusive.uOfXT(x, t)]
                u_diff = BC.us_diffusive.uOfXT
                v_diff = BC.us_diffusive.uOfXT
            us_diff += [BC.us_diffusive.uOfXT(x, t)]
            vs_diff += [BC.vs_diffusive.uOfXT(x, t)]
            ws_diff += [BC.ws_diffusive.uOfXT(x, t)]
            p_dir = [BC.p_dirichlet.uOfXT(x,t)]
            pInc_dir += [BC.pInc_dirichlet.uOfXT(x, t)]
            pInit_dir = [BC.pInit_dirichlet.uOfXT(x, t)]
            k_dir = [BC.k_dirichlet.uOfXT(x, t)]
            d_dir = [BC.dissipation_dirichlet.uOfXT(x, t)]
            d_diff += [BC.dissipation_diffusive.uOfXT(x, t)]
            k_diff += [BC.k_diffusive.uOfXT(x, t)]
            npt.assert_almost_equal(p_dir,[p+rho*sum(g*x)])
            npt.assert_almost_equal(pInit_dir,[p+rho*sum(g*x)])
            npt.assert_equal(k_dir, [kk])
            npt.assert_equal(d_dir, [dd])
        zeros = np.zeros(len(t_list))
        if b_or[0] == 1. or b_or[0] == -1.:
            npt.assert_equal(u_dir, zeros)
            npt.assert_equal(BC.v_diffusive.uOfXT, None)
            npt.assert_equal(BC.w_diffusive.uOfXT, None)
        if b_or[1] == 1. or b_or[1] == -1.:
            npt.assert_equal(v_dir, zeros)
            npt.assert_equal(BC.u_diffusive.uOfXT, None)
            npt.assert_equal(BC.w_diffusive.uOfXT, None)
        if b_or[2] == 1. or b_or[2] == -1.:
            npt.assert_equal(w_dir, zeros)
            npt.assert_equal(BC.u_diffusive.uOfXT, None)
            npt.assert_equal(BC.v_diffusive.uOfXT, None)
        npt.assert_equal(BC.p_advective.uOfXT, None)
        npt.assert_equal(BC.pInc_advective.uOfXT, None)
        npt.assert_equal(BC.pInit_advective.uOfXT, None)
        npt.assert_equal(BC.pInc_diffusive.uOfXT, None)
        npt.assert_equal(BC.pInit_diffusive.uOfXT, None)
        npt.assert_equal(us_dir, zeros)
        npt.assert_equal(vs_dir, zeros)
        npt.assert_equal(ws_dir, zeros)
        npt.assert_equal(BC.vof_dirichlet.uOfXT, None)
        npt.assert_equal(BC.vos_dirichlet.uOfXT, None)
        npt.assert_equal(BC.k_advective.uOfXT, None)
        npt.assert_equal(BC.dissipation_advective.uOfXT, None)
        npt.assert_equal(us_diff, zeros)
        npt.assert_equal(vs_diff, zeros)
        npt.assert_equal(ws_diff, zeros)
        npt.assert_equal(k_diff, zeros)
        npt.assert_equal(d_diff, zeros)
    def test_open_air(self):
        # BC = create_BC(folder='mprans')
        BC = create_BC(folder='mprans', b_or=np.array([[0., 0., 1.]]), b_i=0)
        BC.setAtmosphere()
        p_dir, u_dir, v_dir, w_dir, vof_dir, u_diff, v_diff, w_diff, k_diff, k_dir,d_dir,d_diff     = [], [], [], [], [], [], [], [], [], [], [],[]
        pInc_dir, pInit_dir, us_dir, vs_dir, ws_dir, vos_dir, us_diff, vs_diff, ws_diff = [], [], [], [], [], [], [], [], []
        t_list = get_time_array()
        for t in t_list:
            x = get_random_x()
            p_dir += [BC.p_dirichlet.uOfXT(x, t)]
            pInc_dir += [BC.pInc_dirichlet.uOfXT(x, t)]
            pInit_dir += [BC.pInit_dirichlet.uOfXT(x, t)]
            u_dir += [BC.u_dirichlet.uOfXT(x, t)]
            v_dir += [BC.v_dirichlet.uOfXT(x, t)]
            w_dir += [BC.w_dirichlet.uOfXT(x, t)]
            us_dir += [BC.us_dirichlet.uOfXT(x, t)]
            vs_dir += [BC.vs_dirichlet.uOfXT(x, t)]
            ws_dir += [BC.ws_dirichlet.uOfXT(x, t)]
            vof_dir += [BC.vof_dirichlet.uOfXT(x, t)]
            vos_dir += [BC.vos_dirichlet.uOfXT(x, t)]
            # u_diff += [BC.u_diffusive.uOfXT(x, t)]
            # v_diff += [BC.v_diffusive.uOfXT(x, t)]
            w_diff += [BC.w_diffusive.uOfXT(x, t)]
            # us_diff += [BC.us_diffusive.uOfXT(x, t)]
            # vs_diff += [BC.vs_diffusive.uOfXT(x, t)]
            ws_diff += [BC.ws_diffusive.uOfXT(x, t)]
            k_diff += [BC.k_diffusive.uOfXT(x, t)]
            k_dir += [BC.k_dirichlet.uOfXT(x, t)]
            d_dir += [BC.dissipation_dirichlet.uOfXT(x, t)]
            d_diff += [BC.dissipation_diffusive.uOfXT(x, t)]
        zeros = np.zeros(len(t_list))
        vofAir = zeros + 1.
        kdir = zeros + 1e-20
        ddir = zeros + 1e-10
        npt.assert_equal(p_dir, zeros)
        npt.assert_equal(pInc_dir, zeros)
        npt.assert_equal(pInit_dir, zeros)
        npt.assert_equal(u_dir, zeros)
        npt.assert_equal(v_dir, zeros)
        npt.assert_equal(w_dir, zeros)
        npt.assert_equal(us_dir, zeros)
        npt.assert_equal(vs_dir, zeros)
        npt.assert_equal(ws_dir, zeros)
        npt.assert_equal(vof_dir, vofAir)
        npt.assert_equal(k_dir, kdir)
        npt.assert_equal(d_dir, ddir)
        npt.assert_equal(BC.p_advective.uOfXT, None)
        npt.assert_equal(BC.pInc_advective.uOfXT, None)
        npt.assert_equal(BC.pInit_advective.uOfXT, None)
        npt.assert_equal(BC.u_advective.uOfXT, None)
        npt.assert_equal(BC.v_advective.uOfXT, None)
        npt.assert_equal(BC.w_advective.uOfXT, None)
        npt.assert_equal(BC.vof_advective.uOfXT, None)
        npt.assert_equal(BC.k_advective.uOfXT, None)
        npt.assert_equal(BC.dissipation_advective.uOfXT, None)
        npt.assert_equal(BC.u_diffusive.uOfXT, None)
        npt.assert_equal(BC.v_diffusive.uOfXT, None)
        npt.assert_equal(w_diff, zeros)
        npt.assert_equal(k_diff, zeros)
        npt.assert_equal(d_diff, zeros)
        # check if other BC are None
    # def test_unsteady_two_phase_velocity_inlet(self):
        BC = create_BC(folder='mprans', b_or=np.array([[0., 1., 0.]]), b_i=0)
        BC.setAtmosphere()
        p_dir, u_dir, v_dir, w_dir, vof_dir, u_diff, v_diff, w_diff, k_diff, k_dir,d_dir,d_diff     = [], [], [], [], [], [], [], [], [], [], [],[]
        pInc_dir, pInit_dir, us_dir, vs_dir, ws_dir, vos_dir, us_diff, vs_diff, ws_diff = [], [], [], [], [], [], [], [], []
        t_list = get_time_array()
        for t in t_list:
            x = get_random_x()
            p_dir += [BC.p_dirichlet.uOfXT(x, t)]
            pInc_dir += [BC.pInc_dirichlet.uOfXT(x, t)]
            pInit_dir += [BC.pInit_dirichlet.uOfXT(x, t)]
            u_dir += [BC.u_dirichlet.uOfXT(x, t)]
            v_dir += [BC.v_dirichlet.uOfXT(x, t)]
            w_dir += [BC.w_dirichlet.uOfXT(x, t)]
            us_dir += [BC.us_dirichlet.uOfXT(x, t)]
            vs_dir += [BC.vs_dirichlet.uOfXT(x, t)]
            ws_dir += [BC.ws_dirichlet.uOfXT(x, t)]
            vof_dir += [BC.vof_dirichlet.uOfXT(x, t)]
            vos_dir += [BC.vos_dirichlet.uOfXT(x, t)]
            # u_diff += [BC.u_diffusive.uOfXT(x, t)]
            v_diff += [BC.v_diffusive.uOfXT(x, t)]
            # w_diff += [BC.w_diffusive.uOfXT(x, t)]
            # us_diff += [BC.us_diffusive.uOfXT(x, t)]
            vs_diff += [BC.vs_diffusive.uOfXT(x, t)]
            # ws_diff += [BC.ws_diffusive.uOfXT(x, t)]
            k_diff += [BC.k_diffusive.uOfXT(x, t)]
            k_dir += [BC.k_dirichlet.uOfXT(x, t)]
            d_dir += [BC.dissipation_dirichlet.uOfXT(x, t)]
            d_diff += [BC.dissipation_diffusive.uOfXT(x, t)]
        zeros = np.zeros(len(t_list))
        vofAir = zeros + 1.
        kdir = zeros + 1e-20
        ddir = zeros + 1e-10
        npt.assert_equal(p_dir, zeros)
        npt.assert_equal(pInc_dir, zeros)
        npt.assert_equal(pInit_dir, zeros)
        npt.assert_equal(u_dir, zeros)
        npt.assert_equal(v_dir, zeros)
        npt.assert_equal(w_dir, zeros)
        npt.assert_equal(us_dir, zeros)
        npt.assert_equal(vs_dir, zeros)
        npt.assert_equal(ws_dir, zeros)
        npt.assert_equal(vof_dir, vofAir)
        npt.assert_equal(k_dir, kdir)
        npt.assert_equal(d_dir, ddir)
        npt.assert_equal(BC.p_advective.uOfXT, None)
        npt.assert_equal(BC.pInc_advective.uOfXT, None)
        npt.assert_equal(BC.pInit_advective.uOfXT, None)
        npt.assert_equal(BC.u_advective.uOfXT, None)
        npt.assert_equal(BC.v_advective.uOfXT, None)
        npt.assert_equal(BC.w_advective.uOfXT, None)
        npt.assert_equal(BC.vof_advective.uOfXT, None)
        npt.assert_equal(BC.k_advective.uOfXT, None)
        npt.assert_equal(BC.dissipation_advective.uOfXT, None)
        npt.assert_equal(BC.u_diffusive.uOfXT, None)
        npt.assert_equal(BC.w_diffusive.uOfXT, None)
        npt.assert_equal(v_diff, zeros)
        npt.assert_equal(k_diff, zeros)
        npt.assert_equal(d_diff, zeros)
        # other BC orientation
        BC = create_BC(folder='mprans', b_or=np.array([[1., 0., 0.]]), b_i=0)
        BC.setAtmosphere()
        p_dir, u_dir, v_dir, w_dir, vof_dir, u_diff, v_diff, w_diff, k_diff, k_dir,d_dir,d_diff     = [], [], [], [], [], [], [], [], [], [], [],[]
        pInc_dir, pInit_dir, us_dir, vs_dir, ws_dir, vos_dir, us_diff, vs_diff, ws_diff = [], [], [], [], [], [], [], [], []
        t_list = get_time_array()
        for t in t_list:
            x = get_random_x()
            p_dir += [BC.p_dirichlet.uOfXT(x, t)]
            pInc_dir += [BC.pInc_dirichlet.uOfXT(x, t)]
            pInit_dir += [BC.pInit_dirichlet.uOfXT(x, t)]
            u_dir += [BC.u_dirichlet.uOfXT(x, t)]
            v_dir += [BC.v_dirichlet.uOfXT(x, t)]
            w_dir += [BC.w_dirichlet.uOfXT(x, t)]
            us_dir += [BC.us_dirichlet.uOfXT(x, t)]
            vs_dir += [BC.vs_dirichlet.uOfXT(x, t)]
            ws_dir += [BC.ws_dirichlet.uOfXT(x, t)]
            vof_dir += [BC.vof_dirichlet.uOfXT(x, t)]
            vos_dir += [BC.vos_dirichlet.uOfXT(x, t)]
            u_diff += [BC.u_diffusive.uOfXT(x, t)]
            # v_diff += [BC.v_diffusive.uOfXT(x, t)]
            # w_diff += [BC.w_diffusive.uOfXT(x, t)]
            us_diff += [BC.us_diffusive.uOfXT(x, t)]
            # vs_diff += [BC.vs_diffusive.uOfXT(x, t)]
            # ws_diff += [BC.ws_diffusive.uOfXT(x, t)]
            k_diff += [BC.k_diffusive.uOfXT(x, t)]
            k_dir += [BC.k_dirichlet.uOfXT(x, t)]
            d_dir += [BC.dissipation_dirichlet.uOfXT(x, t)]
            d_diff += [BC.dissipation_diffusive.uOfXT(x, t)]
        zeros = np.zeros(len(t_list))
        vofAir = zeros + 1.
        kdir = zeros + 1e-20
        sdir = zeros + 1e-10
        npt.assert_equal(p_dir, zeros)
        npt.assert_equal(pInc_dir, zeros)
        npt.assert_equal(pInit_dir, zeros)
        npt.assert_equal(u_dir, zeros)
        npt.assert_equal(v_dir, zeros)
        npt.assert_equal(w_dir, zeros)
        npt.assert_equal(us_dir, zeros)
        npt.assert_equal(vs_dir, zeros)
        npt.assert_equal(ws_dir, zeros)
        npt.assert_equal(vof_dir, vofAir)
        npt.assert_equal(k_dir, kdir)
        npt.assert_equal(d_dir, ddir)
        npt.assert_equal(BC.p_advective.uOfXT, None)
        npt.assert_equal(BC.pInc_advective.uOfXT, None)
        npt.assert_equal(BC.pInit_advective.uOfXT, None)
        npt.assert_equal(BC.u_advective.uOfXT, None)
        npt.assert_equal(BC.v_advective.uOfXT, None)
        npt.assert_equal(BC.w_advective.uOfXT, None)
        npt.assert_equal(BC.vof_advective.uOfXT, None)
        npt.assert_equal(BC.k_advective.uOfXT, None)
        npt.assert_equal(BC.dissipation_advective.uOfXT, None)
        npt.assert_equal(BC.v_diffusive.uOfXT, None)
        npt.assert_equal(BC.w_diffusive.uOfXT, None)
        npt.assert_equal(u_diff, zeros)
        npt.assert_equal(k_diff, zeros)
        npt.assert_equal(d_diff, zeros)

    def test_set_tank(self):
        BC = create_BC(folder='mprans', b_or=np.array([[0., 1., 0.]]), b_i=0)
        BC.setTank()
        # checking if other BC leaves setTank BC as it should be
        BC.setFreeSlip()
        BC.reset()
        # ----
        t_list = get_time_array()
        hy_dir, u_stress, w_stress = [], [], []
        for t in t_list:
            x = get_random_x()
            hy_dir += [BC.hy_dirichlet.uOfXT(x, t)]
            u_stress += [BC.u_stress.uOfXT]
            w_stress += [BC.w_stress.uOfXT]
        zeros = np.zeros(len(t_list))
        npt.assert_equal(BC.hx_dirichlet.uOfXT, None)
        npt.assert_equal(hy_dir, zeros)
        npt.assert_equal(BC.hz_dirichlet.uOfXT, None)
        npt.assert_equal(hy_dir, zeros)
        npt.assert_equal(u_stress, zeros)
        npt.assert_equal(BC.v_stress.uOfXT, None)
        npt.assert_equal(w_stress, zeros)

    def test_move_mesh(self):
        BC = create_BC(folder='mprans')
        last_pos = np.array([1., 1., 1.])
        h = np.array([0., 0., 0.])
        rot_matrix = np.eye(3)
        BC.setMoveMesh(last_pos=last_pos, h=h, rot_matrix=rot_matrix)
        # checking if other BC leaves setTank BC as it should be
        BC.setFreeSlip()
        BC.reset()
        # ----
        t_list = get_time_array()
        hx_dir, hy_dir, hz_dir = [], [], []
        displacement = []
        from proteus.SpatialTools import rotation3D
        for t in t_list:
            new_pos = np.array(get_random_x())
            h[:] = new_pos-last_pos
            rot_angle = get_random_x()[0]
            rot_axis = get_random_x()
            rot_matrix[:] = rotation3D(rot_matrix, rot=rot_angle, axis=rot_axis)
            x = np.array(get_random_x())
            hx_dir += [BC.hx_dirichlet.uOfXT(x, t)]
            hy_dir += [BC.hy_dirichlet.uOfXT(x, t)]
            hz_dir += [BC.hz_dirichlet.uOfXT(x, t)]
            x0 = x-last_pos
            displacement += [(np.dot(x0, rot_matrix)-x0)+h]
            last_pos[:] = x
        zeros = np.zeros(len(t_list))
        displacement = np.array(displacement)
        npt.assert_equal(hx_dir, displacement[:, 0])
        npt.assert_equal(hy_dir, displacement[:, 1])
        npt.assert_equal(hz_dir, displacement[:, 2])

    def test_unsteady_two_phase_velocity_inlet(self):
        from proteus.WaveTools import MonochromaticWaves
        b_or = np.array([[0., -1., 0.]])
        b_i = 0
        BC = create_BC(folder='mprans', b_or=b_or, b_i=b_i)
        # creating a wave
        period = 0.8
        height = 0.029
        mwl = depth = 0.9
        direction = np.array([1., 0., 0.])
        g = np.array([0., -9.81, 0.])
        waves = MonochromaticWaves(period, height, mwl, depth, g, direction)
        # need to set epsFact and he with context as they are called in BC...
        ct = get_context()
        from proteus.ctransportCoefficients import smoothedHeaviside
        #-----
        # set BC
        wind_speed=np.array([1., 2., 3.4])
        smoothing = 0.
        BC.setUnsteadyTwoPhaseVelocityInlet(waves, smoothing, vert_axis=1,
                                            wind_speed=wind_speed)
        BC.getContext(ct)
        BC.u_dirichlet.uOfXT = BC.u_dirichlet.init_cython()
        BC.v_dirichlet.uOfXT = BC.v_dirichlet.init_cython()
        BC.w_dirichlet.uOfXT = BC.w_dirichlet.init_cython()
        BC.vof_dirichlet.uOfXT = BC.vof_dirichlet.init_cython()
        BC.p_advective.uOfXT = BC.p_advective.init_cython()
        BC.pInit_advective.uOfXT = BC.pInit_advective.init_cython()
        u_dir, v_dir, w_dir, vof_dir, p_adv = [], [], [], [], []
        pInc_adv, pInit_adv, us_dir, vs_dir,ws_dir, vof_dir, vos_dir = [], [], [], [], [],[],[]
        u_calc, vof_calc, p_calc = [], [], []
        k_dir,d_dir, k_dif,d_dif = [],[],[],[]
        t_list = get_time_array()
        zeros = np.zeros(len(t_list))
        for t in t_list:
            x = np.array(get_random_x())
            u_dir += [BC.u_dirichlet.uOfXT(x, t)]
            v_dir += [BC.v_dirichlet.uOfXT(x, t)]
            w_dir += [BC.w_dirichlet.uOfXT(x, t)]
            k_dir += [BC.k_dirichlet.uOfXT(x, t)]
            d_dir += [BC.dissipation_dirichlet.uOfXT(x, t)]
            k_dif += [BC.k_diffusive.uOfXT(x, t)]
            d_dif += [BC.dissipation_diffusive.uOfXT(x, t)]
            us_dir += [BC.us_dirichlet.uOfXT(x, t)]
            vs_dir += [BC.vs_dirichlet.uOfXT(x, t)]
            ws_dir += [BC.ws_dirichlet.uOfXT(x, t)]
            vof_dir += [BC.vof_dirichlet.uOfXT(x, t)]
            vos_dir += [BC.vos_dirichlet.uOfXT(x, t)]
            p_adv += [BC.p_advective.uOfXT(x, t)]
            pInc_adv += [BC.pInc_advective.uOfXT(x, t)]
            pInit_adv += [BC.pInit_advective.uOfXT(x, t)]
            # calculations
            waveHeight = waves.mwl+waves.eta(x, t)
            wavePhi = x[1]-waveHeight
            if wavePhi <= 0:
                H = 0.
                wave_u = waves.u(x, t)
            elif smoothing > 0 and 0 < wavePhi <= smoothing:
                H = smoothedHeaviside(smoothing, wavePhi-0.5*smoothing)
                x_max = list(x)
                x_max[1] = waveHeight
                wave_u = waves.u(x_max, t)
            else:
                H = 1.
                wave_u = np.array([0., 0., 0.])
            U = H*wind_speed + (1-H)*wave_u
            u_calc += [U]
            p_calc += [np.sum(U*b_or[b_i])]
            if wavePhi >= smoothing/2.:
                Hvof = 1.
            elif smoothing > 0 and -smoothing/2. < wavePhi < smoothing/2.:
                Hvof = smoothedHeaviside(smoothing, wavePhi)
            elif wavePhi <= -smoothing/2.:
                Hvof = 0.
            vof_calc += [Hvof]
        u_calc = np.array(u_calc)
        vof_calc = np.array(vof_calc)
        kInf = 1e-30+zeros
        dInf = 1e-10 +zeros
        npt.assert_equal(BC.p_dirichlet.uOfXT, None)
        npt.assert_equal(BC.pInc_dirichlet.uOfXT, None)
        npt.assert_equal(BC.pInit_dirichlet.uOfXT, None)
        npt.assert_equal(u_dir, u_calc[:, 0])
        npt.assert_equal(v_dir, u_calc[:, 1])
        npt.assert_equal(w_dir, u_calc[:, 2])
        npt.assert_equal(us_dir, zeros)
        npt.assert_equal(vs_dir, zeros)
        npt.assert_equal(ws_dir, zeros)
        npt.assert_equal(vof_dir, vof_calc)
        npt.assert_equal(vos_dir, zeros)
        npt.assert_equal(k_dir, kInf)
        npt.assert_equal(d_dir, dInf)
        npt.assert_equal(p_adv, p_calc)
        npt.assert_equal(BC.u_advective.uOfXT, None)
        npt.assert_equal(BC.v_advective.uOfXT, None)
        npt.assert_equal(BC.w_advective.uOfXT, None)
        npt.assert_equal(BC.us_advective.uOfXT, None)
        npt.assert_equal(BC.vs_advective.uOfXT, None)
        npt.assert_equal(BC.ws_advective.uOfXT, None)
        npt.assert_equal(BC.vof_advective.uOfXT, None)
        npt.assert_equal(BC.vos_advective.uOfXT, None)
        npt.assert_equal(BC.k_advective.uOfXT,None )
        npt.assert_equal(BC.dissipation_advective.uOfXT, None)
        npt.assert_equal(BC.u_diffusive.uOfXT, None)
        npt.assert_equal(BC.v_diffusive.uOfXT, None)
        npt.assert_equal(BC.w_diffusive.uOfXT, None)
        npt.assert_equal(BC.us_diffusive.uOfXT, None)
        npt.assert_equal(BC.vs_diffusive.uOfXT, None)
        npt.assert_equal(BC.ws_diffusive.uOfXT, None)
        npt.assert_equal(k_dif, zeros)
        npt.assert_equal(d_dif, zeros)

    def test_two_phase_velocity_inlet(self):
        from proteus.ctransportCoefficients import smoothedHeaviside
        # input 
        ct = get_context()
        b_or = np.array([[0., -1., 0.]])
        b_i = 0
        U0 = [0.1, 0.2, 0.3] # m/s
        waterDepth = 0.5 # m
        smoothing = 3.*0.01 # m
        Uwind =  [0.01, 0.02, 0.03] # m/s
        vert_axis = 1 # by default alligned with the gravity
        air = 1.
        water = 0.
        kInflow = 0.00005
        dissipationInflow = 0.00001
        kInflowAir = kInflow/10.
        dissipationInflowAir = dissipationInflow/10.
        BC = create_BC(folder='mprans', b_or=b_or, b_i=b_i)
        # setting variables
        uDir, vDir, wDir, vofDir, pAdv, kDir, dissipationDir = [],[],[],[],[],[],[]
        uCalc, vCalc, wCalc, vofCalc, pCalc, kCalc, dissipationCalc = [],[],[],[],[],[],[]
        t_list = get_time_array()
        BC.setTwoPhaseVelocityInlet(U0, waterDepth, smoothing, Uwind, 
                                    vert_axis, air, water,
                                    kInflow, dissipationInflow,
                                    kInflowAir, dissipationInflowAir)
        # time step iterations
        for t in t_list:
            x = np.array(get_random_x())
            uDir += [BC.u_dirichlet.uOfXT(x, t)]
            vDir += [BC.v_dirichlet.uOfXT(x, t)]
            wDir += [BC.w_dirichlet.uOfXT(x, t)]
            vofDir += [BC.vof_dirichlet.uOfXT(x, t)]
            pAdv += [BC.p_advective.uOfXT(x, t)]
            kDir += [BC.k_dirichlet.uOfXT(x, t)]
            dissipationDir += [BC.dissipation_dirichlet.uOfXT(x, t)]
            phiCalc = x[vert_axis] - waterDepth
            # smoothing for velocity, kappa, dissipation field activated only along the 'air phase' side
            if phiCalc <= 0.: 
                Heav = 0.
            elif 0. < phiCalc <= smoothing: 
                Heav = smoothedHeaviside(smoothing, phiCalc - smoothing/2.)
            else: 
                Heav = 1.
            u, v, w = Heav*np.array(Uwind) + (1.-Heav)*np.array(U0)
            uCalc += [u]
            vCalc += [v]
            wCalc += [w]
            up = np.sqrt( (u**2)*abs(b_or[0][0])+(v**2)*abs(b_or[0][1])+(w**2)*abs(b_or[0][2]) )
            pCalc += [-up]
            kCalc += [Heav*kInflowAir + (1.-Heav)*kInflow]
            dissipationCalc += [Heav*dissipationInflowAir + (1.-Heav)*dissipationInflow]
            # smoothing for vof activated along either the water-phase and air phase side
            if phiCalc <= -smoothing: 
                Heav = 0.
            elif -smoothing < phiCalc < smoothing: 
                Heav = smoothedHeaviside(smoothing, phiCalc)
            else: 
                Heav = 1.
            vofCalc += [Heav*air + (1.-Heav)*water]
        npt.assert_equal(uDir, uCalc)
        npt.assert_equal(vDir, vCalc)
        npt.assert_equal(wDir, wCalc)
        npt.assert_equal(vofDir, vofCalc)
        npt.assert_equal(BC.p_dirichlet.uOfXT, None)
        npt.assert_equal(kDir, kCalc)
        npt.assert_equal(dissipationDir, dissipationCalc)
        npt.assert_equal(pAdv, pCalc)
            
    def test_hydrostatic_pressure_outlet_with_depth(self):
        from proteus.ctransportCoefficients import smoothedHeaviside, smoothedHeaviside_integral
        # input 
        ct = get_context()
        b_or = np.array([[0., -1., 0.]])
        b_i = 0
        seaLevel = 0.5 # m
        rhoUp = 1.004e-6 # kg/m3
        rhoDown = 1.500e-5 # kg/m3
        g = np.array([0., -9.81, 0.]) # m/s2
        refLevel = seaLevel
        smoothing = 3.*0.01 # m
        nd = 2
        vert_axis = nd - 1
        air = 1.
        water = 0.
        pRef = 0. # Pa
        BC = create_BC(folder='mprans', b_or=b_or, b_i=b_i)
        # setting variables
        uDir, vDir, wDir, vofDir, pDir, vDiff, kDiff, dissDiff = [],[],[],[],[],[],[],[]
        pInc_dir, pInit_dir = [],[]
        vofCalc, pCalc = [],[]
        t_list = get_time_array()
        BC.setHydrostaticPressureOutletWithDepth(seaLevel, rhoUp, rhoDown, g, refLevel, smoothing)
        # time step iterations
        for t in t_list:
            x = np.array(get_random_x())
            vofDir += [BC.vof_dirichlet.uOfXT(x, t)]
            pDir += [BC.p_dirichlet.uOfXT(x, t)]
            # Relative system of coordinates based on the point chosen as reference with pressure=pRef
            phiCalc = x[vert_axis] - seaLevel
            phi_top = refLevel - seaLevel
            phi_ref = phi_top - phiCalc
            rho_diff = rhoUp - rhoDown
            phi_diff = smoothedHeaviside_integral(smoothing, phi_top) - smoothedHeaviside_integral(smoothing, phiCalc)
            pTot = pRef - (g[vert_axis]*rhoDown*phi_ref) - (g[vert_axis]*rho_diff*phi_diff)
            pCalc += [pTot]
            # smoothing for vof activated along either the water-phase and air phase side
            if phiCalc <= -smoothing: 
                Heav = 0.
            elif -smoothing < phiCalc < smoothing: 
                Heav = smoothedHeaviside(smoothing, phiCalc)
            else: 
                Heav = 1.
            vofCalc += [Heav*air + (1.-Heav)*water]
            # Velocity and turbulence variables
            uDir += [BC.u_dirichlet.uOfXT(x, t)]
            vDir += [BC.v_dirichlet.uOfXT]
            wDir += [BC.w_dirichlet.uOfXT(x, t)]
            vDiff += [BC.v_diffusive.uOfXT(x,t)]
            kDiff += [BC.k_diffusive.uOfXT(x, t)]
            dissDiff += [BC.dissipation_diffusive.uOfXT(x, t)]
        nt = len(t_list)
        uCalc, vCalc, wCalc, vDiffCalc, kCalc, dissCalc = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
        npt.assert_equal(uDir, uCalc)
        npt.assert_equal(BC.v_dirichlet.uOfXT, None)
        npt.assert_equal(wDir, wCalc)
        npt.assert_allclose(pDir, pCalc, atol=1e-10)
        npt.assert_equal(vofDir, vofCalc)
        npt.assert_equal(vDiff, vDiffCalc)
        npt.assert_equal(kDiff, kCalc)
        npt.assert_equal(dissDiff, dissCalc)

    def test_wall_functions(self):
        from proteus import BoundaryConditions as bc
        mbc = bc  # the two-phase classes were merged into proteus.BoundaryConditions (C49 R1)
        # input 
        ct = get_context() 
        b_or = np.array([[0., -1., 0.]])
        b_or_wall = np.array([0., -1., 0.])
        b_i = 0
        BC = create_BC(folder='mprans', b_or=b_or, b_i=b_i)
        Y = 0.01 # m
        U0 = np.array( [0.1, 0., 0.]) # m/s
        U0abs = np.sqrt(np.sum(U0**2))
        Cmu = 0.09
        B = 5.57
        # Normal and tangential vectors
        nV = -b_or_wall/np.sqrt(np.sum(b_or_wall**2))
        uTan = U0 - U0*(nV**2)
        tV = uTan/np.sqrt(np.sum(uTan**2))
        uTanAbs = np.sqrt(np.sum(uTan**2))
        # Calculation of turbulent variables
        Re0 = U0abs * Y / 1.004e-6
        cf = 0.045/(Re0**0.25)
        ut = U0abs * np.sqrt(cf/2.)
        Yplus = Y*ut/1.004e-6
        turbModel = 'ke' # 'kw'
        kappaP = (ut**2)/(Cmu**0.5)
        if turbModel == 'ke':
            dissipationP = (ut**3)/(0.41*Y) # ke model
        elif turbModel == 'kw':
            dissipationP = np.sqrt(kappaP)/(0.41*Y*(Cmu**0.25)) # kw model
        # Log law
        E = np.exp(0.41*B)
        utStar = (kappaP**0.5)*(0.09**0.25)
        uStar = utStar * np.log(E*Yplus) / 0.41
        ut = utStar * np.sqrt(uTanAbs/uStar)
        gradU = (ut/0.41/Y) * tV
        uDir = uTan - gradU*Y
        # Wall objects
        kWall = mbc.kWall(Y=Y, Yplus=Yplus)
        wall = mbc.WallFunctions(turbModel=turbModel, kWall=kWall, Y=Y, Yplus=Yplus, U0=U0)
        kWall.model = None
        wall.model = None
        # Boundary conditions
        x = np.array(get_random_x())  
        t = random.uniform(0., 10.)
        BC.setWallFunction(wall)
        npt.assert_allclose(BC.u_dirichlet.uOfXT(x, t, b_or_wall), uDir[0], atol=1e-10)
        npt.assert_allclose(BC.v_dirichlet.uOfXT(x, t, b_or_wall), uDir[1], atol=1e-10)
        npt.assert_allclose(BC.w_dirichlet.uOfXT(x, t, b_or_wall), uDir[2], atol=1e-10)
        npt.assert_allclose(BC.k_dirichlet.uOfXT(x, t, b_or_wall), kappaP, atol=1e-10)
        npt.assert_allclose(BC.dissipation_dirichlet.uOfXT(x, t, b_or_wall), dissipationP, atol=1e-10)
        



if __name__ == '__main__':

    unittest.main(verbosity=2)
