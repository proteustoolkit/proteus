#!python
# distutils: language = c++
# cython: profile=True, binding=True, embedsignature=True
# cython: wraparound=False
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: language_level=3
import cython
import os, sys
import numpy as np
"""
Module for creating boundary conditions: the generic classes (BC_Base, BoundaryCondition) and the two-phase
flow ones (BC_RANS, RelaxationZone, RelaxationZoneWaveGenerator, WallFunctions, kWall; formerly
proteus.kernels.BoundaryConditions, merged here by C49 R1). Imported in SpatialTools.py

.. inheritance-diagram:: proteus.BoundaryConditions
   :parts: 1
"""
from proteus import AuxiliaryVariables, Comm
from proteus.ctransportCoefficients import (smoothedHeaviside,
                                            smoothedHeaviside_integral)
from proteus import WaveTools as wt
from proteus.Profiling import logEvent
from math import cos, sin, sqrt, atan2, acos, asin
from mpi4py import MPI
from scipy import spatial

__all__ = ['BC_Base',
           'BoundaryCondition',
           'BC_RANS',
           'RelaxationZone',
           'RelaxationZoneWaveGenerator',
           '_cppClass_WavesCharacteristics',
           'WallFunctions',
           'kWall']

class BC_Base:
    """
    Generic class regrouping boundary conditions
    """
    def __init__(self, shape=None, name=None, b_or=None, b_i=0, nd=None):
        self.Shape = shape
        self.name = name
        self.BC_type = 'None'
        if shape is not None:
          self.nd = self.Shape.Domain.nd
        elif nd is not None:
            self.nd = nd
        else:
            assert nd is not None, 'Shape or nd must be passed to BC'
        if b_or is not None:
            self._b_or = b_or[b_i]  # array of orientation of all boundaries of shape
        else:
            self._b_or = None

    # @staticmethod
    # def newGlobalBC(name, default_value):
    #     """
    #     Makes a new BoundaryCondition class instance attached to BC_Base.
    #     This creates a new class attributed that will be added to all BC_Base
    #     class instances.

    #     Parameters
    #     ----------
    #     name: string
    #     """
    #     setattr(BC_Base, name, default_value)

    def getContext(self, context=None):
        """
        Gets context from proteus.Context or

        Parameters
        ----------
        context: class, optional
             if set to None, the context will be created from proteus.Context
        """
        if context:
            from proteus import Context
            self.ct = Context.get()
        else:
            self.ct = context


class BoundaryCondition:
    """
    Boundary condition class

    Attributes
    ----------
    uOfXT: func or None
        boundary condition function of x (array_like) and t (float) or None for
        no boundary condition
    """
    def __init__(self):
        self.uOfXT = None

    def init_cython(self):
        return self.uOfXT

    def resetBC(self):
        self.uOfXT = None

    def setConstantBC(self, value):
        """
        function returning constant BC

        Parameters
        ----------
        value : float
                Constant value

        """
        self.uOfXT = lambda x, t, n=np.zeros(3,): value


    def setLinearBC(self, a0, a):
        """
        function returning value=a0+ax*x+ay*y+az*z

        Parameters
        ----------
        a0 : float
             constant
        a: numpy.ndarray
            ax,ay,az

        """
        
        self.uOfXT = lambda x, t, n=np.zeros(3,): a0+sum(a[:]*x[:])

    def setLinearRamp(self,t1,value):
        """
        function setting a linear ramp from t=0 to t=t1

        Parameters
        -----------
        t1: float
            Ramp end time
        value: float
            Variable value
        """
        self.uOfXT = lambda x, t, n=np.zeros(3,): min( t/t1, 1)*value


class BC_RANS(BC_Base):
    """
    Class regrouping boundary conditions for two-phase flows
    """

    def __init__(self, shape=None, name=None, b_or=None, b_i=0., nd=None):
        super(BC_RANS, self).__init__(shape=shape, name=name, b_or=b_or, b_i=b_i, nd=nd)
        # _dirichlet
        self.p_dirichlet = BoundaryCondition()  # pressure
        self.u_dirichlet = BoundaryCondition()  # velocity u
        self.v_dirichlet = BoundaryCondition()  # velocity v
        self.w_dirichlet = BoundaryCondition()  # velocity w
        self.phi_dirichlet = BoundaryCondition()  # phi
        self.vof_dirichlet = BoundaryCondition()  # VOF
        self.k_dirichlet = BoundaryCondition()  # kappa
        self.dissipation_dirichlet = BoundaryCondition()  # dissipation
        self.pAddedMass_dirichlet = BoundaryCondition()
        # _advective
        self.p_advective = BoundaryCondition()
        self.u_advective = BoundaryCondition()
        self.v_advective = BoundaryCondition()
        self.w_advective = BoundaryCondition()
        self.vof_advective = BoundaryCondition()
        self.k_advective = BoundaryCondition()
        self.dissipation_advective = BoundaryCondition()
        # _diffusive
        self.u_diffusive = BoundaryCondition()
        self.v_diffusive = BoundaryCondition()
        self.w_diffusive = BoundaryCondition()
        self.k_diffusive = BoundaryCondition()
        self.dissipation_diffusive = BoundaryCondition()
        # moveMesh boundary conditions
        self.hx_dirichlet = BoundaryCondition()
        self.hy_dirichlet = BoundaryCondition()
        self.hz_dirichlet = BoundaryCondition()
        self.u_stress = BoundaryCondition()
        self.v_stress = BoundaryCondition()
        self.w_stress = BoundaryCondition()
        self.u_stress.uOfXT = 0.
        self.v_stress.uOfXT = 0.
        self.w_stress.uOfXT = 0.
        # sediment solver
        self.us_dirichlet = BoundaryCondition()  # sediment velocity u
        self.vs_dirichlet = BoundaryCondition()  # sediment velocity v
        self.ws_dirichlet = BoundaryCondition()  # sediment velocity w
        self.vos_dirichlet = BoundaryCondition()  # VOS
        self.us_advective = BoundaryCondition()  
        self.vs_advective = BoundaryCondition()  
        self.ws_advective = BoundaryCondition()  
        self.vos_advective = BoundaryCondition() 
        self.us_diffusive = BoundaryCondition()   
        self.vs_diffusive = BoundaryCondition()  
        self.ws_diffusive = BoundaryCondition() 
        # projection scheme
        self.pInit_dirichlet = BoundaryCondition() # initial pressure
        self.pInc_dirichlet = BoundaryCondition() # pressure increment
        self.pInit_advective = BoundaryCondition() 
        self.pInc_advective = BoundaryCondition() 
        self.pInit_diffusive = BoundaryCondition() 
        self.pInc_diffusive = BoundaryCondition() 
        # clsvof
        self.clsvof_dirichlet = BoundaryCondition()
        self.clsvof_advective = BoundaryCondition()
        self.clsvof_diffusive = BoundaryCondition()

    def reset(self):
        """
        Resets all BoundaryCondtion functions to None, apart from the BCs
        affecting: moving mesh
        """
        # self.BC_type = 'None'
        self.p_dirichlet.resetBC()
        self.pInit_dirichlet.resetBC()
        self.pInc_dirichlet.resetBC()
        self.u_dirichlet.resetBC()
        self.v_dirichlet.resetBC()
        self.w_dirichlet.resetBC()
        self.phi_dirichlet.resetBC()
        self.vof_dirichlet.resetBC()
        self.k_dirichlet.resetBC()
        self.dissipation_dirichlet.resetBC()
        self.us_dirichlet.resetBC()
        self.vs_dirichlet.resetBC()
        self.ws_dirichlet.resetBC()
        self.vos_dirichlet.resetBC()
        self.p_advective.resetBC()
        self.pInit_advective.resetBC()
        self.pInc_advective.resetBC()
        self.u_advective.resetBC()
        self.v_advective.resetBC()
        self.w_advective.resetBC()
        self.vof_advective.resetBC()
        self.k_advective.resetBC()
        self.dissipation_advective.resetBC()
        self.us_advective.resetBC()
        self.vs_advective.resetBC()
        self.ws_advective.resetBC()
        self.vos_advective.resetBC()
        self.pInit_diffusive.resetBC()
        self.pInc_diffusive.resetBC()
        self.u_diffusive.resetBC()
        self.v_diffusive.resetBC()
        self.w_diffusive.resetBC()
        self.k_diffusive.resetBC()
        self.dissipation_diffusive.resetBC()
        self.us_diffusive.resetBC()
        self.vs_diffusive.resetBC()
        self.ws_diffusive.resetBC()
        self.clsvof_dirichlet.resetBC()
        self.clsvof_advective.resetBC()
        self.clsvof_diffusive.resetBC()

    def setNonMaterial(self):
        """
        Sets non-material boundary conditions (diffusive flux and advective vof
        to 0.).
        """
        self.reset()
        self.BC_type = 'NonMaterial'
        self.vof_advective.setConstantBC(0.)
        self.vos_advective.setConstantBC(0.)
        self.p_advective.setConstantBC(0.)
        self.u_advective.setConstantBC(0.)
        self.v_advective.setConstantBC(0.)
        self.w_advective.setConstantBC(0.)
        self.u_diffusive.setConstantBC(0.)
        self.v_diffusive.setConstantBC(0.)
        self.w_diffusive.setConstantBC(0.)
        self.us_diffusive.setConstantBC(0.)
        self.vs_diffusive.setConstantBC(0.)
        self.ws_diffusive.setConstantBC(0.)
        self.k_diffusive.setConstantBC(0.)
        self.dissipation_diffusive.setConstantBC(0.)
        self.pInc_diffusive.setConstantBC(0.)

    def setTank(self, b_or=None):
        if b_or is None:
            assert self._b_or is not None, 'Boundary orientation must be defined!'
            b_or = self._b_or
        self.u_stress.uOfXT = 0.
        self.v_stress.uOfXT = 0.
        self.w_stress.uOfXT = 0.
        self.hx_dirichlet.resetBC()
        self.hy_dirichlet.resetBC()
        self.hz_dirichlet.resetBC()
        if b_or[0] == 1 or b_or[0] == -1:
            self.hx_dirichlet.setConstantBC(0.)
            self.u_stress.uOfXT = None
        elif b_or[1] == 1 or b_or[1] == -1:
            self.hy_dirichlet.setConstantBC(0.)
            self.v_stress.uOfXT = None
        elif len(b_or) > 2 and (b_or[2] == 1 or b_or[2] == -1):
            self.hz_dirichlet.setConstantBC(0.)
            self.w_stress.uOfXT = None

    def setFixedNodes(self):
        """
        For moving domains: fixes nodes/boundary
        """
        self.hx_dirichlet.setConstantBC(0.)
        self.hy_dirichlet.setConstantBC(0.)
        self.hz_dirichlet.setConstantBC(0.)
        self.u_stress.uOfXT = 0.
        self.v_stress.uOfXT = 0.
        self.w_stress.uOfXT = 0.

    def setNoSlip(self):
        """
        Sets no slip conditions at the boundary
        """
        self.reset()
        self.BC_type = 'NoSlip'
        # dirichlet
        self.u_dirichlet.setConstantBC(0.)
        self.v_dirichlet.setConstantBC(0.)
        self.w_dirichlet.setConstantBC(0.)
        self.us_dirichlet.setConstantBC(0.)
        self.vs_dirichlet.setConstantBC(0.)
        self.ws_dirichlet.setConstantBC(0.)  
        self.k_dirichlet.setConstantBC(1e-20)
        self.dissipation_dirichlet.setConstantBC(1e-10) 
        # advective
        self.p_advective.setConstantBC(0.)
        self.pInit_advective.setConstantBC(0.)
        self.pInc_advective.setConstantBC(0.)  
        self.vof_advective.setConstantBC(0.)
        self.vos_advective.setConstantBC(0.)
        # diffusive
        self.pInc_diffusive.setConstantBC(0.)
        self.k_diffusive.setConstantBC(0.)
        self.dissipation_diffusive.setConstantBC(0.)  

    def setFreeSlip(self):
        """
        Sets free slip conditions at the boundary
        """
        self.reset()
        self.BC_type = 'FreeSlip'
        # dirichlet
        self.k_dirichlet.setConstantBC(1e-20)
        self.dissipation_dirichlet.setConstantBC(1e-10) 
        # advective        
        self.p_advective.setConstantBC(0.)
        self.pInit_advective.setConstantBC(0.)
        self.pInc_advective.setConstantBC(0.)
        self.u_advective.setConstantBC(0.)
        self.v_advective.setConstantBC(0.)
        self.w_advective.setConstantBC(0.)
        self.us_advective.setConstantBC(0.)
        self.vs_advective.setConstantBC(0.)
        self.ws_advective.setConstantBC(0.)
        self.vof_advective.setConstantBC(0.)
        self.vos_advective.setConstantBC(0.)
        # diffusive
        self.u_diffusive.setConstantBC(0.)
        self.v_diffusive.setConstantBC(0.)
        self.w_diffusive.setConstantBC(0.)
        self.us_diffusive.setConstantBC(0.)
        self.vs_diffusive.setConstantBC(0.)
        self.ws_diffusive.setConstantBC(0.)
        self.pInc_diffusive.setConstantBC(0.)
        self.k_diffusive.setConstantBC(0.)
        self.dissipation_diffusive.setConstantBC(0.)
        
    def setConstantInletVelocity(self, U, ramp, kk, dd , b_or):
        """
        Sets constant velocity in each inlet face with ramping up and turbulence properties
        """
        self.reset()
        self.BC_type = 'constantInletVelocity'
        Uin = np.dot(U,b_or)
        uu = U[0]
        vv = U[1]
        ww = U[2]
        
        # dirichlet
        self.u_dirichlet.setLinearRamp(ramp, uu)
        self.v_dirichlet.setLinearRamp(ramp ,vv)
        self.w_dirichlet.setLinearRamp(ramp ,ww)
        self.us_dirichlet.setConstantBC(0.)
        self.vs_dirichlet.setConstantBC(0.)
        self.ws_dirichlet.setConstantBC(0.)  
        self.k_dirichlet.setConstantBC(kk)
        self.dissipation_dirichlet.setConstantBC(dd)  
        #advective
        self.p_advective.setLinearRamp(ramp, Uin)
        self.pInit_advective.setConstantBC(0.)
        self.pInc_advective.setLinearRamp(ramp,Uin)
        # diffusive
        self.k_diffusive.setConstantBC(0.)
        self.dissipation_diffusive.setConstantBC(0.)
        self.us_diffusive.setConstantBC(0.)
        self.vs_diffusive.setConstantBC(0.)
        self.ws_diffusive.setConstantBC(0.)
    def setConstantOutletPressure(self, p, rho, g ,kk, dd, b_or):
        """
        Sets constant pressure in each outlet face for single phase flows
        """
        self.reset()
        self.BC_type = 'constantOutletPressure'
        # dirichlet
        a1 = rho*g
        self.u_dirichlet.setConstantBC(0.)
        self.v_dirichlet.setConstantBC(0.)
        self.w_dirichlet.setConstantBC(0.)
        self.us_dirichlet.setConstantBC(0.)
        self.vs_dirichlet.setConstantBC(0.)
        self.ws_dirichlet.setConstantBC(0.)

        self.p_dirichlet.setLinearBC(p,a1)
        self.pInit_dirichlet.setLinearBC(p, a1)
        self.pInc_dirichlet.setConstantBC(0.)
        self.k_dirichlet.setConstantBC(kk)
        self.dissipation_dirichlet.setConstantBC(dd)  
           
        # diffusive
        if b_or[0] == 1. or b_or[0] == -1.:
            self.u_diffusive.setConstantBC(0.)
        if b_or[1] == 1. or b_or[1] == -1.:
            self.v_diffusive.setConstantBC(0.)
        if b_or[2] == 1. or b_or[2] == -1.:
            self.w_diffusive.setConstantBC(0.)
        self.k_diffusive.setConstantBC(0.)
        self.dissipation_diffusive.setConstantBC(0.)
        self.us_diffusive.setConstantBC(0.)
        self.vs_diffusive.setConstantBC(0.)
        self.ws_diffusive.setConstantBC(0.)

    def setAtmosphere(self, orientation=None, vof_air=1.,kInflow=None,dInflow=None):
        """
        Sets atmosphere boundary conditions (water can come out)
        (!) pressure dirichlet set to 0 for this BC

        Parameters
        ----------
        orientation: Optional[array_like]
            orientation of the boundary. Optional if orientation was already
            passed when creating the BC_RANS class instance.
        vof_air: Optional[float]
            VOF value of air (default is 1.)
        """
        self.BC_type = 'Atmosphere'
        if orientation is None and self._b_or is not None:
            orientation = self._b_or
        assert orientation is not None, 'oritentation must be set for BC'
        self.reset()
        self.p_dirichlet.setConstantBC(0.)
        self.pInc_dirichlet.setConstantBC(0.)
        self.pInit_dirichlet.setConstantBC(0.)
        self.vof_dirichlet.setConstantBC(vof_air)  # air
        self.vos_dirichlet.setConstantBC(0.)
        self.u_dirichlet.setConstantBC(0.)
        self.v_dirichlet.setConstantBC(0.)
        self.w_dirichlet.setConstantBC(0.)
        self.us_dirichlet.setConstantBC(0.)
        self.vs_dirichlet.setConstantBC(0.)
        self.ws_dirichlet.setConstantBC(0.)
        self.k_dirichlet.setConstantBC(1e-20)
        self.k_diffusive.setConstantBC(0.)
        self.dissipation_dirichlet.setConstantBC(1e-10)
        self.dissipation_diffusive.setConstantBC(0.)

        if orientation[0] == 1. or orientation[0] == -1.:
            self.u_diffusive.setConstantBC(0.)
            self.us_diffusive.setConstantBC(0.)
        if orientation[1] == 1. or orientation[1] == -1.:
            self.v_diffusive.setConstantBC(0.)
            self.vs_diffusive.setConstantBC(0.)
        if orientation[2] == 1. or orientation[2] == -1.:
            self.w_diffusive.setConstantBC(0.)
            self.ws_diffusive.setConstantBC(0.)
        
        if kInflow is not None:
            self.k_dirichlet.setConstantBC(kInflow)
        else:
            logEvent("WARNING: Dirichlet condition for k in "+str(self.BC_type)+" has not been set. Ignore if RANS is not used")
        if dInflow is not None:
            self.dissipation_dirichlet.setConstantBC(dInflow)
        else:
            logEvent("WARNING: Dirichlet condition for dissipation in "+str(self.BC_type)+" has not been set. Ignore if RANS is not used")
    def setRigidBodyMoveMesh(self, body):
        """
        Sets boundary conditions for moving the mesh with a rigid body

        Parameters
        ----------
        last_pos: array_like
            last position of rigig body
        h: array_like
            displacement of the body
        rot_matrix:
            rotation matrix describing displament due to rotation between last
            position and new position (3x3 array)

        (!) if set manually, the input arrays should be updated externally
            without loosing their memory address
        """
        def get_DBC_h(i):
            def DBC_h(x, t, n=None):
                x_0 = x - body.last_position
                new_x_0 = np.dot(x_0, body.rotation_matrix)
                hx = new_x_0 - x_0 + body.h
                return hx[i]
            return DBC_h
        self.hx_dirichlet.uOfXT = get_DBC_h(0)
        self.hy_dirichlet.uOfXT = get_DBC_h(1)
        if body.nd > 2:
            self.hz_dirichlet.uOfXT = get_DBC_h(2)

    def setChMoveMesh(self, body):
        self.hx_dirichlet.uOfXT = lambda x, t, n=None: body.hx(x, t)
        self.hy_dirichlet.uOfXT = lambda x, t, n=None: body.hy(x, t)
        self.hz_dirichlet.uOfXT = lambda x, t, n=None: body.hz(x, t)

    def setTurbulentDirichlet(self, kVal, dissipationVal):
        """
        Sets only dirichlet conditions for turbulence at the boundary.
        It's a rough approximation for evalueting the near wall turbulence
        based on empirical assumptions.
        More sophisticated wall functions are recommended to be used.

        Parameters
        ----------
        kVal: float.
            constant value applied on k.
        dissipationVal: float.
            constant value applied on dissipation.
        """
        # turbulent boundary conditions
        self.k_dirichlet.setConstantBC(kVal)
        self.dissipation_dirichlet.setConstantBC(dissipationVal)
        self.k_advective.resetBC()
        self.dissipation_advective.resetBC()
        self.k_diffusive.resetBC()
        self.dissipation_diffusive.resetBC()

    def setTurbulentZeroGradient(self):
        """
        Sets only zero-gradient conditions for turbulence at the boundary.
        More sophisticated wall functions are recommended to be used.
        """
        # turbulent boundary conditions
        self.k_dirichlet.setConstantBC(0.)
        self.dissipation_dirichlet.setConstantBC(0.)
        self.k_advective.resetBC()
        self.dissipation_advective.resetBC()
        self.k_diffusive.setConstantBC(0.)
        self.dissipation_diffusive.setConstantBC(0.)

    def setWallFunction(self, wall, shearStress=False):
        """
        Sets turbulent boundaries for wall treatment.
        Calculation made on nodes outside the viscous sublayer and based
        on assumption on the velocity profile close to the wall in order to
        impose the wall shear stress.

        Parameters
        ----------
        wall: wall object.
            BoundaryConditions class to be attached for setting up
            all the turbulent parameters.
        shearStress: True/False.
            At the moment version with shearStress=False is the only one that
            returns good results.
            Keep it False at the moment!

            - When True, the wall function prescribes diffusive boundaries
              for velocity and kappa. It's like imposing the shear stress.
            - If False, the wall function prescribes dirichlet conditions.
        """
        wf = wall
        self.reset()
        self.setNoSlip()
        self.BC_type = "Wall function"
        self.dissipation_diffusive.resetBC()
        self.k_dirichlet.uOfXT = lambda x, t, n=None: wf.get_k_dirichlet(x, t, n)
        self.dissipation_dirichlet.uOfXT = lambda x, t, n=None: wf.get_dissipation_dirichlet(x, t ,n)
        self.u_dirichlet.uOfXT = lambda x, t, n=None: wf.get_u_dirichlet(x, t ,n)
        self.v_dirichlet.uOfXT = lambda x, t, n=None: wf.get_v_dirichlet(x, t ,n)
        self.w_dirichlet.uOfXT = lambda x, t, n=None: wf.get_w_dirichlet(x, t ,n)
        """
        self.dissipation_dirichlet.uOfXT = lambda x, t: wf.get_dissipation_dirichlet(x, t)
        self.vof_advective.setConstantBC(0.)
        self.p_advective.setConstantBC(0.)
        if shearStress:
            #self.k_dirichlet.uOfXT = lambda x, t: wf.get_k_dirichlet(x, t)
            self.k_advective.setConstantBC(0.)
            self.u_diffusive.uOfXT = lambda x, t: wf.get_u_diffusive(x, t)
            self.v_diffusive.uOfXT = lambda x, t: wf.get_v_diffusive(x, t)
            self.w_diffusive.uOfXT = lambda x, t: wf.get_w_diffusive(x, t)
            self.k_diffusive.setConstantBC(0.)
        else:
            self.u_dirichlet.uOfXT = lambda x, t: wf.get_u_dirichlet(x, t)
            self.v_dirichlet.uOfXT = lambda x, t: wf.get_v_dirichlet(x, t)
            self.w_dirichlet.uOfXT = lambda x, t: wf.get_w_dirichlet(x, t)
            self.k_dirichlet.uOfXT = lambda x, t: wf.get_k_dirichlet(x, t)
        """
    def setMoveMesh(self, last_pos, h=(0., 0., 0.), rot_matrix=None):
        """
        Sets boundary conditions for moving the mesh with a rigid body

        Parameters
        ----------
        last_pos: array_like
            last position of rigig body
        h: array_like
            displacement of the body
        rot_matrix:
            rotation matrix describing displament due to rotation between last
            position and new position (3x3 array)

        (!) if set manually, the input arrays should be updated externally
            without loosing their memory address
        """
        if rot_matrix is None:
            rot_matrix = np.array([[1., 0., 0.],
                                   [0., 1., 0.],
                                   [0., 0., 1.]])
        self.body_python_rot_matrix = rot_matrix
        self.body_python_last_pos = last_pos
        self.body_python_h = h
        self.hx_dirichlet.uOfXT = lambda x, t, n=None: self.__cpp_MoveMesh_hx(x, t)
        self.hy_dirichlet.uOfXT = lambda x, t, n=None: self.__cpp_MoveMesh_hy(x, t)
        self.hz_dirichlet.uOfXT = lambda x, t, n=None: self.__cpp_MoveMesh_hz(x, t)

    def __cpp_MoveMesh_h(self, x, t):
        cython.declare(x_0=cython.double[3])
        cython.declare(new_x_0=cython.double[3])
        hx = np.zeros(3)
        x_0[0] = x[0] - self.body_python_last_pos[0]
        x_0[1] = x[1] - self.body_python_last_pos[1]
        x_0[2] = x[2] - self.body_python_last_pos[2]
        new_x_0 = np.dot(x_0, self.body_python_rot_matrix)
        hx[0] = new_x_0[0] - x_0[0] + self.body_python_h[0]
        hx[1] = new_x_0[1] - x_0[1] + self.body_python_h[1]
        hx[2] = new_x_0[2] - x_0[2] + self.body_python_h[2]
        return hx

    def __cpp_MoveMesh_hx(self, x, t):
        return self.__cpp_MoveMesh_h(x, t)[0]

    def __cpp_MoveMesh_hy(self, x, t):
        return self.__cpp_MoveMesh_h(x, t)[1]

    def __cpp_MoveMesh_hz(self, x, t):
        return self.__cpp_MoveMesh_h(x, t)[2]

    def setUnsteadyTwoPhaseVelocityInlet(self, wave, smoothing, vert_axis=None,orientation=None,
                                         wind_speed=None, vof_air=1., vof_water=0.,kInflow=1e-30, dInflow = 1e-10):
        """
        Imposes a velocity profile on the fluid with input wave and wind
        conditions.

        Parameters
        ----------
        wave: proteus.WaveTools
            class describing a wave (from proteus.WaveTools)
        smoothing: float
            smoothing distance (typically 3.*he)
        vert_axis: Optional[int]
            index of vertical position vector (x:0, y:1, z:2), must always be
            aligned with gravity. If not set, will be 1 in 2D (y), 2 in 3D (z).
        wind_speed: Optional[array_like]
            speed of air phase
        vof_air: Optional[float]
            VOF value of air (default is 1.)
        vof_water: Optional[float]
            VOF value of water (default is 0.)

        Below the sea water level: fluid velocity to wave speed.
        Above the sea water level: fluid velocity set to wind speed
        (with smoothing).
        """
        self.reset()
        if vert_axis is None:
            vert_axis = self.nd - 1
        if wind_speed is None:
            wind_speed = np.zeros(3)
        if orientation is not None:
            b_or = orientation
        else:
            b_or=self._b_or
        assert b_or is not None, "ERROR: Boundary orientation for Unsteady flow inlet is not defined"
        self.waves = __cppClass_WavesCharacteristics(waves=wave, vert_axis=vert_axis, b_or=b_or,
                                                     wind_speed=wind_speed, smoothing=smoothing, vof_water=vof_water, vof_air=vof_air)

        self.u_dirichlet.uOfXT = lambda x, t, n=None: self.__cpp_UnsteadyTwoPhaseVelocityInlet_u_dirichlet(x, t)
        self.v_dirichlet.uOfXT = lambda x, t, n=None: self.__cpp_UnsteadyTwoPhaseVelocityInlet_v_dirichlet(x, t)
        self.w_dirichlet.uOfXT = lambda x, t, n=None: self.__cpp_UnsteadyTwoPhaseVelocityInlet_w_dirichlet(x, t)
        self.phi_dirichlet.uOfXT = lambda x, t, n=None: self.__cpp_UnsteadyTwoPhaseVelocityInlet_phi_dirichlet(x, t)
        self.vof_dirichlet.uOfXT = lambda x, t, n=None: self.__cpp_UnsteadyTwoPhaseVelocityInlet_vof_dirichlet(x, t)
        self.p_advective.uOfXT = lambda x, t, n=None: self.__cpp_UnsteadyTwoPhaseVelocityInlet_p_advective(x, t)
        self.pInc_advective.uOfXT = lambda x, t, n=None: self.__cpp_UnsteadyTwoPhaseVelocityInlet_p_advective(x, t)

        self.pInc_diffusive.setConstantBC(0.0)
        self.pInit_advective.uOfXT = lambda x, t, n=None: self.__cpp_UnsteadyTwoPhaseVelocityInlet_p_advective(x, t)#setConstantBC(0.0)
        self.vos_dirichlet.setConstantBC(0.0)
        self.us_dirichlet.setConstantBC(0.0)
        self.vs_dirichlet.setConstantBC(0.0)
        self.ws_dirichlet.setConstantBC(0.0)
        self.k_dirichlet.setConstantBC(kInflow)
        self.dissipation_dirichlet.setConstantBC(dInflow)
        self.dissipation_diffusive.setConstantBC(0.)
        self.k_diffusive.setConstantBC(0.0)

    def __cpp_UnsteadyTwoPhaseVelocityInlet_u_dirichlet(self, x, t):
        cython.declare(xx=cython.double[3])
        xx[0] = x[0]
        xx[1] = x[1]
        xx[2] = x[2]
        return self.waves.__cpp_calculate_velocity(xx, t)[0]

    def __cpp_UnsteadyTwoPhaseVelocityInlet_v_dirichlet(self, x, t):
        cython.declare(xx=cython.double[3])
        xx[0] = x[0]
        xx[1] = x[1]
        xx[2] = x[2]
        return self.waves.__cpp_calculate_velocity(xx, t)[1]

    def __cpp_UnsteadyTwoPhaseVelocityInlet_w_dirichlet(self, x, t):
        cython.declare(xx=cython.double[3])
        xx[0] = x[0]
        xx[1] = x[1]
        xx[2] = x[2]
        return self.waves.__cpp_calculate_velocity(xx, t)[2]

    def __cpp_UnsteadyTwoPhaseVelocityInlet_p_advective(self, x, t):
        cython.declare(xx=cython.double[3])
        xx[0] = x[0]
        xx[1] = x[1]
        xx[2] = x[2]
        return self.waves.__cpp_calculate_pressure(xx, t)

    def __cpp_UnsteadyTwoPhaseVelocityInlet_phi_dirichlet(self, x, t):
        cython.declare(xx=cython.double[3])
        xx[0] = x[0]
        xx[1] = x[1]
        xx[2] = x[2]
        return self.waves.__cpp_calculate_phi(xx, t)
    
    def __cpp_UnsteadyTwoPhaseVelocityInlet_vof_dirichlet(self, x, t):
        cython.declare(xx=cython.double[3])
        xx[0] = x[0]
        xx[1] = x[1]
        xx[2] = x[2]
        return self.waves.__cpp_calculate_vof(xx, t)

    def setTwoPhaseVelocityInlet(self, U, waterLevel, smoothing, Uwind=None,
                                 vert_axis=None, air=1., water=0.,
                                 kInflow=None, dissipationInflow=None,
                                 kInflowAir=None, dissipationInflowAir=None):
        """
        Imposes a velocity profile lower than the sea level and an open
        boundary for higher than the sealevel.

        Parameters
        ----------
        U: list.
            Velocity vector at the global system.
        Uwind: list.
            Air velocity vector at the global system.
        waterLevel: float.
            water level at global coordinate system.
        smoothing: float.
            range within smoothing function is valid.
            [3.0 times mesh element size can be a good value]
        vert_axis: optional. 
            index of vertical in position vector, must always be
            aligned with gravity, by default set to 1].
        air: optional.
            Volume fraction for air (1.0 by default).
        water: optional.
            Volume fraction for water (0.0 by default).
        kInflow: float (optional).
            K inflow value for turbulent model imposed at the boundary.
        dissipationInflow: float (optional).
            Dissipation inflow value for turbulent model imposed at the boundary.
        kInflowAir: float (optional).
            Air K inflow value for turbulent model imposed at the boundary.
        dissipationInflowAir: float (optional).
            Air dissipation inflow value for turbulent model imposed at the boundary.

        Notes
        -----
        Below the seawater level, the condition returns the _dirichlet and
        p_advective condition according to the inflow velocity.
        Above the sea water level, the condition returns the gravity as zero,
        and sets _dirichlet condition to zero, only if there is a zero inflow
        velocity component.
        (!) This condition is best used for boundaries and gravity aligned with
        one of the main axes.
        """
        self.reset()
        self.BC_type = 'TwoPhaseVelocityInlet'

        if vert_axis is None:
            vert_axis = self.nd - 1
        if Uwind is None:
            Uwind = np.zeros(3)

        U = np.array(U)
        Uwind = np.array(Uwind)

        def get_inlet_ux_dirichlet(i):
            def ux_dirichlet(x, t, n=None):
                phi = x[vert_axis] - waterLevel
                if phi <= 0.:
                    H = 0.0
                elif 0 < phi <= smoothing:
                    H = smoothedHeaviside(smoothing, phi-smoothing/2.)#smoothedHeaviside(smoothing/2., phi - smoothing/2.)
                else:
                    H = 1.0
                u = H * Uwind[i] + (1 - H) * U[i]
                return u
            return ux_dirichlet

        def inlet_phi_dirichlet(x, t, n=None):
            phi = x[vert_axis] - waterLevel
            return phi

        def inlet_vof_dirichlet(x, t, n=None):
            phi = x[vert_axis] - waterLevel
            if phi >= smoothing:
                H = 1.
            elif smoothing > 0 and -smoothing < phi < smoothing:
                H = smoothedHeaviside(smoothing, phi)
            elif phi <= -smoothing:
                H = 0.
            vof = H * air + (1 - H) * water
            return vof

        def inlet_p_advective(x, t, n=None):
            b_or = self._b_or
            phi = x[vert_axis] - waterLevel
            if phi <= 0.:
                H = 0.0
            elif 0 < phi <= smoothing:
                H = smoothedHeaviside(smoothing, phi-smoothing/2.)#smoothedHeaviside(smoothing/2., phi - smoothing/2.)
            else:
                H = 1.0
            u = H * Uwind + (1 - H) * U
            # This is the normal velocity, based on the inwards boundary
            # orientation -b_or
            u_p = np.sum(u * np.abs(b_or))
            return -u_p

        def inlet_k_dirichlet(x, t, n=None):
            phi = x[vert_axis] - waterLevel
            if phi <= 0.:
                H = 0.0
            elif 0 < phi <= smoothing:
                H = smoothedHeaviside(smoothing, phi-smoothing/2.)#smoothedHeaviside(smoothing/2., phi - smoothing/2.)
            else:
                H = 1.0
            return H * kInflowAir + (1 - H) * kInflow

        def inlet_dissipation_dirichlet(x, t, n=None):
            phi = x[vert_axis] - waterLevel
            if phi <= 0.:
                H = 0.0
            elif 0 < phi <= smoothing:
                H = smoothedHeaviside(smoothing, phi-smoothing/2.)#smoothedHeaviside(smoothing/2., phi - smoothing/2.)
            else:
                H = 1.0
            return H * dissipationInflowAir + (1 - H) * dissipationInflow

        self.u_dirichlet.uOfXT = get_inlet_ux_dirichlet(0)
        self.v_dirichlet.uOfXT = get_inlet_ux_dirichlet(1)
        self.w_dirichlet.uOfXT = get_inlet_ux_dirichlet(2)
        self.phi_dirichlet.uOfXT = inlet_phi_dirichlet
        self.vof_dirichlet.uOfXT = inlet_vof_dirichlet
        self.p_advective.uOfXT = inlet_p_advective
        if kInflow is not None:
            self.k_dirichlet.uOfXT = inlet_k_dirichlet
            self.k_advective.resetBC()
            self.k_diffusive.resetBC()
        if dissipationInflow is not None:
            self.dissipation_dirichlet.uOfXT = inlet_dissipation_dirichlet
            self.dissipation_advective.resetBC()
            self.dissipation_diffusive.resetBC()

    def setHydrostaticPressureOutletWithDepth(self, seaLevel, rhoUp, rhoDown, g,
                                              refLevel, smoothing, orientation=None,U=None, Uwind=None,
                                              pRef=0.0, vert_axis=None,
                                              air=1.0, water=0.0,
                                              kInflow=None, dissipationInflow=None,
                                              kInflowAir=None, dissipationInflowAir=None):
        """
        Returns the pressure and vof profile based on the known depth.
        If the boundary is aligned with one of the main axes, sets the tangential
        velocity components to zero as well.
        (!) This condition is best used for boundaries and gravity aligned with
        one of the main axes.

        Parameters
        ----------
        rhoUp: Phase density of the upper part.
        rhoDown: Phase density of the lower part.
        g: Gravitational acceleration vector.
        refLevel: Level at which pressure = pRef.
        pRef: Reference value for the pressure at x[vert_axis]=refLevel, by default set to 0.
        vert_axis: index of vertical in position vector, must always be aligned with gravity, by default set to 1.
        """
        self.reset()

        if vert_axis is None:
            vert_axis = self.nd - 1

        def hydrostaticPressureOutletWithDepth_p_dirichlet(x, t , n=None):
            p_top = pRef
            phi_top = refLevel - seaLevel
            phi = x[vert_axis] - seaLevel
            return p_top - g[vert_axis] * (rhoDown * (phi_top - phi) +
                                           (rhoUp - rhoDown) *
                                           (smoothedHeaviside_integral(smoothing, phi_top)
                                            -
                                            smoothedHeaviside_integral(smoothing, phi)))

        def hydrostaticPressureOutletWithDepth_phi_dirichlet(x, t, n=None):
            phi = x[vert_axis] - seaLevel
            return phi

        def hydrostaticPressureOutletWithDepth_vof_dirichlet(x, t, n=None):
            phi = x[vert_axis] - seaLevel
            if phi >= smoothing:
                H = 1.
            elif smoothing > 0 and -smoothing < phi < smoothing:
                H = smoothedHeaviside(smoothing, phi)
            elif phi <= -smoothing:
                H = 0.
            return H * air + (1 - H) * water

        def inlet_k_dirichlet(x, t, n=None):
            phi = x[vert_axis] - seaLevel
            if phi <= 0.:
                H = 0.0
            elif 0 < phi <= smoothing:
                H = smoothedHeaviside(smoothing/2., phi - smoothing/2.)
            else:
                H = 1.0
            return H * kInflowAir + (1 - H) * kInflow

        def inlet_dissipation_dirichlet(x, t, n=None):
            phi = x[vert_axis] - seaLevel
            if phi <= 0.:
                H = 0.0
            elif 0 < phi <= smoothing:
                H = smoothedHeaviside(smoothing/2., phi - smoothing/2.)
            else:
                H = 1.0
            return H * dissipationInflowAir + (1 - H) * dissipationInflow

        if orientation is not None:
            b_or = orientation
        else:
            b_or=self._b_or
        assert b_or is not None, "ERROR: Boundary orientation for Hydrostatic flow outlet is not defined"
        if b_or[0] == 1. or b_or[0] == -1.:
            self.v_dirichlet.setConstantBC(0.)
            self.w_dirichlet.setConstantBC(0.)
            self.u_diffusive.setConstantBC(0.)
        if b_or[1] == 1. or b_or[1] == -1.:
            self.u_dirichlet.setConstantBC(0.)
            self.w_dirichlet.setConstantBC(0.)
            self.v_diffusive.setConstantBC(0.)
        if b_or[2] == 1. or b_or[2] == -1.:
            self.u_dirichlet.setConstantBC(0.)
            self.v_dirichlet.setConstantBC(0.)
            self.w_diffusive.setConstantBC(0.)
#sediment
        self.us_advective.setConstantBC(0.)
        self.vs_advective.setConstantBC(0.)
        self.ws_advective.setConstantBC(0.)
        self.vos_advective.setConstantBC(0.)
        self.us_diffusive.setConstantBC(0.)
        self.vs_diffusive.setConstantBC(0.)
#end sediment
        self.p_dirichlet.uOfXT = hydrostaticPressureOutletWithDepth_p_dirichlet
        self.pInit_dirichlet.uOfXT = hydrostaticPressureOutletWithDepth_p_dirichlet
        self.pInc_dirichlet.setConstantBC(0.)
        self.phi_dirichlet.uOfXT = hydrostaticPressureOutletWithDepth_phi_dirichlet
        self.vof_dirichlet.uOfXT = hydrostaticPressureOutletWithDepth_vof_dirichlet
        self.k_diffusive.setConstantBC(0.)
        self.dissipation_diffusive.setConstantBC(0.)
        if U is not None:
            def get_inlet_ux_dirichlet(i):
                def ux_dirichlet(x, t, n=None):
                    phi = x[vert_axis] - seaLevel
                    if phi <= 0.:
                        H = 0.0
                    elif 0 < phi <= smoothing:
                        H = smoothedHeaviside(smoothing/2., phi - smoothing/2.)
                    else:
                        H = 1.0
                    return H * Uwind[i] + (1 - H) * U[i]
                return ux_dirichlet

            if Uwind is None:
                Uwind = np.zeros(3)
            U = np.array(U)
            Uwind = np.array(Uwind)
            self.u_dirichlet.uOfXT = get_inlet_ux_dirichlet(0)
            self.v_dirichlet.uOfXT = get_inlet_ux_dirichlet(1)
            self.w_dirichlet.uOfXT = get_inlet_ux_dirichlet(2)
            self.u_diffusive.resetBC()

        if kInflow is not None:
            self.k_dirichlet.uOfXT = inlet_k_dirichlet
            self.k_advective.resetBC()
            self.k_diffusive.resetBC()
        if dissipationInflow is not None:
            self.dissipation_dirichlet.uOfXT = inlet_dissipation_dirichlet
            self.dissipation_advective.resetBC()
            self.dissipation_diffusive.resetBC()

# for regions

class RelaxationZone:
    """
    Holds information about a relaxation zone (wave generation/absorption
    or porous zone)

    Parameters
    ----------
    zone_type: string
        type of zone, can be set to 'absorption', 'generation', or 'porous'
    center: array_like
        coordinates of center of the zone
    orientation: array_like
        orientation for absorption/generation zones: from boundary to tank
    epsFact_porous: float
        half the zone length
    waves: Optional[proteus.WaveTools]
        class instance of a wave from proteus.WaveTools (must be set for
        generation zone)
    shape: Optional[proteus.SpatialTools.Shape]
        shape class instance containing region
    dragAlpha: Optional[float]
        parameter for porous zones (default: 0.5/1.005e-6)
    dragBeta: Optional[float]
        parameter for porous zones (default: 0.)
    porosity: Optional[float]
        parameter for porous zone (default: 1.)
    vert_axis: Optional[int]
        index of vertical position vector (x:0, y:1, z:2), must always be
        aligned with gravity. If not set, will be 1 in 2D (y), 2 in 3D (z).
    smoothing: Optional[float]
        smoothing distance from the free surface (usually 3*he)
    vof_water: Optional[int]
        VOF value of water (default: 0)
    vof_air: Optional[int]
        VOF value of air (default: 1)
    """

    def __cinit__(self, zone_type, center, orientation, epsFact_porous,
                  waves=None, shape=None, wind_speed=np.array([0., 0., 0.]),
                  dragAlpha=0.5/1.005e-6, dragBeta=0., porosity=1., vert_axis=None, smoothing=0.,
                  vof_water=0., vof_air=1.):
        self.Shape = shape
        self.nd = self.Shape.Domain.nd
        self.zone_type = zone_type
        self.center = center
        self.orientation = orientation
        if vert_axis is None:
            vert_axis = self.Shape.Domain.nd - 1
        self.vert_axis = vert_axis
        if waves is not None:
            self.waves = __cppClass_WavesCharacteristics(waves=waves, wind_speed=wind_speed, vert_axis=vert_axis,
                                                         smoothing=smoothing, vof_water=vof_water, vof_air=vof_air)
        self.epsFact_porous = epsFact_porous
        self.dragAlpha = dragAlpha
        self.dragBeta = dragBeta
        self.porosity = porosity
        self.zero_vel = np.zeros(3)

    def calculate_init(self):
        if self.zone_type == 'generation':
            # self.u = &self.waves.u
            # self.eta = &self.waves.eta
            self.uu = self.__cpp_calculate_vel_wave
            self.phi = self.__cpp_calculate_phi_porous_sponge
        elif self.zone_type == 'absorption':
            self.uu = self.__cpp_calculate_vel_zero
            self.phi = self.__cpp_calculate_phi_porous_sponge
        elif self.zone_type == 'porous':
            self.uu = self.__cpp_calculate_vel_zero
            self.phi = self.__cpp_calculate_phi_porous

    def calculate_phi(self, x):
        return self.phi(self, x)

    def __cpp_calculate_phi_porous_sponge(self, x):
        """
        Used for RelaxationZone only
        """
        cython.declare(d=cython.double[3], o=cython.double[3])
        d[0] = self.center[0] - x[0]
        d[1] = self.center[1] - x[1]
        o[0] = self.orientation[0]
        o[1] = self.orientation[1]
        if self.nd > 2:
            d[2] = self.center[2] - x[2]
            o[2] = self.orientation[2]
        else:
            d[2] = 0
            o[2] = 0
        phi = o[0] * d[0] + o[1] * d[1] + o[2] * d[2]
        return phi

    def __cpp_calculate_phi_porous(self, x):
        IN_POROUS_ZONE=-10000.0
        return IN_POROUS_ZONE

    def calculate_vel(self, x, t):
        cython.declare(d=cython.double[3], o=cython.double[3])
        d[0] = x[0]
        d[1] = x[1]
        d[2] = x[2]
        return self.uu(self, d, t)

    def calculate_phi_python(self, x):
        cython.declare(xx=cython.double[3], tt=cython.double)
        xx[0] = x[0]
        xx[1] = x[1]
        xx[2] = x[2]
        ph = self.phi(self, xx)
        return ph

    def calculate_vel_python(self, x, t, n=None):
        cython.declare(xx=cython.double[3], tt=cython.double)
        xx[0] = x[0]
        xx[1] = x[1]
        xx[2] = x[2]
        tt = t
        return self.uu(self, xx, tt)

    def __cpp_calculate_vel_zero(self, x, t):
        return self.zero_vel

    def __cpp_calculate_vel_wave(self, x, t):
        return self.waves.__cpp_calculate_velocity(x, t)


class RelaxationZoneWaveGenerator:
    """
    Prescribe a velocity penalty scaling in a material zone via a
    Darcy-Forchheimer penalty

    Parameters
    ----------
    zones: dict
        dictionary with key as the region flag and values as a RelaxationZone
        class
    nd: int
        number of dimensions of domain
    """

    def __init__(self, zones, nd):
        self.zones = zones
        self.nd = nd

    def attachModel(self, model, ar):
        self.model = model
        self.ar = ar
        return self

    def attachAuxiliaryVariables(self, avDict):
        pass

    def calculate_init(self):
        max_key = 0
        for key, zone in list(self.zones.items()):
            zone.calculate_init()
            if key > max_key:
                max_key = key
        self.max_flag = max_key
        self.zones_array = np.empty(self.max_flag + 1, dtype=RelaxationZone)
        for key, zone in list(self.zones.items()):
            self.zones_array[key] = zone

    def calculate(self):
        self.__cpp_iterate()

    def __cpp_iterate(self):
        nl = len(self.model.levelModelList)
        for l in range(nl):  # usually only 1
            # initialisation of variables before costly loop
            m = self.model.levelModelList[l]
            nE = m.coefficients.q_phi.shape[0]
            nk = m.coefficients.q_phi.shape[1]
            t = m.timeIntegration.t
            qx = m.q['x']
            q_phi_porous = m.coefficients.q_phi_porous
            q_velocity_porous = m.coefficients.q_velocity_porous
            mTypes = m.mesh.elementMaterialTypes
            # costly loop
            for eN in range(nE):
                mType = mTypes[eN]
                if mType <= self.max_flag:
                    zone = self.zones_array[mType]
                    if zone is not None:
                        for k in range(nk):
                            x[0] = qx[eN, k, 0]
                            x[1] = qx[eN, k, 1]
                            x[2] = qx[eN, k, 2]
                            phi = zone.calculate_phi(x)
                            q_phi_porous[eN, k] = phi
                            u = zone.calculate_vel(x, t)
                            q_velocity_porous[eN, k, 0] = u[0]
                            q_velocity_porous[eN, k, 1] = u[1]
                            if self.nd > 2:
                                q_velocity_porous[eN, k, 2] = u[2]
            m.q['phi_porous'] = q_phi_porous
            m.q['velocity_porous'] = q_velocity_porous


class __cppClass_WavesCharacteristics:
    """
    Class holding information from WaveTools waves and cnvering it to
    boundary conditions to use for relaxation zones and wave inlet.
    This class is created automatically when passing WaveTools class
    instances to a relaxation zone or wave inlet BC.

    Parameters
    ----------
    wave: proteus.WaveTools
        class describing a wave (from proteus.WaveTools)
    vert_axis: int
        index of vertical position vector (x:0, y:1, z:2), must always be
        aligned with gravity. If not set, will be 1 in 2D (y), 2 in 3D (z).
    wind_speed: Optional[array_like]
        speed of air phase
    b_or: Optional[array_like]
        boundary orientation. Necessary for pressure calculations. Used
        for boundary conditions but not in relaxation zones.
    smoothing: Optional[float]
        smoothing distance from the free surface (usually 3*he)
    vof_water: Optional[float]
        VOF value of water (default: 0)
    vof_air: Optional[float]
        VOF value of air (default: 1)
    """

    def __init__(self, waves, vert_axis, wind_speed=None, b_or=None,
                 smoothing=0., vof_water=0., vof_air=1.):
        self.WT = waves  # wavetools wave
        self.vert_axis = vert_axis
        self.zero_vel = np.zeros(3)
        self._b_or = b_or
        self.smoothing = smoothing
        self.vof_air = vof_air
        self.vof_water = vof_water
        if wind_speed is None:
            self.wind_speed = self.zero_vel
        else:
            self.wind_speed = wind_speed

    def __cpp_calculate_velocity(self, x, t):
        # u is returned as a memoryview (pxd: cdef double[:]); a C array declared here lives on this
        # function's stack and the view dangles once it returns. Unnoticed until the module was built
        # with line tracing, which reuses that stack: the caller then read 0.0.
        u = np.empty(3)
        cython.declare(xx=cython.double[3])
        cython.declare(x_max=cython.double[3])
        xx[0] = x[0]
        xx[1] = x[1]
        xx[2] = x[2]
        phi = self.__cpp_calculate_phi(x, t)
        if phi <= 0.:
            # no smoothing below mwl, or wave peak could get chopped off
            H = 0
            waterSpeed = self.WT.u(xx, t)
        elif 0 < phi <= self.smoothing:
            # smoothing on half the range of VOF (above wave crest)
            H = smoothedHeaviside(self.smoothing/2., phi - self.smoothing/2.)
            # use max velocity of wave for water
            x_max[0] = x[0]
            x_max[1] = x[1]
            x_max[2] = x[2]
            x_max[self.vert_axis] = x[self.vert_axis] - phi
            waterSpeed = self.WT.u(x_max, t)
        else:
            H = 1.
            waterSpeed = self.zero_vel
        u[0] = H * self.wind_speed[0] + (1 - H) * waterSpeed[0]
        u[1] = H * self.wind_speed[1] + (1 - H) * waterSpeed[1]
        u[2] = H * self.wind_speed[2] + (1 - H) * waterSpeed[2]
        return u

    def __cpp_calculate_pressure(self, x, t):
        # This is the normal velocity, based on the outwards boundary
        # orientation b_or
        # needs to be equal to -ux_dirichlet
        ux = self.__cpp_calculate_velocity(x, t)
        b0, b1, b2 = self._b_or[0], self._b_or[1], self._b_or[2]
        u0, u1, u2 = ux[0], ux[1], ux[2]
        return b0 * u0 + b1 * u1 + b2 * u2

    def __cpp_calculate_phi(self, x, t):
        cython.declare(xx=cython.double[3])
        xx[0] = x[0]
        xx[1] = x[1]
        xx[2] = x[2]
        level = self.WT.mwl + self.WT.eta(xx, t)
        return x[self.vert_axis] - level

    def __cpp_calculate_vof(self, x, t):
        phi = self.__cpp_calculate_phi(x, t)
        H = self.__cpp_calculate_smoothing_H(phi)
        return H

    def __cpp_calculate_smoothing_H(self, phi):
        if phi >= self.smoothing:
            H = 1.
        elif self.smoothing > 0 and -self.smoothing < phi < self.smoothing:
            H = smoothedHeaviside(self.smoothing, phi)
        elif phi <= -self.smoothing:
            H = 0.
        return H


def __x_to_cpp(x):
    cython.declare(xx=double[3])
    xx[0] = x[0]
    xx[1] = x[1]
    xx[2] = x[2]
    return xx


class WallFunctions(AuxiliaryVariables.AV_base):
    """
    Auxiliary variable used to calculate attributes of an associated shape
    class instance acting as a wall.
    """

    def __init__(self, turbModel, kWall, Y, Yplus, U0, nu=1.004e-6, Cmu=0.09, K=0.41, B=5.57):
        """
        Sets turbulent boundaries for wall treatment.
        Calculation made on nodes outside the viscous sublayer and based
        on assumption on the velocity profile close to the wall in order to
        impose the wall shear stress.

        - k is assumed to be constant in the fully turbulent region close to the wall,
          in this way kv = kp.
        - dissipation is calculated.

        Parameters
        ----------
        turbModel: string.
            'ke' or 'kw', for switching between k-epsilon or k-omega models.
        kWall: object.
            Class kWall object for extracting kappa from the model.
        Y: float.
            size of the nearest element to the boundary.
        Yplus: float.
            size of the nearest element to the boundary in terms of wall unit.
        U0: array_like.
            stream velocity.
        nu: float.
            fluid viscosity.
        Cmu: float.
            turbulent viscosity constant.
        K: float.
            von Karman coefficient.
        B: float.
            roughness coefficient for walls.
        """
        self.turbModel = turbModel
        self.Y = Y
        self.Yplus = Yplus
        self.U0 = U0
        self.nu = nu
        self.Cmu = Cmu
        self.K = K
        self.B = B
        # initialise variables
        self.Ubound = np.zeros(3)
        self.kappa = 1e-10
        self.tau_rho = 0.
        self.utAbs = 1e-10
        self.ut = np.zeros(3)
        self.x = np.zeros(3)
        self.t = 0.
        self.model = None
        self.xi, self.element, self.rank = None, None, None
        self.kWall = kWall

    def attachModel(self, model, ar):
        """
        Attaches model to auxiliary variable
        """
        self.model = model
        self.ar = ar
        self.nd = model.levelModelList[0].nSpace_global
        self.Closure_0_model = model.levelModelList[0].coefficients.Closure_0_model
        self.Closure_1_model = model.levelModelList[0].coefficients.Closure_1_model
        return self

    def attachAuxiliaryVariables(self, avDict):
        pass

    def calculate_init(self):
        pass

    def calculate(self):
        pass

    def getLocalNearestNode(self, coords, kdtree):
        """
        Finds nearest node to coordinates (local)
        Parameters
        ----------
        coords: array_like
            coordinates from which to find nearest node
        kdtree: scipy.spatial.cKDTree
            instance of scipy kdtree
        Returns
        -------
        node: int
            nearest node index
        distance: float
            distance to nearest node
        """
        # determine local nearest node distance
        distance, node = kdtree.query(coords)
        return node, distance

    def getLocalElement(self, femSpace, coords, node):
        """
        Given coordinates and its nearest node, determine if it is on a
        local element.
        Parameters
        ----------
        femSpace: object
            finite element space
        coords: array_like
            coordinates from which to element
        node: int
            nearest node index
        Returns
        -------
        eN: int or None
            local index of element (None if not found)
        """
        patchBoundaryNodes = set()
        checkedElements = []
        # nodeElementOffsets give the indices to get the elements sharing the node
        statem1 = node + 1 < len(femSpace.mesh.nodeElementOffsets)
        for eOffset in range(femSpace.mesh.nodeElementOffsets[node], femSpace.mesh.nodeElementOffsets[node + 1]):
            eN = femSpace.mesh.nodeElementsArray[eOffset]
            checkedElements.append(eN)
            # union of set
            patchBoundaryNodes |= set(femSpace.mesh.elementNodesArray[eN])
            # evaluate the inverse map for element eN (global to local)
            xi = femSpace.elementMaps.getInverseValue(eN, coords)
            #J = femSpace.elementMaps.getJacobianValues(eN, )
            # query whether xi lies within the reference element
            if femSpace.elementMaps.referenceElement.onElement(xi):
                return eN
        # extra loop if case coords is in neighbour element
        for node in patchBoundaryNodes:
            for eOffset in range(femSpace.mesh.nodeElementOffsets[node], femSpace.mesh.nodeElementOffsets[node + 1]):
                eN = femSpace.mesh.nodeElementsArray[eOffset]
                if eN not in checkedElements:
                    checkedElements.append(eN)
                    # evaluate the inverse map for element eN
                    xi = femSpace.elementMaps.getInverseValue(eN, coords)
                    # query whether xi lies within the reference element
                    if femSpace.elementMaps.referenceElement.onElement(xi):
                        return eN
        # no elements found
        return None

    def findElementContainingCoords(self, coords):
        """
        Given global coordinates of a point, returns
        local coordinates and the owner of the point.

        Parameters
        ----------
        coords: array_like
            global coordinates to look for
        Returns
        -------
        xi:
            local coordinates
        eN: int
            (local) element number
        rank: int
            processor rank containing element
        """
        comm = Comm.get().comm.tompi4py()
        xi = owning_proc = element = rank = None  # initialised as None
        self.xi, self.element, self.rank = xi, element, rank
        # get nearest node on each processor
        # comm.barrier()
        self.u = self.model.levelModelList[0].u
        self.femSpace_velocity = self.u[1].femSpace
        nodes_kdtree = spatial.cKDTree(self.model.levelModelList[0].mesh.nodeArray)
        nearest_node, nearest_node_distance = self.getLocalNearestNode(coords, nodes_kdtree)
        # look for element containing coords on each processor (if it exists)
        local_element = self.getLocalElement(self.femSpace_velocity, coords, nearest_node)
        if local_element:
            xi = self.femSpace_velocity.elementMaps.getInverseValue(local_element, coords)
            rank = comm.rank
        else:
            xi = None
            rank = None
        #rank = comm.allreduce(rank, op=MPI.MAX)
        return xi, local_element, rank

    def getFluidVelocityLocalCoords(self, xi, element, rank):
        """
        Given local details, returns velocity field at those coordinates.

        Parameters
        ----------
        xi:
            local coords in element
        element: int
            element number (local to processor 'rank')
        rank: int
            rank of processor owning the element
        """
        comm = Comm.get().comm.tompi4py()
        if comm.rank == rank:
            u = self.u[1].getValue(element, xi)
            v = self.u[2].getValue(element, xi)
            if self.nd > 2:
                w = self.u[3].getValue(element, xi)
            if self.nd <= 2:
                w = 0.
        else:
            u = v = w = None
        return u, v, w

    def setYplusNormalDirection(self, x, t,n, relax=1.0):
        """
        Return the point at y+ distance in normal
        direction from the boundary.
        """
        # near wall point
        if len(n) < 3:
            nP = (relax * self.Y * (-n[:2])) + x[:2]
        else:
            nP = (relax * self.Y * (-n)) + x
        return nP

    def extractVelocity(self, x, t, n):
        """
        Extraction of the velocity at y+ distance from the boundary.
        """
        coords = self.setYplusNormalDirection(x, t, n)
        xi, element, rank = self.findElementContainingCoords(coords)
        if rank is not None:
            u, v, w = self.getFluidVelocityLocalCoords(xi, element, rank)
        else:
            relax = 0.5
            while rank is None:
                coords_relax = self.setYplusNormalDirection(x, t, n ,relax)
                xi, element, rank = self.findElementContainingCoords(coords_relax)
                relax *= 0.5
            # just use the element containing the boundary quadrature point to interpolate to the y+ point
            u, v, w = self.getFluidVelocityLocalCoords(self.femSpace_velocity.elementMaps.getInverseValue(element, coords),
                                                       element,
                                                       rank)
        self.xi, self.element, self.rank = xi, element, rank
        return u, v, w

    def tangentialVelocity(self, x, t, n, uInit=None):
        """
        Given the velocity, calculates its
        tangential component to the wall.

        Parameters
        ----------
        uInit: True/False.
            Switch for initializing the module.
            True only during the first time step.
        """
        n=np.pad(n,(0,3-n.shape[0]),mode='constant',constant_values=0.0)
        if uInit is True or self.model is None:
            u0, u1, u2 = self.U0
        else:
            u0, u1, u2 = self.extractVelocity(x, t, n)
        self.meanV = np.array([u0, u1, u2])
        # projection of u vector over an ortoganal plane to n
        self.tanU = self.meanV - self.meanV * (n**2)
        # tangential unit vector
        self.tV = self.tanU/np.sqrt(np.sum(self.tanU**2))

    def getVariables(self, x, t):
        """
        Calculates velocity, gradient of the velocity and
        kappa according with wall functions theory (see
        S. B. Pope pg 442-443).
        """
        comm = Comm.get().comm.tompi4py()
        # Extraction of kappa
        self.kappa = self.kWall.getKappa(x, t, self.xi, self.element, self.rank)
        # Calculation of nominal friction velocity based on kappa
        self.utStar = (self.kappa**0.5) * (self.Cmu**0.25)
        self.Ystar = self.Y * self.utStar / self.nu
        # Absolute value of the extracted velocity at y+ location.
        Up = np.sqrt(np.sum(self.tanU**2))
        # viscous layer
        if self.Ystar < 11.225:
            self.Ustar = self.Ystar
            self.uDir = (self.utStar*self.Ystar) * self.tV
            self.gradU = ( (self.utStar**2)/self.nu ) * self.tV
        # log-law layer
        else:
            # Wall function theory from S.B. Pope, page 442-443
            E = np.exp(self.B * self.K)
            self.Ustar = self.utStar * np.log(E * self.Ystar) / self.K
            self.utAbs = self.utStar * np.sqrt(Up/self.Ustar)
            # Velocity vector and velocity gradient multiplied by the tangential vector unit
            self.gradU = (self.utAbs/(self.K*self.Y)) * self.tV
            # Linear approximation for velocity at the wall (using the gradU of the logLaw)
            self.uDir = self.tanU - (self.gradU * self.Y)

    def get_u_dirichlet(self, x, t, n):
        if t > 0.:
            uInit = False
        else:
            uInit = True
        self.tangentialVelocity(x, t, n, uInit)
        self.getVariables(x, t)
        return self.uDir[0]

    def get_v_dirichlet(self, x, t, n):
        if t > 0.:
            uInit = False
        else:
            uInit = True
        self.tangentialVelocity(x, t, n, uInit)
        self.getVariables(x, t)
        return self.uDir[1]

    def get_w_dirichlet(self, x, t, n):
        if t > 0.:
            uInit = False
        else:
            uInit = True
        self.tangentialVelocity(x, t, n, uInit)
        self.getVariables(x, t)
        return self.uDir[2]

    def get_k_dirichlet(self, x, t, n):
        if t > 0.:
            uInit = False
        else:
            uInit = True
        self.tangentialVelocity(x, t,n, uInit)
        self.getVariables(x, t)
        return self.kappa

    def get_dissipation_dirichlet(self, x, t, n):
        if t > 0.:
            uInit = False
        else:
            uInit = True
        self.tangentialVelocity(x, t, n, uInit)
        self.getVariables(x, t)
        d = 0.
        if self.turbModel == 'ke':
            d = (self.utStar**3)/(self.K*self.Y)
        elif self.turbModel == 'kw' and self.kappa > 0.:
            d = np.sqrt(self.kappa)/(self.K*self.Y*(self.Cmu**0.25))
        return d

    def get_u_diffusive(self, x, t, n):
        if t > 0.:
            uInit = False
        else:
            uInit = True
        self.tangentialVelocity(x, t, n, uInit)
        self.getVariables(x, t)
        gradU = self.gradU[0]
        return gradU

    def get_v_diffusive(self, x, t, n ):
        if t > 0.:
            uInit = False
        else:
            uInit = True
        self.tangentialVelocity(x, t, n, uInit)
        self.getVariables(x, t)
        gradU = self.gradU[1]
        return gradU

    def get_w_diffusive(self, x, t, n):
        if t > 0.:
            uInit = False
        else:
            uInit = True
        self.tangentialVelocity(x, t, n, uInit)
        self.getVariables(x, t)
        gradU = self.gradU[2]
        return gradU


class kWall(AuxiliaryVariables.AV_base):
    """
    Auxiliary variable used to calculate attributes of an associated shape
    class instance acting as a wall for the k variable.
    """

    def __init__(self, Y, Yplus, nu=1.004e-6, Cmu=0.09):
        """
        Sets turbulent boundaries for wall treatment.
        """
        self.kappa = 1e-10
        self.Y = Y
        self.Yplus = Yplus
        self.nu = nu
        self.model = None
        self.Cmu = Cmu

    def attachModel(self, model, ar):
        """
        Attaches model to auxiliary variable
        """
        self.model = model
        self.ar = ar
        self.nd = model.levelModelList[0].nSpace_global
        return self

    def attachAuxiliaryVariables(self, avDict):
        pass

    def calculate_init(self):
        pass

    def calculate(self):
        pass

    def getFluidKappaLocalCoords(self, xi, element, rank):
        """

        Parameters
        ----------
        xi:
            local coords in element
        element: int
            element number (local to processor 'rank')
        rank: int
            rank of processor owning the element
        """
        comm = Comm.get().comm.tompi4py()
        # solution of the selected model
        self.u = self.model.levelModelList[0].u
        #self.femSpace_kappa = self.u[0].femSpace
        if comm.rank == rank:
            kappa = self.u[0].getValue(element, xi)
        else:
            kappa = None
        return kappa

    def kappaNearWall(self, xi, element, rank, kInit=None):
        if kInit is True or self.model is None:
            self.ut = self.Yplus * self.nu / self.Y
            self.kappa = (self.ut**2)/np.sqrt(self.Cmu)
        else:
            self.kappa = self.getFluidKappaLocalCoords(xi, element, rank)

    def getKappa(self, x, t, xi, element, rank):
        if t > 0.:
            kInit = False
        else:
            kInit = True
        self.kappaNearWall(xi, element, rank, kInit)
        return abs(self.kappa)