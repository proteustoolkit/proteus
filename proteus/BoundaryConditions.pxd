# cython: language_level=3
cimport cython
cimport numpy as np
import cython

cdef class BC_Base:
    cdef double[:] _b_or
    cdef public:
        cdef int nd
        object Shape
        str name
        str BC_type
        object ct
    # cpdef void newGlobalBC(BC_Base cls, newstr name, object default_value)
    # cpdef void getContext(BC_Base cls, object context=*)

ctypedef double (*cpp_uOfXT) (BoundaryCondition, double[:], double)
cdef class BoundaryCondition:
    cdef cpp_uOfXT uuOfXT
    cdef public:
        object uOfXT
    cpdef void resetBC(self)
    # cpdef void setConstantBC(self, double value)
    # cpdef void setLinearBC(self, double a0, double a1, int i)

from libcpp cimport bool

ctypedef double[:,:,:] double_memview3
ctypedef double[:,:] double_memview2
ctypedef double[:] double_memview1
ctypedef int[:] int_memview1

# ctypedef double (*cpp_uOfXT) (BC_RANS, double[:], double)

cdef class BC_RANS(BC_Base):
    cdef double[:] zero_array
    cdef double f
    cdef __cppClass_WavesCharacteristics waves
    cdef object body
    cdef double[:] wind_speed
    cdef double[:, :] body_python_rot_matrix
    cdef double[:] body_python_last_pos
    cdef double[:] body_python_h
    cdef public:
        # dirichlet
        cdef BoundaryCondition p_dirichlet
        cdef BoundaryCondition u_dirichlet
        cdef BoundaryCondition v_dirichlet
        cdef BoundaryCondition w_dirichlet
        cdef BoundaryCondition phi_dirichlet
        cdef BoundaryCondition vof_dirichlet
        cdef BoundaryCondition k_dirichlet
        cdef BoundaryCondition dissipation_dirichlet
        cdef BoundaryCondition pAddedMass_dirichlet
        # advective
        cdef BoundaryCondition p_advective
        cdef BoundaryCondition u_advective
        cdef BoundaryCondition v_advective
        cdef BoundaryCondition w_advective
        cdef BoundaryCondition vof_advective
        cdef BoundaryCondition k_advective
        cdef BoundaryCondition dissipation_advective
        #diffusive
        cdef BoundaryCondition u_diffusive
        cdef BoundaryCondition v_diffusive
        cdef BoundaryCondition w_diffusive
        cdef BoundaryCondition k_diffusive
        cdef BoundaryCondition dissipation_diffusive
        # mesh
        cdef BoundaryCondition hx_dirichlet
        cdef BoundaryCondition hy_dirichlet
        cdef BoundaryCondition hz_dirichlet
        cdef BoundaryCondition u_stress
        cdef BoundaryCondition v_stress
        cdef BoundaryCondition w_stress
        # sediment solver
        cdef BoundaryCondition us_dirichlet
        cdef BoundaryCondition vs_dirichlet
        cdef BoundaryCondition ws_dirichlet
        cdef BoundaryCondition vos_dirichlet
        cdef BoundaryCondition us_advective
        cdef BoundaryCondition vs_advective
        cdef BoundaryCondition ws_advective
        cdef BoundaryCondition vos_advective
        cdef BoundaryCondition us_diffusive
        cdef BoundaryCondition vs_diffusive
        cdef BoundaryCondition ws_diffusive
        # projection scheme
        cdef BoundaryCondition pInit_dirichlet
        cdef BoundaryCondition pInc_dirichlet
        cdef BoundaryCondition pInit_advective
        cdef BoundaryCondition pInc_advective
        cdef BoundaryCondition pInit_diffusive
        cdef BoundaryCondition pInc_diffusive
        # clsvof
        cdef BoundaryCondition clsvof_dirichlet
        cdef BoundaryCondition clsvof_advective
        cdef BoundaryCondition clsvof_diffusive


        # functions
        # cpdef void reset(self)
        # cpdef void setNonMaterial(self)
        # cpdef void setTank(self)
        # cpdef void setFixedNodes(self)
        # cpdef void setNoSlip(self)
        # cpdef void setFreeSlip(self)
        # cpdef void setAtmosphere(self, double[:] orientation=*, double vof_air=*)
        # cpdef void setMoveMesh(self, double[:] last_pos, double[:] h=*, double[:,:] rot_matrix=*)
        @cython.locals(hx=np.ndarray)
        cpdef double[:] __cpp_MoveMesh_h(self, double[:] x, double t)
        cpdef double __cpp_MoveMesh_hx(self, double[:] x, double t)
        cpdef double __cpp_MoveMesh_hy(self, double[:] x, double t)
        cpdef double __cpp_MoveMesh_hz(self, double[:] x, double t)
#         @cython.locals(waveHeight=double, wavePhi=double,
#                        water_speed=double_memview1, x_max=double_memview1,
#                        u=cython.p_double, H=double)
        cdef double __cpp_UnsteadyTwoPhaseVelocityInlet_u_dirichlet(self, double[:] x, double t)
        cdef double __cpp_UnsteadyTwoPhaseVelocityInlet_v_dirichlet(self, double[:] x, double t)
        cdef double __cpp_UnsteadyTwoPhaseVelocityInlet_w_dirichlet(self, double[:] x, double t)
        cdef double __cpp_UnsteadyTwoPhaseVelocityInlet_phi_dirichlet(self, double[:] x, double t)
        cdef double __cpp_UnsteadyTwoPhaseVelocityInlet_vof_dirichlet(self, double[:] x, double t)
        cdef double __cpp_UnsteadyTwoPhaseVelocityInlet_p_advective(self, double[:] x, double t)

ctypedef double[:] (*cfvel) (double[3], double)  # pointer to velocity function
ctypedef double (*cfeta) (double[3], double)  # pointer to eta function
ctypedef double[:] (*cfvelrel) (RelaxationZone, double[3], double)  # pointer to velocity function of RelaxationZone class
ctypedef double (*cfphirel) (RelaxationZone, double[3])  # pointer to phi function of RelaxationZone class
ctypedef double (*cfphiwave) (RelaxationZone, double[3])  # pointer to phi function of RelaxationZone class
ctypedef np.float64_t float64_t
ctypedef np.int64_t int64_t

cdef class RelaxationZone:
    cdef double[:] wind_speed
    cdef double[:] u_calc  # calculated velocity
    # wave characteristics (if any)
    cdef cfvelrel uu
    cdef cfphirel phi
    cdef double mwl
    cdef cfeta eta
    cdef double waveHeight
    cdef double wavePhi
    cdef double waterSpeed
    cdef int nd
    cdef double[:] zero_vel
    cdef __cppClass_WavesCharacteristics waves
    cdef public:
        object Shape
        str zone_type
        double dragAlpha
        double dragBeta
        double porosity
        double epsFact_solid
        double epsFact_porous
        double vert_axis	
        double[:] center
        double[:] orientation
    cpdef void calculate_init(self)
    cdef double[:] calculate_vel(self, cython.p_double x, double t)
    @cython.locals(xx=double_memview1, waveHeight=double, wavePhi=double,
                   waterSpeed=double_memview1, x_max=double_memview1, H=double,
                   u=cython.double[3], o1=double, o2=double, o3=double)
    cdef double[:] __cpp_calculate_vel_wave(self, double* x, double t)
    cdef double[:] __cpp_calculate_vel_zero(self, double* x, double t)
    cdef double calculate_phi(self, double* x)
    @cython.locals(d1=double, d2=double, d3=double, phi=double,
                   o1=double, o2=double, o3=double)
    cdef double __cpp_calculate_phi_porous(self, double[3] x)
    cdef double __cpp_calculate_phi_porous_sponge(self, double[3] x)

cdef class RelaxationZoneWaveGenerator:
    cdef int nd  # dimension
    cdef int max_flag  # maximum region flag of relaxation zones (initialised in calculate_init)
    cdef np.ndarray zones_array  # zones array for fast access
    cdef public:
        dict zones  # zones dictionary
        object model  # model attached to zone
        object ar  #
    # cpdef RelaxationZoneWaveGenerator attachModel(self, dict, int)  # needed
    # cpdef void attachAuxiliaryVariables(self, dict)  # needed
    @cython.locals(zones=np.ndarray)
    cpdef void calculate_init(self)
    cpdef void calculate(self)
    @cython.locals(m=object, zone=RelaxationZone, mType=int, nE=int, nk=int,
                    qx=double_memview3, x=cython.double[3], t=double, nl=int,
                    q_phi_porous=double_memview2,  phi=double,
                   q_velocity_solid=double_memview3, u=double_memview1, mTypes=int_memview1)
    cdef void __cpp_iterate(self)  # main iteration loop


cdef class __cppClass_WavesCharacteristics:
    cdef int vert_axis  # vertical axis
    # relaxation zone info (if any)
    cdef double[:] center  # center of zone
    cdef double[:] orientation  # orientation of zone
    # zero array
    cdef double[:] zero_vel
    cdef double[:] _b_or
    cdef double[:] wind_speed
    cdef double vof_air
    cdef double vof_water
    cdef double smoothing
    cdef public:
        # wave class from WaveTools
        object WT
    @cython.locals(phi=double, waterSpeed=double_memview1,
                   H=double)
    cdef double[:]  __cpp_calculate_velocity(self, double* x, double t)
    @cython.locals(ux=double_memview1, b0=double, b1=double, b2=double, u0=double,
                   u1=double, u2=double)
    cdef double __cpp_calculate_pressure(self, double* x, double t)
    @cython.locals(level=double)
    cdef double __cpp_calculate_phi(self, double* x, double t)
    @cython.locals(phi=double, H=double)
    cdef double __cpp_calculate_vof(self, double* x, double t)
    @cython.locals(H=double)
    cdef double __cpp_calculate_smoothing_H(self, double phi)

cdef double* __x_to_cpp(double[:])
