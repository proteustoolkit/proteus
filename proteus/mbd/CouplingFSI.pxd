# cython: language_level=3
cimport numpy as np
import numpy as np
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp.memory cimport (shared_ptr)
# chrono C++ headers
cimport ChronoHeaders as ch


cdef extern from "swigpyobject.h":
    ctypedef struct SwigPyObject:
        void *ptr

cdef extern from "ProtChMoorings.h":
    shared_ptr[ch.ChPhysicsItem] getPhysicsItemSharedPtr(ch.ChPhysicsItem* item)
    shared_ptr[ch.ChPhysicsItem] getPhysicsItemSharedPtr2(ch.ChPhysicsItem* item)
    cdef cppclass cppCable:
        ch.ChSystem& system
        ch.ChMesh& mesh
        vector[shared_ptr[ch.ChNodeFEAxyzD]] nodes
        vector[shared_ptr[ch.ChNodeFEAxyzrot]] nodesRot
        vector[shared_ptr[ch.ChVector3d]] forces_drag
        vector[shared_ptr[ch.ChVector3d]] forces_addedmass
        double L0
        double length
        int nb_elems
        bool applyDrag
        bool applyAddedMass
        bool applyBuoyancy
        vector[ch.ChVector3d] mvecs
        vector[ch.ChVector3d] mvecs_tangents
        void buildNodes()
        void buildMaterials()
        void buildElements()
        void buildMesh()
        void updateDragForces()
        void updateBuoyancyForces()
        void setDragCoefficients(double Cd_axial, double Cd_normal)
        void setAddedMassCoefficients(double Cd_axial, double Cd_normal)
        void setRestLengthPerElement(vector[double] length_array);
        void setIyy(double Iyy_in)
    cdef cppclass cppMultiSegmentedCable:
        ch.ChSystem& system
        ch.ChMesh& mesh
        int nb_nodes_tot
        int nb_elems_tot
        vector[shared_ptr[cppCable]] cables
        vector[shared_ptr[ch.ChNodeFEAxyzD]] nodes
        vector[shared_ptr[ch.ChNodeFEAxyzrot]] nodesRot
        vector[shared_ptr[ch.ChElementCableANCF]] elemsCableANCF
        vector[shared_ptr[ch.ChElementBeamEuler]] elemsBeamEuler
        shared_ptr[ch.ChLinkPointFrame] constraint_back
        shared_ptr[ch.ChLinkPointFrame] constraint_front
        vector[shared_ptr[ch.ChVector3d]] forces_drag
        vector[shared_ptr[ch.ChVector3d]] forces_addedmass
        shared_ptr[ch.ChContactMaterialSMC] contact_material
        void buildNodes()
        void buildElements()
        void buildCable()
        # void setVelocityAtNodes(double* fluid_velocity)
        void attachFrontNodeToBody(shared_ptr[ch.ChBody])
        void attachBackNodeToBody(shared_ptr[ch.ChBody])
        void updateDragForces()
        void updateAddedMassForces()
        void applyForces()
        void updateBuoyancyForces()
        void setFluidVelocityAtNodes(vector[ch.ChVector3d] fluid_velocity)
        void setFluidAccelerationAtNodes(vector[ch.ChVector3d] fluid_acceleration)
        void setFluidDensityAtNodes(vector[double] dens)
        void setContactMaterial(shared_ptr[ch.ChContactMaterialSMC] material)
        ch.ChVector3d getTensionElement(int i, double eta)
    cppMultiSegmentedCable * newMoorings(shared_ptr[ch.ChSystem] system,
                                         shared_ptr[ch.ChMesh] mesh,
                                         vector[double] length,
                                         vector[int] nb_elems,
                                         vector[double] d,
                                         vector[double] rho,
                                         vector[double] E,
                                         string beam_type
        )
    void cppAttachNodeToNodeFEAxyzD(cppMultiSegmentedCable* cable1,
                                    int node1,
                                    cppMultiSegmentedCable* cable2,
                                    int node2)
    void cppAttachNodeToNodeFEAxyzrot(cppMultiSegmentedCable* cable1,
                                      int node1,
                                      cppMultiSegmentedCable* cable2,
                                      int node2)


cdef extern from "ProtChBody.h":
    cdef cppclass cppSystem:
        shared_ptr[ch.ChSystemSMC] systemSMC
        shared_ptr[ch.ChSystem] system
        double chrono_dt
        void DoStepDynamics(dt)
        void step(double proteus_dt, int n_substeps)
        void setDirectory(string directory)
        void setTimestepperType(string tstype, bool verbose)
        void setCollisionEnvelopeMargin(double envelope, double margin)
        void addMesh(shared_ptr[ch.ChMesh] mesh)
        vector[double] stateGather() except +
        void stateScatter(vector[double] y) except +
    cppSystem * newSystem()
    cdef cppclass cppRigidBody:
        shared_ptr[ch.ChBody] body
        double mass
        ch.ChVector3d pos
        ch.ChVector3d pos_last
        ch.ChVector3d vel
        ch.ChVector3d vel_last
        ch.ChVector3d acc
        ch.ChVector3d acc_last
        ch.ChVector3d angvel
        ch.ChVector3d angvel_last
        ch.ChVector3d angacc
        ch.ChVector3d angacc_last
        # ChVector3d inertia
        ch.ChMatrix33 rotm
        ch.ChMatrix33 rotm_last
        ch.ChQuaternion rotq
        ch.ChQuaternion rotq_last
        ch.ChVector3d free_x
        ch.ChVector3d free_r
        ch.ChVector3d F
        ch.ChVector3d F_last
        ch.ChVector3d M
        ch.ChVector3d M_last
        shared_ptr[ch.ChTriangleMeshConnected] trimesh
        bool has_trimesh
        vector[ch.ChVector3d] trimesh_pos
        vector[ch.ChVector3d] trimesh_pos0
        ch.ChVector3d pos0_trimesh
        ch.ChQuaternion rotq0_trimesh
        cppRigidBody(cppSystem* system)
        void calculate_init()
        void addAccumulators()
        void prestep(double* force, double* torque)
        void poststep()
        ch.ChVector3d hxyz(double* x, double dt)
        double hx(double* x, double dt)
        double hy(double* x, double dt)
        double hz(double* x, double dt)
        void addSpring(double stiffness,
                       double damping,
                       double* fairlead,
                       double* anchor,
                       double rest_length)
        void addPrismaticLinksWithSpring(double* pris1,
                                         double* pris2,
                                         double stiffness,
                                         double damping,
                                         double rest_length);
        void addPrismaticLinkX(double* pris1);
        void setConstraints(double* free_x, double* free_r)
        void setName(string name)
        void setPrescribedMotionCustom(vector[double] t, vector[double] x,
                                       vector[double] y, vector[double] z,
                                       vector[double] ang, vector[double] ang2,
                                       vector[double] ang3, double t_max)
        void setPrescribedMotionPoly(double coeff1)
        void setPrescribedMotionSine(double a, double f)
        void updateTriangleMeshVisualisationPos()
    cppRigidBody * newRigidBody(cppSystem* system)
    void ChLinkLockBodies(shared_ptr[ch.ChBody] body1,
                          shared_ptr[ch.ChBody] body2,
                          shared_ptr[ch.ChSystem] system,
                          ch.ChCoordsys coordsys,
                          double limit_X,
                          double limit_Y,
                          double limit_Z,
                          double limit_Rx,
                          double limit_Ry,
                          double limit_Rz)

cdef extern from "swigpyobject.h":
    ctypedef struct SwigPyObject:
        void *ptr


cdef class ProtChBody:
    cdef cppRigidBody * thisptr
    cdef ch.ChQuaternion rotation
    cdef ch.ChQuaternion rotation_last
    cdef vector[ch.ChVector3d] trimesh_nodes
    cdef vector[ch.ChTriangle] trimesh_triangles
    cdef public:
      str record_file
      object model
      ProtChSystem ProtChSystem
      object Shape
      int nd
      double dt
      double width_2D
      object record_dict
      object prescribed_motion_function
      object ChBody
      ChBodyAddedMass ChBodyAddedMass
      np.ndarray position
      np.ndarray position_last
      np.ndarray F  # force as retrieved from Chrono
      np.ndarray M  # moment as retreived from Chrono
      np.ndarray F_last
      np.ndarray M_last
      np.ndarray F_prot  # force retrieved from Proteus (fluid)
      np.ndarray M_prot  # moment retrieved from Proteus (fluid)
      np.ndarray F_prot_last
      np.ndarray M_prot_last
      np.ndarray F_applied  # force applied and passed to Chrono
      np.ndarray M_applied  # moment applied and passed to Chrono
      np.ndarray F_applied_last
      np.ndarray M_applied_last
      np.ndarray F_Aij
      np.ndarray M_Aij
      np.ndarray F_Aij_last
      np.ndarray M_Aij_last
      np.ndarray acceleration
      np.ndarray acceleration_last
      np.ndarray velocity
      np.ndarray velocity_last
      np.ndarray ang_acceleration_last
      np.ndarray ang_acceleration
      np.ndarray ang_velocity_last
      np.ndarray ang_velocity
      double ang_vel_norm
      double ang_vel_norm_last
      np.ndarray barycenter0
      np.ndarray rotation_init
      np.ndarray rotm
      np.ndarray rotm_last
      np.ndarray rotq
      np.ndarray rotq_last
      np.ndarray adams_vel
      string name
      bool predicted
      double dt_predict
      np.ndarray h_predict  # predicted displacement
      double h_ang_predict  # predicted angular displacement (angle)
      np.ndarray h_ang_vel_predict  # predicted angular velocity
      np.ndarray h_predict_last  # predicted displacement
      double h_ang_predict_last  # predicted angular displacement (angle)
      np.ndarray h_ang_vel_predict_last  # predicted angular velocity
      np.ndarray Aij  # added mass array
      bool applyAddedMass  # will apply added mass if True (default)
      bool Aij_updated_global  # True if added-mass was updated in global coords
      bool Aij_transform_local  # True for transforming global Aij to local
      bool useIBM
      double radiusIBM
      object sdfIBM
      string hdfFileName
      double Aij_factor
      np.ndarray boundaryFlags
    cdef np.ndarray callPrescribedMotion(self, double t)


cdef class ProtChSystem:
    cdef cppSystem * thisptr
    cdef double proteus_dt
    cdef double proteus_dt_last
    cdef double proteus_dt_next
    cdef string directory
    cdef object u
    cdef int nd
    cdef object femSpace_velocity
    cdef object femSpace_pressure
    cdef object nodes_kdtree
    cdef int min_nb_steps
    cdef double dt_fluid_next
    cdef double dt
    cdef double dt_last
    cdef double t
    cdef vector[shared_ptr[ch.ChPhysicsItem]] myphysicsitem
    cdef vector[shared_ptr[ch.ChBody]] mybodies
    cdef public:
        object ChSystemSMC
        object ChSystem
        object model
        object model_module
        double dt_init
        double dt_fluid
        double dt_fluid_last
        cdef object subcomponents
        double chrono_dt
        bool build_kdtree
        bool dist_search
        bool first_step
        string scheme  # coupling scheme
        string prediction  # force for prediction
        int step_nb  # number of steps
        int step_start  # starting step
        double sampleRate
        double next_sample
        bool record_values
        object model_mesh
        object model_addedmass
        ProtChAddedMass ProtChAddedMass
        int tCount
        bool initialized
        bool update_substeps
        int nBodiesIBM
        object log_chrono_bodies
        object log_chrono_springs
        object log_chrono_residuals
        string log_chrono_format


cdef class ProtChMesh:
    cdef shared_ptr[ch.ChMesh] mesh
    cdef public:
        object ChMeshh
        ProtChSystem ProtChSystem


cdef class ProtChMoorings:
    cdef cppMultiSegmentedCable * thisptr
    cdef public:
      # vector[pyfea.ChNodeFEAxyzD] nodes
      # vector[pyfea.ChElementCableANCF] elems
      # vector[ProtChCable] cables
      str record_file
      object model
      ProtChSystem ProtChSystem
      object Mesh
      int nd
      object nodes_function
      object nodes_function_tangent
      object fluid_velocity_function
      ProtChBody body_front
      ProtChBody body_back
      bool front_body
      bool back_body
      bool nodes_built
      bool external_forces_from_ns
      bool external_forces_manual
      np.ndarray fluid_density_array
      np.ndarray fluid_velocity_array
      np.ndarray fluid_velocity_array_previous
      np.ndarray fluid_acceleration_array
      string name
      string beam_type
      int nodes_nb # number of nodes
      np.ndarray nb_elems
      double[:] _record_etas
      object _record_names
      object _record_etas_names
      bool initialized
      int[:] nearest_node_array
      int[:] containing_element_array
      int[:] owning_rank
      string hdfFileName
      double[:] tCount_value
      int tCount
      object nodes
      object elements


cdef class ProtChAddedMass:
    cdef public:
        object model
        ProtChSystem ProtChSystem


cdef class ChBodyAddedMass:
    cdef shared_ptr[ch.ChBodyAddedMass] sharedptr
    cdef shared_ptr[ch.ChBody] sharedptr_chbody
    cdef ch.ChBodyAddedMass * thisptr
    cdef ch.ChBody * bodyptr
    cdef public:
        object ChBodySWIG
    cdef void SetMfullmass(self, ch.ChMatrixDynamic Mfullmass_in)
    # cdef void SetInvMfullmass(self, ch.ChMatrixDynamic inv_Mfullmass_in)
