#!python
# distutils: language = c++
# cython: profile=True, binding=True, embedsignature=True
# cython: language_level=3
"""
Coupling between Chrono and Proteus is done in this file.

Objects (classes) starting with 'ProtCh' (e.g. ProtChBody) are objects that
have logic specifically developed for communication between Proteus and Chrono.

Objects starting with 'Ch' (e.g. ChBody) are objects that only have Chrono
logic associated to them.

Some ProtCh objects give access to the Chrono object:
my_protchsystem = ProtChSystem()
my_protchbody = ProtChBody(system=my_protchsystem)
my_chbody = my_protchbody.ChBody
my_chbody.SetPos(...)
my_chbody.SetRot(...)

# pass the index of the boundaries (or particle index) where forces must be integrated
my_protchbody.setIndexBoundary([0, 1, 2, 3])
# alternatively, if you use a Shape instance from proteus.SpatialTools
# the boundaries indice will be set automatically after calling SpatialTools.assembleDomain()
my_protchbody.setShape(my_shape)
"""


import os
import sys
import csv
import copy
cimport numpy as np
import numpy as np
from mpi4py import MPI
from scipy import spatial
from proteus import AuxiliaryVariables, Archiver, Comm, Profiling
from proteus import SpatialTools as st
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp.memory cimport (shared_ptr,
                            make_shared)
from collections import OrderedDict
from cython.operator cimport dereference as deref
# from cython.cpp cimport static_cast
import xml.etree.ElementTree as ET
import h5py
# chrono C++ headers
cimport ChronoHeaders as ch
# chrono Python headers
from proteus.mprans import BodyDynamics as bd
import pychrono as chrono
from pychrono import fea as chrono_fea


def indentXML(elem, level=0):
    """Pretty-print an ElementTree in place.

    This module still writes its own XDMF for the bodies and cables it
    moves (one ``.xmf`` per object, with the per-step fragments stored in
    the HDF5 file the way proteus <= 1.9.x did). proteus's own archive no
    longer builds XML, so the helper that used to live in Archiver lives
    here until these writers go through the archive too.
    """
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for e in elem:
            indentXML(e, level+1)
            if not e.tail or not e.tail.strip():
                e.tail = i + "  "
        if not e.tail or not e.tail.strip():
            e.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

# needed for sphinx docs
__all__ = ['ProtChSystem',
           'ProtChBody',
           'ProtChMesh',
           'ProtChMoorings',
           'ProtChAddedMass']


cdef class ProtChBody:

    def __cinit__(self,
                  ProtChSystem system=None):
        self.ProtChSystem = system
        # create new cppRigidBody
        self.thisptr = newRigidBody(system.thisptr)
        self.ChBodyAddedMass = ChBodyAddedMass()
        self.ChBody = self.ChBodyAddedMass.ChBodySWIG
        self.thisptr.body = self.ChBodyAddedMass.sharedptr_chbody  # give pointer to cpp class
        # cppRigidBody's constructor got force/torque accumulator indices from
        # its own (now-discarded) placeholder ChBody -- get fresh ones for the
        # body actually assigned above, or prestep() segfaults on first use
        # indexing into the new body's own (empty) accumulators vector.
        self.thisptr.addAccumulators()
        # # add body to system
        if system is not None:
            self.ProtChSystem.addProtChBody(self)
        # initialise values
        self.record_dict = OrderedDict()
        self.F_prot = np.zeros(3)  # initialise empty Proteus force
        self.M_prot = np.zeros(3)  # initialise empty Proteus moment
        self.F_applied = np.zeros(3)  # initialise empty Applied force
        self.M_applied = np.zeros(3)  # initialise empty Applied moment
        self.F_Aij = np.zeros(3)  # initialise empty added mass force
        self.M_Aij = np.zeros(3)  # initialise empty added mass moment
        self.prescribed_motion_function = None
        self.acceleration = np.zeros(3)
        self.acceleration_last = np.zeros(3)
        self.velocity = np.zeros(3)
        self.velocity_last = np.zeros(3)
        self.ang_velocity = np.zeros(3)
        self.position = np.zeros(3)
        self.position_last = np.zeros(3)
        self.ang_vel_norm = 0.  # used for mesh disp prediction
        self.ang_vel_norm_last = 0.  # used for mesh disp prediction
        self.h_predict = np.zeros(3)
        self.h_ang_predict = 0.
        self.h_ang_vel_predict = np.zeros(3)
        self.h_predict_last = np.zeros(3)
        self.h_ang_predict_last = 0.
        self.h_ang_vel_predict_last = np.zeros(3)
        self.predicted = False
        self.adams_vel = np.zeros((5, 3))
        self.Aij = np.zeros((6, 6))  # added mass array
        self.applyAddedMass = True  # will apply added mass in Chrono calculations if True
        self.useIBM = False
        self.Aij_factor = 1.
        self.Aij_updated_global = False
        self.Aij_transform_local = False
        self.boundaryFlags = np.empty(0, 'i')
        self.setName(b'rigidbody')

    def getChronoObject(self):
        return self.ChBody

    def attachShape(self,
                    shape,
                    take_shape_name=True):
        """Attach proteus.SpatialTools shape to body.
        Used for automatic calculation of external forces from Proteus.
        Called automatically when creating a body and passing a shape
        instance.

        Parameters
        ----------
        shape: SpatialTools.Shape
            instance of shape from proteus.SpatialTools or
            proteus.mprans.SpatialTools
        """
        assert self.Shape is None, 'Shape '+self.Shape.name+' was already attached'
        self.Shape = shape
        if 'ChRigidBody' not in shape.auxiliaryVariables:
            shape.auxiliaryVariables['ChRigidBody'] = self
            if take_shape_name is True:
                self.setName(bytes(shape.name,'utf-8'))
        self.nd = shape.Domain.nd
        new_vec = chrono.ChVector3d(shape.barycenter[0],
                                   shape.barycenter[1],
                                   shape.barycenter[2])
        self.ChBody.SetPos(new_vec)

    def addTriangleMeshFromShape(self,
                                 object shape=None,
                                 double[:] pos=None,
                                 double[:,:] rot=None,
                                 bool is_static=False,
                                 bool is_convex=False,
                                 double sphereswept_thickness=0.005):
        """Adds triangle mesh to collision model and for IBM calculations
        """
        if shape is None:
            shape = self.Shape
        assert shape is not None, 'no Shape was defined for making a triangle mesh'
        vertices = shape.vertices
        for f_i, facet in enumerate(shape.facets):
            f = facet[0]
            assert len(f) == 3, 'Facets must be triangles for triangle mesh but facet '+str(f_i)+' is not of length 3'
        facets = np.array(shape.facets, dtype=np.int32)
        self.addTriangleMeshFromVerticesFaces(vertices=vertices,
                                              facets=facets,
                                              pos=pos,
                                              rot=rot,
                                              is_static=is_static,
                                              is_convex=is_convex,
                                              sphereswept_thickness=sphereswept_thickness)

    # def addTriangleMeshFromVerticesFaces(self,
    #                                      double[:,:] vertices,
    #                                      int[:,:,:] facets,
    #                                      double[:] pos=None,
    #                                      double[:,:] rot=None,
    #                                      bool is_static=False,
    #                                      bool is_convex=False,
    #                                      double sphereswept_thickness=0.005):
    #     """Adds triangle mesh to collision model and for IBM calculations
    #     """
    #     self.thisptr.trimesh = make_shared[ch.ChTriangleMeshConnected]()
    #     self.trimesh_nodes.clear()
    #     self.trimesh_triangles.clear()
    #     for v in vertices:
    #         self.trimesh_nodes.push_back(ch.ChVector(v[0], v[1], v[2]))
    #     for f_i, facet in enumerate(facets):
    #         f = facet[0]
    #         assert len(f) == 3, 'Facets must be triangles for triangle mesh but facet '+str(f_i)+' is not of length 3'
    #         self.trimesh_triangles.push_back(ch.ChTriangle(self.trimesh_nodes.at(f[0]),
    #                                                        self.trimesh_nodes.at(f[1]),
    #                                                        self.trimesh_nodes.at(f[2])))
    #         deref(self.thisptr.trimesh).addTriangle(self.trimesh_triangles.at(f_i))
    #     if pos is None:
    #         pos = np.zeros(3)
    #     cdef ch.ChMatrix33 rotmat
    #     if rot is None:
    #         rot = np.eye(3)
    #     for i in range(rot.shape[0]):
    #         for j in range(rot.shape[1]):
    #             rotmat.SetElement(i, j, rot[i, j])
    #     # deref(deref(self.thisptr.body).GetCollisionModel()).ClearModel()
    #     deref(deref(self.thisptr.body).GetCollisionModel()).AddTriangleMesh(<shared_ptr[ch.ChTriangleMesh]> self.thisptr.trimesh,
    #                                                                         is_static,
    #                                                                         is_convex,
    #                                                                         ch.ChVector(pos[0],
    #                                                                                     pos[1],
    #                                                                                     pos[2]),
    #                                                                         rotmat,
    #                                                                         sphereswept_thickness)
    #     self.thisptr.has_trimesh = True
    #     cdef ch.ChVector pos0 = deref(self.thisptr.body).GetPos()
    #     self.thisptr.pos0_trimesh = pos0
    #     cdef ch.ChQuaternion rot0 = deref(self.thisptr.body).GetRot()
    #     self.thisptr.rotq0_trimesh = rot0

    # # (!) # cannot use right now because of cython error when C++ function has default
    # # (!) # arguments (known bug in cython community, silent error)
    # # (!) # logic left here for when cython bug is solved
    # def addTriangleMeshFromWavefront(self,
    #                                  string filename,
    #                                  bool load_normals=True,
    #                                  bool load_uv=False,
    #                                  double[:] pos=None,
    #                                  double[:,:] rot=None,
    #                                  bool is_static=False,
    #                                  bool is_convex=False,
    #                                  double sphereswept_thickness=0.005):
    #     """Adds triangle mesh to collision model and for IBM calculations
    #     """
    #     self.thisptr.trimesh.LoadWavefrontMesh(filename,
    #                                            load_normals,
    #                                            load_uv)
    #     if pos is None:
    #         pos = np.zeros(3)
    #     cdef ch.ChMatrix33 rotmat
    #     if rot is None:
    #         rot = np.eye(3)
    #     for i in range(rot.shape[0]):
    #         for j in range(rot.shape[1]):
    #             rotmat.SetElement(i, j, rot[i, j])
    #     deref(deref(self.thisptr.body).GetCollisionModel()).AddTriangleMesh(self.thisptr.trimesh,
    #                                                                         is_static,
    #                                                                         is_convex,
    #                                                                         ch.ChVector(pos[0],
    #                                                                                     pos[1],
    #                                                                                     pos[2]),
    #                                                                         rotmat,
    #                                                                         sphereswept_thickness)

    def getTriangleMeshInfo(self):
        # vertices
        cdef vector[ch.ChVector3d] chpos = deref(self.thisptr.trimesh).GetCoordsVertices()
        cdef double[:,:] pos = np.zeros((chpos.size(),3 ))
        for i in range(chpos.size()):
            pos[i, 0] = chpos.at(i).x()
            pos[i, 1] = chpos.at(i).y()
            pos[i, 2] = chpos.at(i).z()
        cdef vector[ch.ChVector3i] chel_connect = deref(self.thisptr.trimesh).GetIndicesVertexes()
            # connection of vertices
        cdef int[:,:] el_connect = np.zeros((chel_connect.size(), 3), dtype=np.int32)
        for i in range(chel_connect.size()):
            el_connect[i, 0] = int(chel_connect.at(i).x())
            el_connect[i, 1] = int(chel_connect.at(i).y())
            el_connect[i, 2] = int(chel_connect.at(i).z())
        return pos, el_connect

    def setCollisionOptions(self,
                            double envelope=0.001,
                            double margin=0.0005,
                            bool collide=True):
        deref(self.thisptr.body).EnableCollision(collide)
        deref(deref(self.thisptr.body).GetCollisionModel()).SetEnvelope(envelope)
        deref(deref(self.thisptr.body).GetCollisionModel()).SetSafeMargin(margin)

    def setBoundaryFlags(self, flags):
        """Sets the flags of the boundaries of the body
        numbers must be gloabal (from domain.segmentFlags or
        domain.facetFlags).

        Parameters
        ----------
        flags: array_like
            list of flags that belong to this body
        """
        self.boundaryFlags = np.array(flags, 'i')

    def setIBM(self, useIBM=True, radiusIBM=0., sdfIBM=None):
        """Sets IBM mode for retrieving fluid forces

        Parameters
        ----------
        useIBM: bool
            set if IBM should be used
        radiusIBM: double
            radius of the particle for IBM
        radiusIBM: double
            sdf relative to barycentre of body for IBM
        """
        self.useIBM = useIBM
        self.radiusIBM = radiusIBM
        self.sdfIBM = sdfIBM

    def setWidth2D(self, width):
        """Sets width of 2D body (for forces and moments calculation)

        Parameters
        ----------
        width: float
            width of the body
        """
        self.width_2D = width

    def attachAuxiliaryVariables(self,avDict):
        pass

    def setInitialRot(self, rot):
        cdef np.ndarray zeros = np.zeros(3)
        self.rotation_init = rot
        self.thisptr.prestep(<double*> zeros.data,
                             <double*> zeros.data)
        if self.rotation_init is not None:
            new_quat = chrono.ChQuaternionD(rot[0], rot[1], rot[2], rot[3])
            self.ChBody.SetRot(new_quat)
        self.thisptr.poststep()

    def hxyz(self, np.ndarray x, double t, debug=False):
        cdef np.ndarray h
        cdef np.ndarray xx
        cdef double ang, ang_last
        cdef np.ndarray d_tra, d_tra_last # translational displacements
        cdef np.ndarray d_rot, d_rot_last # rotational displacements
        cdef np.ndarray h_body  # displacement from body
        cdef ch.ChVector3d h_body_vec
        h = np.zeros(3)
        if self.predicted is False:
            self.prediction()
        # if self.ProtChSystem.thisptr.system.GetChTime() > 0.0003:
        # if self.ProtChSystem.step_nb > self.ProtChSystem.step_start:
            # self.ChBody.SetBodyFixed(False)
        if self.ProtChSystem.scheme == b"CSS":
            h_body_vec = self.thisptr.hxyz(<double*> x.data, t)
            h_body = np.array([h_body_vec.x(), h_body_vec.y(), h_body_vec.z()])
            h += h_body
        elif self.ProtChSystem.scheme == b"ISS":
            # remove previous prediction
            # translate back first
            if debug is True:
                print("$$$$$$$$$$$$$$$$$$")
                print("x: ", x)
            # d_tra_last = -self.velocity_last*dt_half_last
            # h += d_tra_last
            h += -self.h_predict_last
            if debug is True:
                print("h_predict_last: ", self.h_predict_last)
                print("h_predict: ", self.h_predict)
                print("h_ang_predict_last: ", self.h_ang_predict_last)
                print("h_ang_predict: ", self.h_ang_predict)
                print("h_ang_vel_predict_last: ", self.h_ang_vel_predict_last)
                print("h_ang_vel_predict: ", self.h_ang_vel_predict)
            # rotate back
            # ang_last = -self.ang_vel_norm_last*dt_half_last
            ang_last = -self.h_ang_predict_last
            if ang_last > 0:
                d_rot_last = (st.rotation3D(points=x+h,  # (translated back)
                                            rot=ang_last,
                                            axis=self.h_ang_vel_predict_last,
                                            pivot=self.position_last)
                            -x+h)
                h += d_rot_last
            # add rigid body displacement
            xx = x+h # previous position of body
            if debug is True:
                print("x_old: ", x+h)
                print("F_applied: ", self.F_applied)
            h_body_vec = self.thisptr.hxyz(<double*> xx.data, t)
            h_body = np.array(h_body_vec.x(), h_body_vec.y(), h_body_vec.z())
            h += h_body
            # add current prediction
            # rotate first
            # ang = self.ang_vel_norm*dt_half
            ang = self.h_ang_predict
            if debug is True:
                print("x_body: ", x+h)
                print("pos_body: ", self.ChBody.GetPos())
            if ang > 0:
                d_rot = (st.rotation3D(points=x+h,
                                    rot=ang,
                                    axis=self.h_ang_vel_predict,
                                    pivot=self.position)
                        -x+h)
                h += d_rot
            if debug is True:
                print("x_new_rot: ", x+h)
            # translate
            # d_tra = self.velocity*dt_half
            # h += d_tra
            h += self.h_predict
            if debug is True:
                print("x_new_trarot: ", x+h)
        comm = Comm.get().comm.tompi4py()
        # print(comm.rank, h, x, t)
        return h

    def hx(self, np.ndarray x, double t):
        """BC function for mesh nodes displacement (x component)

        Parameters
        ----------
        x: array_like
            coordinates of the node before displacement
        t: double
            simulation time
        """
        return self.hxyz(x, t)[0]

    def hy(self, np.ndarray x, double t):
        """BC function for mesh nodes displacement (y component)

        Parameters
        ----------
        x: array_like
            coordinates of the node before displacement
        t: double
            simulation time
        """
        return self.hxyz(x, t)[1]

    def hz(self, np.ndarray x, double t):
        """BC function for mesh nodes displacement (z component)

        Parameters
        ----------
        x: array_like
            coordinates of the node before displacement
        t: double
            simulation time
        """
        return self.hxyz(x, t)[2]

    def hx_translation(self, np.ndarray x, double t):
        """BC function for mesh nodes displacement (x component)

        Parameters
        ----------
        x: array_like
            coordinates of the node before displacement
        t: double
            simulation time
        """
        return (self.position-self.position_last)[0]

    def hy_translation(self, np.ndarray x, double t):
        """BC function for mesh nodes displacement (y component)

        Parameters
        ----------
        x: array_like
            coordinates of the node before displacement
        t: double
            simulation time
        """
        return (self.position-self.position_last)[1]

    def hz_translation(self, np.ndarray x, double t):
        """BC function for mesh nodes displacement (z component)

        Parameters
        ----------
        x: array_like
            coordinates of the node before displacement
        t: double
            simulation time
        """
        return (self.position-self.position_last)[2]

    def hx_rotation(self, np.ndarray x, double t):
        """BC function for mesh nodes displacement (x component)

        Parameters
        ----------
        x: array_like
            coordinates of the node before displacement
        t: double
            simulation time
        """
        return self.hxyz(x, t)[0]-(self.position-self.position_last)[0]

    def hy_rotation(self, np.ndarray x, double t):
        """BC function for mesh nodes displacement (y component)

        Parameters
        ----------
        x: array_like
            coordinates of the node before displacement
        t: double
            simulation time
        """
        return self.hxyz(x, t)[1]-(self.position-self.position_last)[1]

    def hz_rotation(self, np.ndarray x, double t):
        """BC function for mesh nodes displacement (z component)

        Parameters
        ----------
        x: array_like
            coordinates of the node before displacement
        t: double
            simulation time
        """
        return self.hxyz(x, t)[2]-(self.position-self.position_last)[2]

    def addSpring(self, double stiffness, double damping, np.ndarray fairlead,
                  np.ndarray anchor, double rest_length):
        self.thisptr.addSpring(stiffness, damping, <double*> fairlead.data,
                               <double*> anchor.data, rest_length)

    def setConstraints(self, np.ndarray free_x, np.ndarray free_r):
        """Sets constraints on the body
        (!) Only acts on Proteus and gravity forces

        Parameters
        ----------
        free_x: array_like
            Translational constraints.
        free_r: array_like
            Rotational constraints.
        """
        self.thisptr.setConstraints(<double*> free_x.data, <double*> free_r.data)

    def setAddedMass(self, np.ndarray Aij):
        """
        Sets the added mass matrix of the body

        Parameters
        ----------
        added_mass: array_like
            Added mass matrix (must be 6x6 array!)
        """

        assert Aij.shape[0] == Aij.shape[1] == 6, 'Added mass matrix must be 6x6 (np)'
        cdef double mass = self.ChBody.GetMass()
        cdef np.ndarray iner = pymat332array(self.ChBody.GetInertia())
        cdef np.ndarray MM = np.zeros((6,6))  # mass matrix
        cdef np.ndarray FM = np.zeros((6,6))  # full mass matrix

        # added mass matrix
        cdef np.ndarray rotMarr_big
        cdef np.ndarray rotMarrT_big
        # store Aij in global frame
        cdef np.ndarray Aij_global = np.zeros((6, 6))
        Aij_global[:] = Aij[:]
        # transform Aij in local frame
        if self.Aij_updated_global is True and self.Aij_transform_local is True:
            # converting from global to local: Rot*Aij*RotT*v
            rot = self.ChBodyAddedMass.GetRot()
            rotch = chrono.ChMatrix33D(rot)
            rotMarr_big = np.zeros((6, 6))
            rotMarrT_big = np.zeros((6, 6))
            for i in range(6):
                for j in range(6):
                    if i < 3 and j < 3 :
                        rotMarr_big[i, j] = rotch.getitem(i, j)
                        rotMarrT_big[i, j] = rotch.getitem(j, i)
                    elif i >=3 and j >= 3:
                        rotMarr_big[i, j] = rotch.getitem(i-3, j-3)
                        rotMarrT_big[i, j] = rotch.getitem(j-3, i-3)
            # self.Aij[:] = np.matmul(rotMarr_big, np.matmul(Aij, rotMarrT_big))
            self.Aij[:] = rotMarrT_big.dot(Aij).dot(rotMarr_big)
        else:
            self.Aij[:] = Aij
        # remove terms from restrained DOFs
        self.Aij[0, 1:] *= self.thisptr.free_x.x()
        self.Aij[1, 0] *= self.thisptr.free_x.y()
        self.Aij[1, 2:] *= self.thisptr.free_x.y()
        self.Aij[2, :2] *= self.thisptr.free_x.z()
        self.Aij[2, 3:] *= self.thisptr.free_x.z()
        self.Aij[3, :3] *= self.thisptr.free_r.x()
        self.Aij[3, 4:] *= self.thisptr.free_r.x()
        self.Aij[4, :4] *= self.thisptr.free_r.y()
        self.Aij[4, 5] *= self.thisptr.free_r.y()
        self.Aij[5, :5] *= self.thisptr.free_r.z()
        # mass matrix
        MM[0,0] = mass
        MM[1,1] = mass
        MM[2,2] = mass
        for i in range(3):
            for j in range(3):
                MM[i+3, j+3] = iner[i, j]
        # full mass
        FM += self.Aij
        FM += MM
        Profiling.logEvent('Mass Matrix:\n'+str(MM))
        Profiling.logEvent('Added Mass Matrix:\n'+str(self.Aij))
        Profiling.logEvent('Full Mass Matrix:\n'+str(FM))
        # inverse of full mass matrix
        inv_FM = np.linalg.inv(FM)
        #set it to chrono variable
        chFM = chrono.ChMatrixDynamicd(6, 6)
        inv_chFM = chrono.ChMatrixDynamicd(6, 6)
        for i in range(6):
            for j in range(6):
                chFM.SetItem(i, j, FM[i, j])

        # hack for swig
        cdef SwigPyObject *swig_obj = <SwigPyObject*> chFM.this
        cdef ch.ChMatrixDynamic *mycpp_ptr = <ch.ChMatrixDynamic*?>swig_obj.ptr
        cdef ch.ChMatrixDynamic my_instance = deref(mycpp_ptr)

        # set full mass matrix
        self.ChBodyAddedMass.SetMfullmass(my_instance)

        aa = np.zeros(6)

        aa[:3] = pyvec2array(self.ChBody.GetPosDt2())
        aa[3:] = pyvec2array(self.ChBody.GetAngAccParent())
        Aija = np.dot(Aij_global, aa)
        self.F_Aij = Aija[:3]
        self.M_Aij = Aija[3:]

    def getPressureForces(self):
        """Gives pressure forces from fluid (Proteus) acting on body.
        (!) Only works during proteus simulation

        Returns
        -------
        F_p: array_like
            pressure forces (x, y, z) as provided by Proteus
        """
        F_p = np.zeros(3)
        for flag in self.boundaryFlags:
            if self.useIBM:
                F_p += self.ProtChSystem.model.levelModelList[-1].coefficients.particle_netForces[flag]
            else:
                F_p += self.ProtChSystem.model.levelModelList[-1].coefficients.netForces_p[flag]
        return F_p

    def getShearForces(self):
        """Gives shear forces from fluid (Proteus) acting on body
        (!) Only works during proteus simulation

        Returns
        -------
        F_v: array_like
            shear forces (x, y, z) as provided by Proteus
        """
        F_v = np.zeros(3)
        if self.useIBM:
            pass
        else:
            for flag in self.boundaryFlags:
                F_v += self.ProtChSystem.model.levelModelList[-1].coefficients.netForces_v[flag]
        return F_v

    def getMoments(self):
        """Gives moments from fluid (Proteus) acting on body
        (!) Only works during proteus simulation

        Returns
        -------
        M: array_like
            moments (x, y, z) as provided by Proteus
        """
        M = np.zeros(3)
        for flag in self.boundaryFlags:
            if self.useIBM:
                M += self.ProtChSystem.model.levelModelList[-1].coefficients.particle_netMoments[flag]
            else:
                M += self.ProtChSystem.model.levelModelList[-1].coefficients.netMoments[flag]
        # !!!!!!!!!!!! UPDATE BARYCENTER !!!!!!!!!!!!
        Fx, Fy, Fz = self.F_prot
        rx, ry, rz = self.barycenter0-pyvec2array(self.ChBody.GetPos())
        Mp = np.array([ry*Fz-rz*Fy, -(rx*Fz-rz*Fx), (rx*Fy-ry*Fx)])
        M += Mp
        return M

    def getRotationMatrix(self):
        """Gives current rotation (matrix) of body

        Returns
        -------
        rot: array_like
            current rotation (matrix) of body
        """
        x0 = self.thisptr.rotm.GetAxisX().x()
        x1 = self.thisptr.rotm.GetAxisX().y()
        x2 = self.thisptr.rotm.GetAxisX().z()
        y0 = self.thisptr.rotm.GetAxisY().x()
        y1 = self.thisptr.rotm.GetAxisY().y()
        y2 = self.thisptr.rotm.GetAxisY().z()
        z0 = self.thisptr.rotm.GetAxisZ().x()
        z1 = self.thisptr.rotm.GetAxisZ().y()
        z2 = self.thisptr.rotm.GetAxisZ().z()
        rot = np.array([x0, x1, x2],
                       [y0, y1, y2],
                       [z0, z1, z2])
        return rot

    def prestep(self):
        """Called before Chrono system step.
        Sets external forces automatically from Proteus solution.
        """
        self.storeValues()
        # if self.ProtChSystem.thisptr.system.GetChTime() > 0.0003 or self.ProtChSystem.model is None:  # CHANGE
        #     # if self.ProtChSystem.step_nb > self.ProtChSystem.step_start:
        #     self.ChBody.SetBodyFixed(False)
        if self.ProtChSystem.model is not None:
            if self.ProtChSystem.model_addedmass is not None:
                # getting added mass matrix
                self.Aij[:] = 0
                am = self.ProtChSystem.model_addedmass.levelModelList[-1]
                self.Aij_updated_global = am.coefficients.updated_global
                for flag in self.boundaryFlags:
                    if self.useIBM:
                        self.Aij += am.coefficients.particle_Aij[flag]
                    else:
                        self.Aij += am.Aij[flag]
                if self.width_2D:
                    self.Aij *= self.width_2D
            # setting added mass
            if self.applyAddedMass is True:
                Aij = np.zeros((6,6))
                Aij[:] = self.Aij[:]
                Aij *= self.Aij_factor
                self.setAddedMass(Aij)
            self.setExternalForces()

    def setExternalForces(self, np.ndarray forces=None, np.ndarray moments=None):
        """Sets external forces to body.
        Called during prestep or can be called manually. If called manually,
        must be a Chrono only simulation.

        Parameters
        ----------
        forces: array_like
            forces array (length 3)
        moments: array_like
            moments array (length 3)
        """
        if forces is not None:
            self.F_prot = forces
        if moments is not None:
            self.M_prot = moments
        if self.ProtChSystem.model is not None:
            self.F_prot = self.getPressureForces()+self.getShearForces()
            self.M_prot = self.getMoments()
        if self.width_2D:
            self.F_prot *= self.width_2D
            self.M_prot *= self.width_2D
        cdef np.ndarray F_bar = np.zeros(3)
        cdef np.ndarray M_bar = np.zeros(3)
        F_bar[:] = self.F_prot[:]
        M_bar[:] = self.M_prot[:]
        if self.applyAddedMass is True:
            F_bar += self.F_Aij
            M_bar += self.M_Aij
        if self.ProtChSystem.first_step is False:
            # actual force applied to body
            if self.ProtChSystem.prediction == b"backwardEuler":
                F_bar = F_bar
                M_bar = M_bar
            if self.ProtChSystem.prediction == b"forwardEuler":
                F_bar = self.F_prot_last+self.Aij_last
                M_bar = self.M_prot_last+self.Aij_last
            if self.ProtChSystem.prediction == b"implicitOrder2":
                F_bar = (F_bar+self.F_prot_last+self.F_Aij_last)/2.
                M_bar = (M_bar+self.M_prot_last+self.M_Aij_last)/2.
                # self.F_applied = self.F_prot
                # self.M_applied = self.M_prot
                # self.F_applied = 2*F_bar - self.F_applied_last
                # self.M_applied = 2*M_bar - self.M_applied_last
        F_solid_type = 1
        if F_solid_type == 1:
            F_body = F_bar
            M_body = M_bar
        elif F_solid_type == 2:
            if np.linalg.norm(self.F_prot_last) == 0:  # first time step
                F_body = F_bar
                M_body = M_bar
            else:
                F_body = 2*F_bar-self.F_applied_last
                M_body = 2*M_bar-self.M_applied_last
        self.F_applied = F_body
        self.M_applied = M_body
        self.thisptr.prestep(<double*> self.F_applied.data,
                             <double*> self.M_applied.data)
        self.predicted = False

    def poststep(self):
        """Called after Chrono system step.
        Records values to csv, broadcast new position and rotation from
        calculating processor to all processors for moving mesh BC.
        """
        if self.prescribed_motion_function is not None:
            new_x = self.callPrescribedMotion(self.ProtChSystem.model.stepController.t_model_last)
            new_vec = chrono.ChVector3d(new_x[0], new_x[1], new_x[2])
            self.ChBody.SetPos(new_vec)
        self.thisptr.poststep()
        self.getValues()
        comm = Comm.get().comm.tompi4py()
        cdef ch.ChQuaternion rotq
        cdef ch.ChQuaternion rotq_last
        cdef ch.ChVector3d pos
        cdef ch.ChVector3d pos_last
        cdef double e0, e1, e2, e3, e0_last, e1_last, e2_last, e3_last
        cdef double posx, posy, posz, posx_last, posy_last, posz_last
        if comm.rank == 0 and self.ProtChSystem.record_values is True:
            self._recordValues()
            if self.thisptr.has_trimesh:
                self._recordH5()
                self._recordXML()
        # need to pass position and rotation values to C++ side
        # needed for transformations when calling hx, hy, hz, hxyz
        e0, e1, e2, e3 = self.rotq
        e0_last, e1_last, e2_last, e3_last = self.rotq_last
        posx, posy, posz = self.position
        posx_last, posy_last, posz_last = self.position_last
        pos = ch.ChVector3d(posx, posy, posz)
        pos_last = ch.ChVector3d(posx_last, posy_last, posz_last)
        rotq = ch.ChQuaternion[double](e0, e1, e2, e3)
        rotq_last = ch.ChQuaternion[double](e0_last, e1_last, e2_last, e3_last)
        self.thisptr.rotq = rotq
        self.thisptr.rotq_last = rotq_last
        self.thisptr.pos = pos
        self.thisptr.pos_last = pos_last
        if self.ProtChSystem.model_addedmass is not None:
            am = self.ProtChSystem.model_addedmass.levelModelList[-1]
            for flag in self.boundaryFlags:
                am.barycenters[flag] = pyvec2array(self.ChBody.GetPos())
        if self.useIBM and self.ProtChSystem.model is not None:
            self.updateIBM()

    def updateIBM(self):
        c = self.ProtChSystem.model.levelModelList[-1].coefficients
        if c.use_ball_as_particle:
            chpos = self.ChBody.GetPos()
            chvel = self.ChBody.GetPosDt()
            chvel_ang = self.ChBody.GetAngVelLocal()
            for flag in self.boundaryFlags:
                c.ball_radius[flag] = self.radiusIBM
                c.ball_center[flag, 0] = chpos.x
                c.ball_center[flag, 1] = chpos.y
                c.ball_center[flag, 2] = chpos.z
                c.ball_velocity[flag, 0] = chvel.x
                c.ball_velocity[flag, 1] = chvel.y
                c.ball_velocity[flag, 2] = chvel.z
                c.ball_angular_velocity[flag, 0] = chvel_ang.x
                c.ball_angular_velocity[flag, 1] = chvel_ang.y
                c.ball_angular_velocity[flag, 2] = chvel_ang.z
        else:
            for flag in self.boundaryFlags:
                c.particle_sdfList[flag] = self.getDynamicSDF
                c.particle_velocityList[flag] = lambda x, t: self.getVelocity()
            # This body has just moved, so the cached level set is stale.
            # attachModels latches phi_s_isSet so that an expensive *static* sdf is
            # evaluated once and never again; a Chrono-coupled body is dynamic by
            # construction, so clear the latch here and let the flow model's preStep
            # rebuild phi_s at the new position. Static sdf cases never reach this
            # method, so they keep the latch.
            c.phi_s_isSet = False

    def getDynamicSDF(self, t, x):
        chpos = self.ChBody.GetPos()
        relative_x = x-np.array([chpos.x, chpos.y, chpos.z])
        return self.sdfIBM(t, relative_x)

    def setPosition(self, np.ndarray pos):
        chvec = chrono.ChVector3d(pos[0], pos[1], pos[2])
        self.ChBody.SetPos(chvec)

    def getPosition(self):
        chpos = self.ChBody.GetPos()
        return pyvec2array(chpos)

    def setMass(self, double mass):
        self.ChBody.SetMass(mass)

    def getMass(self):
        return self.ChBody.GetMass()

    def setInertiaXX(self, np.ndarray inertia):
        chvec = chrono.ChVector3d(inertia[0], inertia[1], inertia[2])
        self.ChBody.SetInertiaXX(chvec)

    def setInertiaXY(self, np.ndarray inertia):
        chvec = chrono.ChVector3d(inertia[0], inertia[1], inertia[2])
        self.ChBody.SetInertiaXY(chvec)

    def getInertia(self):
        iner = pymat332array(self.ChBody.GetInertia())
        return iner

    def getVelocity(self):
        chvel = self.ChBody.GetPosDt()
        return pyvec2array(chvel)

    def setVelocity(self, np.ndarray vel):
        chvec = chrono.ChVector3d(vel[0], vel[1], vel[2])
        self.ChBody.SetPosDt(chvec)

    def prediction(self):
        comm = Comm.get().comm.tompi4py()
        cdef ch.ChVector3d h_body_vec
        h_body_vec = self.thisptr.hxyz(<double*> self.position_last.data, 0.)
        #print("MY BODY DISP: ", h_body_vec.x(), h_body_vec.y(), h_body_vec.z())
        if self.ProtChSystem.model is not None:
            try:
                dt = self.ProtChSystem.proteus_dt
                dt_next = self.ProtChSystem.proteus_dt_next
            except:
                dt = 0.
                dt_next = 0.
                print("$$$$$$$$$$$$$ ERROR")
        # else:
        #     dt = self.ProtChSystem.dt_fluid
        # dt_half = self.ProtChSystem
        # # if self.ProtChSystem.scheme == b"ISS":
        # self.h_predict_last = self.h_predict
        # self.h_ang_predict_last = self.h_ang_predict
        # self.h_ang_vel_predict_last = self.h_ang_vel_predict
        #     # dt_half = self.ProtChSystem.dt_fluid/2.
        #     # dt_half = self.ProtChSystem.dt_fluid_next/2.
        #     # if substeps is False:
        # self.h_predict = -self.velocity*dt_half
        # print("$$$$$$$$$$$$$$$", comm.rank, self.h_predict, dt, self.velocity[1])
        # self.h_ang_predict = np.sqrt(self.ang_velocity[0]**2
        #                              +self.ang_velocity[1]**2
        #                              +self.ang_velocity[2]**2)*dt_half
        # self.h_ang_vel_predict = self.ang_velocity
        #     # else:
        #     #     nsteps = max(int(dt_half/self.ProtChSystem.chronoDt), 1)
        #     #     if nsteps < self.ProtChSystem.min_nb_steps:
        #     #         nsteps = self.ProtChSystem.min_nb_steps
        #     #     h = np.zeros(3)
        #     #     h_ang = np.zeros(3)
        #     #     v = self.velocity.copy()
        #     #     v_ang = self.ang_velocity.copy()
        #     #     for i in range(nsteps):
        #     #         h, v = bd.forward_euler(h, v, self.acceleration, dt_half/nsteps)
        #     #         h_ang, v_ang = bd.forward_euler(h_ang, v_ang, self.ang_acceleration, dt_half/nsteps)
        #     #     self.h_predict = h
        #     #     self.h_ang_predict = np.sqrt(h_ang[0]**2+h_ang[1]**2+h_ang[2]**2)
        #     #     self.h_ang_vel_predict = v_ang
        #     # h_body_vec = self.thisptr.hxyz(<double*> self.position_last.data, 0)
        #     # h_body = pych.ChVector_to_npArray(h_body_vec)
        #     # if self.ProtChSystem.dt != 0:
        #     #     vel_predict = h_body/(self.ProtChSystem.dt)
        #     #     self.h_predict = vel_predict*dt_half
        #     # else:
        #     #     self.h_predict = np.zeros(3)
        #     # print("ADDED: ", self.h_predict, dt_half)
        #     # print("BODY H:", h_body, self.velocity, self.ChBody.GetPosDt())
        #     self.h_predict = h
        self.predicted = True

    def calculate_init(self):
        """Called from self.ProtChSystem.calculate_init()
        before simulation starts
        """
        # barycenter0 used for moment calculations
        if self.Shape is not None:
            self.barycenter0 = self.Shape.barycenter.copy()
        else:
            self.barycenter0 = pyvec2array(self.ChBody.GetPos())
        self.position_last[:] = pyvec2array(self.ChBody.GetPos())
        self.position[:] = pyvec2array(self.ChBody.GetPos())
        # check if IBM and set index if not set previously by user
        if self.useIBM:
            if self.boundaryFlags.size == 0:
                self.setBoundaryFlags([self.ProtChSystem.nBodiesIBM])
            self.ProtChSystem.nBodiesIBM += 1
        # get the initial values for F and M
        cdef np.ndarray zeros = np.zeros(3)
        self.setExternalForces(zeros, zeros)
        # build collision model
        #if deref(self.thisptr.body).IsCollisionEnabled():
        #    deref(deref(self.thisptr.body).GetCollisionModel()).BuildModel()
        # poststep (record values, etc)
        self.thisptr.poststep()
        # get first, store then on initial time step
        self.getValues()
        self.storeValues()
        # set mass matrix with no added mass
        self.setAddedMass(self.Aij)
        self.thisptr.calculate_init()

    def calculate(self):
        pass

    def setPrescribedMotionCustom(self, double[:] t, double[:] x=None,
                                  double[:] y=None, double[:] z=None,
                                  double[:] ang=None, double[:] ang2=None,
                                  double[:] ang3=None, double t_max=0):
        """Sets custom prescribed motion for body.
        Parameters must have the same length as the time array t

        Parameters
        ----------
        t: array_like
            time array
        x: array_like
            x coordinates of body
        y: array_like
            y coordinates of body
        z: array_like
            z coordinates of body
        ang: array_like
            rotation of body
        ang2: array_like
            rotation of body
        ang3: array_like
            rotation coordinates of body
        t_max: double
            prescribed motion is released when t > t_max.
            if t_max=0, the prescribed motion is never released.
        """
        cdef vector[double] t_vec
        cdef vector[double] x_vec
        cdef vector[double] y_vec
        cdef vector[double] z_vec
        cdef vector[double] ang_vec
        cdef vector[double] ang2_vec
        cdef vector[double] ang3_vec
        for tt in t:
            t_vec.push_back(tt)
        if x is not None:
            assert len(x) == len(t), 'x and t should have the same length'
            for xx in x:
                x_vec.push_back(xx)
        if y is not None:
            assert len(y) == len(t), 'y and t should have the same length'
            for yy in y:
                y_vec.push_back(yy)
        if z is not None:
            assert len(z) == len(t), 'z and t should have the same length'
            for zz in z:
                z_vec.push_back(zz)
        if ang is not None:
            assert len(ang) == len(t), 'ang and t should have the same length'
            for angang in ang:
                ang_vec.push_back(angang)
        if ang2 is not None:
            assert len(ang2) == len(t), 'ang2 and t should have the same length'
            for ang2ang2 in ang2:
                ang2_vec.push_back(ang2ang2)
        if ang3 is not None:
            assert len(ang3) == len(t), 'ang3 and t should have the same length'
            for ang3ang3 in ang3:
                ang3_vec.push_back(ang3ang3)
        self.thisptr.setPrescribedMotionCustom(t_vec, x_vec, y_vec, z_vec,
                                               ang_vec, ang2_vec, ang3_vec,
                                               t_max)

    def setPrescribedMotionSine(self, double a, double f):
        """Sets sinusoidal prescribed motion for body

        Parameters
        ----------
        a: double
            amplitude of sinusoid
        f: double
            frequency of sinusoid
        """
        self.thisptr.setPrescribedMotionSine(a, f)

    def setPrescribedMotionPoly(self, double coeff1):
        """Sets polynomial prescribed motion for body

        Parameters
        ----------
        coeff1: double
            coeff1 of polynomial
        """
        self.thisptr.setPrescribedMotionPoly(coeff1)

    def setPrescribedMotion(self, function):
        """Sets custom prescribed motion function
        (!) should be preferably set only if body is free and not
        linked to other bodies or other elements (such as moorings)
        as this is used only for setting moving mesh BC by enforcing
        position of the body at each time step.
        Use setPrescribedMotionPoly or setPrescribedMotionSine for
        functions that are safe to use with a body linked with other
        elements.

        Parameters
        ----------
        function:
            must be a function of time returning an array of the
            absolute position of the body (numpy array of length 3:
            x, y, z)
        """
        self.prescribed_motion_function = function

    cdef np.ndarray callPrescribedMotion(self, double t):
        return self.prescribed_motion_function(t)

    #: what a ProtChBody carries from one step to the next on the Python side (arrays and numbers); the Chrono side
    #: is the system's state()
    _STATE_ARRAYS = ("position", "position_last", "F", "M", "F_last", "M_last", "F_prot", "M_prot", "F_prot_last",
                     "M_prot_last", "F_applied", "M_applied", "F_applied_last", "M_applied_last", "F_Aij", "M_Aij",
                     "F_Aij_last", "M_Aij_last", "acceleration", "acceleration_last", "velocity", "velocity_last",
                     "ang_acceleration", "ang_acceleration_last", "ang_velocity", "ang_velocity_last", "rotq",
                     "rotq_last", "h_predict", "h_predict_last", "h_ang_vel_predict", "h_ang_vel_predict_last",
                     "adams_vel", "Aij")
    _STATE_NUMBERS = ("ang_vel_norm", "ang_vel_norm_last", "h_ang_predict", "h_ang_predict_last")

    def state(self):
        """The Python-side values a step carries (copies); see ProtChSystem.state()."""
        st = {}
        for name in self._STATE_ARRAYS:
            v = getattr(self, name, None)
            if v is not None:
                st[name] = np.array(v, dtype=float)
        for name in self._STATE_NUMBERS:
            st[name] = float(getattr(self, name))
        st["predicted"] = float(self.predicted)
        return st

    def set_state(self, state):
        for name in self._STATE_ARRAYS:
            if name in state:
                setattr(self, name, np.array(state[name], dtype=float))
        for name in self._STATE_NUMBERS:
            if name in state:
                setattr(self, name, float(state[name]))
        self.predicted = state.get("predicted", 0.) != 0.
        self._passToCpp()

    def _passToCpp(self):
        """Position and rotation (this step and last) on the C++ side, as poststep() leaves them: hxyz reads them."""
        cdef ch.ChQuaternion rotq
        cdef ch.ChQuaternion rotq_last
        cdef double e0, e1, e2, e3, e0_last, e1_last, e2_last, e3_last
        cdef double posx, posy, posz, posx_last, posy_last, posz_last
        e0, e1, e2, e3 = self.rotq
        e0_last, e1_last, e2_last, e3_last = self.rotq_last
        posx, posy, posz = self.position
        posx_last, posy_last, posz_last = self.position_last
        self.thisptr.pos = ch.ChVector3d(posx, posy, posz)
        self.thisptr.pos_last = ch.ChVector3d(posx_last, posy_last, posz_last)
        rotq = ch.ChQuaternion[double](e0, e1, e2, e3)
        rotq_last = ch.ChQuaternion[double](e0_last, e1_last, e2_last, e3_last)
        self.thisptr.rotq = rotq
        self.thisptr.rotq_last = rotq_last

    def storeValues(self):
        self.velocity_last = self.velocity
        self.position_last = self.position
        self.acceleration_last = self.acceleration
        self.rotq_last = self.rotq
        # self.rotm_last = self.rotm
        self.ang_acceleration_last = self.ang_acceleration
        self.ang_velocity_last = self.ang_velocity
        self.F_last = self.F
        self.M_last = self.M
        self.ang_vel_norm_last = self.ang_vel_norm
        # force from fluid at current time step
        self.F_prot_last = np.array(self.F_prot)
        self.M_prot_last = np.array(self.M_prot)
        self.F_applied_last = np.array(self.F_applied)
        self.M_applied_last = np.array(self.M_applied)
        self.F_Aij_last = np.array(self.F_Aij)
        self.M_Aij_last = np.array(self.M_Aij)

    def getValues(self):
        """Get values (pos, vel, acc, etc.) from C++ to python
        """
        # position
        self.position = pyvec2array(self.ChBody.GetPos())
        # rotation
        self.rotq = pyquat2array(self.ChBody.GetRot())
        # self.rotm = mat332array(self.ChBody.GetA())
        # acceleration
        self.acceleration = pyvec2array(self.ChBody.GetPosDt2())
        #velocity
        self.velocity = pyvec2array(self.ChBody.GetPosDt())
        # angular acceleration
        self.ang_acceleration = pyvec2array(self.ChBody.GetAngAccLocal())
        # angular velocity
        self.ang_velocity = pyvec2array(self.ChBody.GetAngVelLocal())
        # norm of angular velocity
        self.ang_vel_norm = np.sqrt(self.ang_velocity[0]**2
                                    +self.ang_velocity[1]**2
                                    +self.ang_velocity[2]**2)
        # force
        self.F = np.array([self.thisptr.F.x(),
                           self.thisptr.F.y(),
                           self.thisptr.F.z()])
        # moment
        self.M = np.array([self.thisptr.M.x(),
                           self.thisptr.M.y(),
                           self.thisptr.M.z()])


    def setRecordValues(self, all_values=False, pos=False,
                        rot=False, ang_disp=False, F=False, M=False,
                        inertia=False, vel=False, acc=False, ang_vel=False,
                        ang_acc=False, h_predict=False):
        """
        Sets the body attributes that are to be recorded in a csv file
        during the simulation.

        Parameters
        ----------
        all_values: bool
            Set to True to record all values listed below.
        time: bool
            Time of recorded row (default: True).
        pos: bool
            Position of body (default: False. Set to True to record).
        rot: bool
            Rotation of body (default: False. Set to True to record).
        ang_disp: array
            Angular displecement calculated during rigid body calculation step.
            Applied on the body in order to make it rotating.
        F: bool
            Forces applied on body (default: False. Set to True to record).
        M: bool
            Moments applied on body (default: False. Set to True to record).
        inertia: bool
            Inertia of body (default: False. Set to True to record).
        vel: bool
            Velocity of body (default: False. Set to True to record).
        acc: bool
            Acceleration of body (default: False. Set to True to record).
        ang_vel: array
            Angular velocity of body (default: False. Set to True to record).
        ang_acc: bool
            Angular acceleration of body (default: False. Set to True to record).

        Notes
        -----
        To add another value manually, add to dictionary self.record_dict::

            key: header of the column in .csv
            value: list of length 2: [variable name, index within variable]
                   (if no index, use None)
            e.g. self.record_dict['m']['mass', None]
        """
        if all_values is True:
            pos = rot = F = M = acc = vel = ang_acc = ang_vel = h_predict = True
        if pos is True:
            self.record_dict['x'] = ['position', 0]
            self.record_dict['y'] = ['position', 1]
            self.record_dict['z'] = ['position', 2]
        if rot is True:
            self.record_dict['rotq_e0'] = ['rotq', 0]
            self.record_dict['rotq_e1'] = ['rotq', 1]
            self.record_dict['rotq_e2'] = ['rotq', 2]
            self.record_dict['rotq_e3'] = ['rotq', 3]
        if F is True:
            self.record_dict['Fx'] = ['F', 0]
            self.record_dict['Fy'] = ['F', 1]
            self.record_dict['Fz'] = ['F', 2]
            self.record_dict['Fx_prot'] = ['F_prot', 0]
            self.record_dict['Fy_prot'] = ['F_prot', 1]
            self.record_dict['Fz_prot'] = ['F_prot', 2]
            self.record_dict['Fx_applied'] = ['F_applied', 0]
            self.record_dict['Fy_applied'] = ['F_applied', 1]
            self.record_dict['Fz_applied'] = ['F_applied', 2]
            self.record_dict['Fx_Aij'] = ['F_Aij', 0]
            self.record_dict['Fy_Aij'] = ['F_Aij', 1]
            self.record_dict['Fz_Aij'] = ['F_Aij', 2]
            Fx = Fy = Fz = True
        if M is True:
            self.record_dict['Mx'] = ['M', 0]
            self.record_dict['My'] = ['M', 1]
            self.record_dict['Mz'] = ['M', 2]
            self.record_dict['Mx_prot'] = ['M_prot', 0]
            self.record_dict['My_prot'] = ['M_prot', 1]
            self.record_dict['Mz_prot'] = ['M_prot', 2]
            self.record_dict['Mx_applied'] = ['M_applied', 0]
            self.record_dict['My_applied'] = ['M_applied', 1]
            self.record_dict['Mz_applied'] = ['M_applied', 2]
        if acc is True:
            self.record_dict['ax'] = ['acceleration', 0]
            self.record_dict['ay'] = ['acceleration', 1]
            self.record_dict['az'] = ['acceleration', 2]
        if vel is True:
            self.record_dict['ux'] = ['velocity', 0]
            self.record_dict['uy'] = ['velocity', 1]
            self.record_dict['uz'] = ['velocity', 2]
        if ang_acc is True:
            self.record_dict['ang_ax'] = ['ang_acceleration', 0]
            self.record_dict['ang_ay'] = ['ang_acceleration', 1]
            self.record_dict['ang_az'] = ['ang_acceleration', 2]
        if ang_vel is True:
            self.record_dict['ang_ux'] = ['ang_velocity', 0]
            self.record_dict['ang_uy'] = ['ang_velocity', 1]
            self.record_dict['ang_uz'] = ['ang_velocity', 2]
        if inertia is True:
            self.record_dict['intertia'] = ['inertia', None]
        if h_predict is True:
            self.record_dict['hx'] = ['h_predict', 0]
            self.record_dict['hy'] = ['h_predict', 1]
            self.record_dict['hz'] = ['h_predict', 2]

    def _recordValues(self):
        """Records values of body attributes in a csv file.
        """
        Profiling.logEvent('recording values file of '+str(self.name))
        record_file = str(os.path.join(bytes(Profiling.logDir,'utf-8'), self.name),'utf-8')
        t_chrono = self.ProtChSystem.ChSystem.GetChTime()
        if self.ProtChSystem.model is not None:
            t_last = self.ProtChSystem.model.stepController.t_model_last
            try:
                dt_last = self.ProtChSystem.model.levelModelList[-1].dt_last
            except:
                dt_last = 0
            t = t_last
        else:
            t = t_chrono
        t_sim = Profiling.time()-Profiling.startTime
        values_towrite = [t, t_chrono, t_sim]
        if t == 0:
            headers = ['t', 't_ch', 't_sim']
            for key in self.record_dict:
                headers += [key]
            with open(record_file+'.csv', 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow(headers)
        for key, val in self.record_dict.items():
            if val[1] is not None:
                values_towrite += [getattr(self, val[0])[val[1]]]
            else:
                values_towrite += [getattr(self, val[0])]
        with open(record_file+'.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(values_towrite)
        ## added mass
        if self.ProtChSystem.model_addedmass is not None:
            if t == 0:
                headers = ['t', 't_ch', 't_sim']
                for i in range(6):
                    for j in range(6):
                        headers += ['A'+str(i)+str(j)]
                with open(record_file+'_Aij.csv', 'w') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',')
                    writer.writerow(headers)
            values_towrite = [t, t_chrono, t_sim]
            for i in range(6):
                for j in range(6):
                    values_towrite += [self.Aij[i, j]]
            with open(record_file+'_Aij.csv', 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow(values_towrite)
        Profiling.logEvent('finished recording values file of '+str(self.name))

    def _recordH5(self):
        Profiling.logEvent('recording h5 file of '+str(self.name))
        tCount = self.ProtChSystem.tCount
        self.hdfFileName = self.name
        hdfFileName = os.path.join(bytes(Profiling.logDir,'utf-8'), self.hdfFileName)+b'.h5'
        if tCount == 0:
            f = h5py.File(hdfFileName, 'w', libver='latest')
        else:
            f = h5py.File(hdfFileName, 'a', libver='latest')
        poss, element_connection = self.getTriangleMeshInfo()
        pos = np.zeros_like(poss)
        self.thisptr.updateTriangleMeshVisualisationPos()
        for i in range(self.thisptr.trimesh_pos.size()):
            pos[i, 0] = self.thisptr.trimesh_pos[i].x()
            pos[i, 1] = self.thisptr.trimesh_pos[i].y()
            pos[i, 2] = self.thisptr.trimesh_pos[i].z()
        dset = f.create_dataset('nodes_t'+str(tCount), pos.shape)
        dset[...] = pos
        dset = f.create_dataset('elements_t'+str(tCount), element_connection.shape, dtype='i8')
        dset[...] = element_connection
        Profiling.logEvent('finished recording h5 file of '+str(self.name))

    def _recordXML(self):
        Profiling.logEvent('recording xml file of '+str(self.name))
        tCount = self.ProtChSystem.tCount
        t = self.ProtChSystem.ChSystem.GetChTime()
        xmlFile = os.path.join(bytes(Profiling.logDir,'utf-8'), self.name)+'.xmf'
        # if tCount == 0:
        root = ET.Element("Xdmf",
                        {"Version": "2.0",
                        "xmlns:xi": "http://www.w3.org/2001/XInclude"})
        domain = ET.SubElement(root, "Domain")
        arGridCollection = ET.SubElement(domain,
                                        "Grid",
                                        {"Name": "Mesh"+" Spatial_Domain",
                                        "GridType": "Collection",
                                        "CollectionType": "Temporal"})
        # else:
        #     tree = ET.parse(xmlFile)
        #     root = tree.getroot()
        #     domain = root[0]
        #     arGridCollection = domain[0]
        Xdmf_ElementTopology = "Triangle"
        pos, el = self.getTriangleMeshInfo()
        Xdmf_NumberOfElements= len(el)
        Xdmf_NodesPerElement = 3
        dataItemFormat = "HDF"

        arGrid = ET.SubElement(arGridCollection,
                               "Grid",
                               {"GridType": "Uniform"})
        arTime = ET.SubElement(arGrid,
                               "Time",
                               {"Value": str(t),
                                "Name": str(tCount)})
        topology = ET.SubElement(arGrid,
                                 "Topology",
                                 {"Type": Xdmf_ElementTopology,
                                  "NumberOfElements": str(Xdmf_NumberOfElements)})

        elements = ET.SubElement(topology,
                                 "DataItem",
                                 {"Format": dataItemFormat,
                                  "DataType": "UInt",
                                  "Dimensions": "%i %i" % (Xdmf_NumberOfElements,
                                                           Xdmf_NodesPerElement)})
        elements.text = self.hdfFileName+".h5:/elementsSpatial_Domain"+str(tCount)
        geometry = ET.SubElement(arGrid,"Geometry",{"Type":"XYZ"})
        nodes = ET.SubElement(geometry,
                              "DataItem",
                              {"Format": dataItemFormat,
                               "DataType": "Float",
                               "Precision": "8",
                               "Dimensions": "%i %i" % (pos.shape[0],
                                                        pos.shape[1])})
        nodes.text = self.hdfFileName+".h5:/nodesSpatial_Domain"+str(tCount)

        tree = ET.ElementTree(root)

        with open(xmlFile, "wb") as f:
            xmlHeader = "<?xml version=\"1.0\" ?>\n<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n"
            f.write(bytes(xmlHeader,"utf-8"))
            indentXML(tree.getroot())
            tree.write(f)

        # dump xml str in h5 file
        hdfFileName = os.path.join(bytes(Profiling.logDir,'utf-8'), self.hdfFileName)+b'.h5'
        f = h5py.File(hdfFileName, 'a', libver='latest')
        datav = ET.tostring(arGrid)
        dset = f.create_dataset('Mesh_Spatial_Domain_'+str(tCount),
                                (1,),
                                dtype=h5py.special_dtype(vlen=str))
        dset[0] = datav
        # close file
        f.close()
        Profiling.logEvent('finished recording xml file of '+str(self.name))


    def addPrismaticLinksWithSpring(self, np.ndarray pris1,
                                    np.ndarray pris2, double stiffness, double damping,
                                    double rest_length):
        """
        fairlead: barycenter coords
        pris: absolute coords
        pris1-------fairlead(barycenter)
        |
        |
        |
        |
        pris2
        """
        self.thisptr.addPrismaticLinksWithSpring(<double*> pris1.data,
                                                 <double*> pris2.data,
                                                 stiffness,
                                                 damping,
                                                 rest_length)
    def addPrismaticLinkX(self, double[:] pris1):
        self.thisptr.addPrismaticLinkX(&pris1[0]);

    def setName(self, string name):
        """Sets name of body (used for csv file)

        Parameters
        ----------
        name: str
            name of the body
        """
        self.name = name
        self.thisptr.setName(self.name)


cdef class ProtChSystem:

    def __cinit__(self, int nd=3, dt_init=0., sampleRate=0):
        self.thisptr = newSystem()
        self.subcomponents = []
        # cannot call it self.ChSystem or self.ChSystemSMC (conflict in C++ - unknown reason)
        self.ChSystemSMC = chrono.ChSystemSMC()
        self.ChSystem = self.ChSystemSMC
        # self.ChSystemSMC.this.disown()
        cdef SwigPyObject *swig_obj = <SwigPyObject*>self.ChSystemSMC.this
        cdef shared_ptr[ch.ChSystemSMC]* pt_to_shp = <shared_ptr[ch.ChSystemSMC]*> swig_obj.ptr;
        self.thisptr.systemSMC = pt_to_shp[0]
        self.thisptr.system = <shared_ptr[ch.ChSystem]> pt_to_shp[0]
        self.dt_init = dt_init
        self.model = None
        self.nd = nd
        self.build_kdtree = True
        self.dist_search = True
        self.min_nb_steps = 1  # minimum number of chrono substeps
        self.proteus_dt = 1.
        self.first_step = True  # just to know if first step
        self.setCouplingScheme(b"CSS")
        self.dt_fluid = 1.
        self.dt_fluid_next = 1.
        self.proteus_dt_next = 1.
        self.dt = 0.
        self.step_nb = 0
        self.step_start = 0
        self.sampleRate = sampleRate
        self.next_sample = 0.
        self.record_values = True
        self.t = 0.
        self.chrono_dt = 1.
        self.ProtChAddedMass = ProtChAddedMass(self)
        self.tCount = 0
        self.initialized = False
        self.update_substeps = False

        # Set the chrono logging values
        self.log_chrono_format = b'h5'
        self.log_chrono_bodies = None
        self.log_chrono_springs = None
        self.log_chrono_residuals = None

    def getChronoObject(self):
        return self.ChSystem

    def setTimeStep(self, double dt):
        """Sets time step for Chrono solver.
        Calculations in Chrono will use this time step within the
        Proteus time step (if bigger)
        Parameters
        ----------
        dt: float
            Chrono time step size
        """
        self.chrono_dt = dt
        self.thisptr.chrono_dt = dt

    def setSampleRate(self, sampleRate):
        self.sampleRate = sampleRate

    def addProtChBody(self, ProtChBody body):
        # self.ChSystemSMC.Add(body.ChBody)
        self.ChSystem.Add(body.ChBody)
        body.ProtChSystem = self
        self.addSubcomponent(body)  # add body to system (for pre and post steps)

    def addProtChMesh(self, ProtChMesh mesh):
        self.thisptr.addMesh(mesh.mesh)
        mesh.ProtChSystem = self

    def setGravitationalAcceleration(self, g):
        chvec = chrono.ChVector3d(g[0], g[1], g[2])
        self.ChSystem.SetGravitationalAcceleration(chvec)

    def getGravitationalAcceleration(self):
        return pyvec2array(self.ChSystem.GetGravitationalAcceleration())

    def setCouplingScheme(self, string scheme, string prediction=b'backwardEuler'):
        assert scheme == b"CSS" or scheme == b"ISS", "Coupling scheme requested unknown"
        assert prediction == b"backwardEuler" or prediction == b"forwardEuler" or prediction == b"implicitOrder2", "Prediction requested unknown"
        self.scheme = scheme
        self.prediction = prediction

    def attachModel(self, model, ar):
        """Attaches Proteus model to auxiliary variable
        """
        c = model.levelModelList[-1].coefficients
        self.model = model
        return self

    def attachAuxiliaryVariables(self,avDict):
        pass

    def setMinimumSubsteps(self, int nb):
        """Sets the minimum nb of chrono substeps per proteus step
        if protDt=0.001 and chDt=0.002, there will be <nb>
        substeps of chrono instead of just 1.

        Parameters
        ----------
        nb: int
            Minimum number of chrono substeps.
        """
        self.min_nb_steps = nb

    def state(self):
        """What a step retry or a hot start needs to put the coupled system back where it was: the Chrono system's
        integrable state (``cppSystem.stateGather``: time, positions, velocities and accelerations of every body and
        node, Chrono's StateGather/StateGatherAcceleration), the step counters and step sizes of this class, and
        each body's ``state()``. Plain arrays and floats. Not included: reaction forces, collision contacts, and the
        bodies' trimesh and recorded-history files."""
        st = {"chrono": np.array(self.thisptr.stateGather(), dtype=float)}
        st.update({"step_nb": float(self.step_nb), "first_step": float(self.first_step), "t": self.t,
                   "tCount": float(self.tCount), "next_sample": self.next_sample,
                   "record_values": float(self.record_values), "proteus_dt": self.proteus_dt,
                   "proteus_dt_last": self.proteus_dt_last, "proteus_dt_next": self.proteus_dt_next,
                   "dt": self.dt, "dt_last": self.dt_last, "dt_fluid": self.dt_fluid,
                   "dt_fluid_last": self.dt_fluid_last, "dt_fluid_next": self.dt_fluid_next})
        for i, s in enumerate(self.subcomponents):
            if isinstance(s, ProtChBody):
                for key, v in s.state().items():
                    st["body%d.%s" % (i, key)] = v
        return st

    def set_state(self, state):
        """Restore what state() kept (the system must hold the same bodies and links)."""
        self.thisptr.stateScatter(np.asarray(state["chrono"], dtype=float))
        self.step_nb = int(state["step_nb"])
        self.first_step = state["first_step"] != 0.
        self.t = state["t"]
        self.tCount = int(state["tCount"])
        self.next_sample = state["next_sample"]
        self.record_values = state["record_values"] != 0.
        self.proteus_dt = state["proteus_dt"]
        self.proteus_dt_last = state["proteus_dt_last"]
        self.proteus_dt_next = state["proteus_dt_next"]
        self.dt = state["dt"]
        self.dt_last = state["dt_last"]
        self.dt_fluid = state["dt_fluid"]
        self.dt_fluid_last = state["dt_fluid_last"]
        self.dt_fluid_next = state["dt_fluid_next"]
        for i, s in enumerate(self.subcomponents):
            if isinstance(s, ProtChBody):
                prefix = "body%d." % i
                s.set_state({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)})

    def step(self, dt):
        self.dt_fluid_last = self.dt_fluid
        self.dt_fluid = dt
        self.dt_fluid_next = self.proteus_dt_next
        self.dt_last = self.dt
        if self.scheme == b"ISS":
            self.dt = (self.dt_fluid+self.dt_fluid_last)/2.
        elif self.scheme == b"CSS":
            self.dt = dt
        # calculate number of time steps
        nb_steps = max(int(dt/self.chrono_dt), 1)
        if nb_steps < self.min_nb_steps:
            nb_steps = self.min_nb_steps
        # solve Chrono system
        self.step_nb += 1
        # self.thisptr.system.setChTime()
        comm = Comm.get().comm.tompi4py()
        t = self.ChSystem.GetChTime()
        Profiling.logEvent('Solving Chrono system from t='
                        +str(t)
                        +' with dt='+str(self.dt)
                        +'('+str(nb_steps)+' substeps)')
        if dt > 0:
            if self.update_substeps is False:
                self.thisptr.step(<double> self.dt, nb_steps)
            else:
                dt_substep = self.dt/nb_steps
                for i in range(nb_steps):
                    self.thisptr.step(<double> dt_substep, 1)
                    # tri: hack to update forces on cables
                    for s in self.subcomponents:
                        if type(s) is ProtChMoorings:
                            # update forces keeping same fluid vel/acc
                            s.updateForces()
        t = self.ChSystem.GetChTime()
        Profiling.logEvent('Solved Chrono system to t='+str(t))
        if self.scheme == b"ISS":
            Profiling.logEvent('Chrono system to t='+str(t+self.dt_fluid_next/2.))

    def calculate(self, proteus_dt=None):
        """Does chrono system calculation for a Proteus time step
        Calls prestep and poststep on all subcomponents (bodies,
        moorings, etc) attached to the system.

        Parameters
        ----------
        proteus_dt: Optional[proteus_dt]
            Manually sets a time step. The time step is set
            automatically when coupled with a Proteus simulation
        """
        self.proteus_dt_last = self.proteus_dt
        if self.model is not None:
            try:
                self.proteus_dt = self.model.levelModelList[-1].dt_last
                self.proteus_dt_next = self.model.levelModelList[-1].dt_last  # wrong prediction is varying time step
                self.t = t = self.model.stepController.t_model_last
            except:
                self.proteus_dt = self.dt_init
                self.t = t = 0.
        elif proteus_dt is not None:
            self.proteus_dt = proteus_dt
            self.proteus_dt_next = proteus_dt  # BAD PREDICTION IF TIME STEP NOT REGULAR
            self.t = t = self.ChSystem.GetChTime()
        else:
            sys.exit('ProtChSystem: no time step set in calculate()')
        if self.model is not None:
            if self.build_kdtree is True:
                Profiling.logEvent("Building k-d tree for mooring nodes lookup")
                self.nodes_kdtree = spatial.cKDTree(self.model.levelModelList[-1].mesh.nodeArray)
        if t >= self.next_sample:
            self.record_values = True
            self.next_sample += self.sampleRate
        import time
        Profiling.logEvent("Chrono prestep")
        for s in self.subcomponents:
            s.prestep()
        self.step(self.proteus_dt)
        Profiling.logEvent("Chrono poststep")
        for s in self.subcomponents:
            s.poststep()

        # Log the chrono phyics of interest
        if self.log_chrono_bodies:
            if self.log_chrono_format == b'h5':
                self.log_bodies_h5(self.log_chrono_bodies)
            else:
                self.log_bodies_text(self.t + self.proteus_dt, self.log_chrono_bodies)

        if self.log_chrono_springs:
            if self.log_chrono_format == b'h5':
                self.log_springs_h5(self.log_chrono_springs)
            else:
                self.log_springs_text(self.t + self.proteus_dt, self.log_chrono_springs)

        if self.log_chrono_residuals:
            if self.log_chrono_format == b'h5':
                self.log_residuals_h5(self.log_chrono_residuals)
            else:
                self.log_residuals_text(self.t + self.proteus_dt, self.log_chrono_residuals)

        if (self.log_chrono_bodies or self.log_chrono_springs or self.log_chrono_residuals) \
                and self.log_chrono_format == b'h5':
            self.log_times_h5(self.t + self.proteus_dt)

        self.record_values = False
        self.first_step = False  # first step passed
        self.tCount += 1
        Profiling.logEvent("Chrono poststep finished")

    def calculate_init(self):
        """Does chrono system initialisation
        (!) Must be called before the first calculate() call.
        Calls calculate_init and poststep on all subcomponents
        (bodies, moorings, etc) attached to the system.
        """
        if self.model is not None:
            self.model_mesh = self.model.levelModelList[-1].mesh
            if self.build_kdtree is True:
                Profiling.logEvent("Building k-d tree for mooring nodes lookup on first time step")
                self.u = self.model.levelModelList[-1].u
                # finite element space (! linear for p, quadratic for velocity)
                self.femSpace_velocity = self.u[1].femSpace
                self.femSpace_pressure = self.u[0].femSpace
                self.nodes_kdtree = spatial.cKDTree(self.model.levelModelList[-1].mesh.nodeArray)
        if not self.initialized:
            self.nBodiesIBM = 0  # will be incremented by bodies calculate_init()
            for s in self.subcomponents:
                s.calculate_init()
            Profiling.logEvent("Setup initial"+str(self.next_sample))
            self.ChSystem.Setup()
            Profiling.logEvent("Finished init"+str(self.next_sample))
            self.initialized = True
        else:
            Profiling.logEvent("Warning: Chrono system was already initialized")
        Profiling.logEvent("Starting init"+str(self.next_sample))
        self.directory = bytes(Profiling.logDir+str("/"),'utf-8')
        self.thisptr.setDirectory(self.directory)
        if self.nBodiesIBM > 0:
            assert self.model is not None, 'IBM set to be used in bodies but model is not attached. Attach rans model BEFORE calling calculate_init() on chrono system'
            c = self.model.levelModelList[-1].coefficients
            # set it only if it was not set manually before
            if not c.nParticles:
                c.nParticles = self.nBodiesIBM
            else:
                assert c.nParticles == self.nBodiesIBM, 'number of RANS particles {nP} != number of IBM bodies {nB}'.format(nP=c.nParticles, nB=self.nBodiesIBM)
            if c.use_ball_as_particle:
                if c.ball_radius is None:
                    c.ball_radius = np.zeros(self.nBodiesIBM, 'd')
                if c.ball_center is None:
                    c.ball_center = np.zeros((self.nBodiesIBM, 3), 'd')
                if c.ball_velocity is None:
                    c.ball_velocity = np.zeros((self.nBodiesIBM, 3), 'd')
                if c.ball_angular_velocity is None:
                    c.ball_angular_velocity = np.zeros((self.nBodiesIBM, 3), 'd')
            else:
                if c.particle_sdfList is None:
                    c.particle_sdfList = [None for i in range(self.nBodiesIBM)]
                if c.particle_velocityList is None:
                    c.particle_velocityList = [None for i in range(self.nBodiesIBM)]
        for s in self.subcomponents:
            s.poststep()

    def setTimestepperType(self, string tstype, bool verbose=False):
        """Change timestepper (default: Euler)

        Parameters
        ----------
        tstype: str
            type of timestepper ('Euler' or 'HHT')
        """
        tstypes = ["Euler", "HHT", "Trapezoidal"]
        assert str(tstype) in tstypes, str(tstype)+" not a valid choice."
        self.thisptr.setTimestepperType(tstype, verbose)

    def addSubcomponent(self, subcomponent):
        """Adds subcomponent to system
        calculate_init() of subcomponent called before initial timestep
        prestep() and poststep of subcomponent() called at all timestep

        Parameters
        ----------
        subcomponent: class instance
            class instance of subcomponent
        """
        self.subcomponents += [subcomponent]

    def findElementContainingCoordsKD(self, coords):
        """
        k-d tree search of nearest node, element containing coords, and owning
        rank.

        Parameters
        ----------
        coords: array_like
            global coordinates to look for

        Returns
        -------
        xi:
            local coordinates
        node: int
            nearest node
        eN: int
            (local) element number
        rank: int
            processor rank containing element
        """
        comm = Comm.get().comm.tompi4py()
        owning_proc = 0
        xi = element = None  # initialised as None
        local_element = None
        # get nearest node on each processor
        comm.barrier()
        nearest_node, nearest_node_distance = getLocalNearestNode(coords, self.nodes_kdtree)
        # look for element containing coords on each processor (if it exists)
        comm.barrier()
        # make sure that processor owns nearest node
        if nearest_node < self.model_mesh.nNodes_owned:
            local_element = getLocalElement(self.femSpace_velocity, coords, nearest_node)

        else:
            nearest_node_distance = 0
        _, owning_proc = comm.allreduce((-nearest_node_distance, comm.rank),
                                        op=MPI.MINLOC)
        comm.barrier()
        if comm.rank == owning_proc and local_element:
            xi = self.femSpace_velocity.elementMaps.getInverseValue(local_element,
                                                                    coords)
        comm.barrier()
        xi = comm.bcast(xi, owning_proc)
        eN = comm.bcast(local_element, owning_proc)
        rank = comm.bcast(owning_proc, owning_proc)
        node = comm.bcast(nearest_node, owning_proc)
        return xi, node, eN, rank


    def findElementContainingCoordsDist(self,
                                        coords,
                                        node_guess,
                                        eN_guess,
                                        rank_guess):
        """
        Distance search of nearest node, element containing coords, and owning
        rank.

        Parameters
        ----------
        coords: array_like
            global coordinates to look for
        node_guess: int
            first guess of closest node
        eN_guess: int
            first guess of element containing coords
        rank_guess: int
            first guess of rank containing coords

        Returns
        -------
        xi:
            local coordinates
        node: int
            nearest node
        eN: int
            (local) element number
        rank: int
            processor rank containing element
        """
        mm = self.model_mesh  # local mesh
        mg = self.model_mesh.globalMesh  # global mesh
        comm = Comm.get().comm.tompi4py()
        xi = None  # initialised as None
        # get nearest node on each processor
        local_element = None
        # first check if element still containing coords
        rank_owning = rank_guess
        nearest_node = node_guess
        if comm.rank == rank_owning:
            if 0 <= eN_guess < mm.nElements_global:
                xi = self.femSpace_velocity.elementMaps.getInverseValue(eN_guess, coords)
                if self.femSpace_velocity.elementMaps.referenceElement.onElement(xi):
                    local_element = eN_guess
                else:
                    xi = None
        local_element = comm.bcast(local_element, rank_owning)
        # if not, find new nearest node, element, and owning processor
        coords_outside = False
        while local_element is None and coords_outside is False:
            rank_owning_previous = rank_owning
            owning_rank = False
            if comm.rank == rank_owning:
                owning_rank = True
                nearest_node, dist = pyxGetLocalNearestNode(coords=coords,
                                                      nodeArray=mm.nodeArray,
                                                      nodeStarOffsets=mm.nodeStarOffsets,
                                                      nodeStarArray=mm.nodeStarArray,
                                                      node=nearest_node,
                                                      rank=rank_owning)
                if nearest_node >= mm.nNodes_owned:
                    # change rank ownership
                    node_nb_global = mg.nodeNumbering_subdomain2global[nearest_node]
                    if not mg.nodeOffsets_subdomain_owned[comm.rank] <= node_nb_global < mg.nodeOffsets_subdomain_owned[comm.rank+1]:
                        new_rank = None
                        for i in range(len(mg.nodeOffsets_subdomain_owned)):
                            if mg.nodeOffsets_subdomain_owned[i] > node_nb_global:
                                # changing processor
                                if new_rank is None:
                                    new_rank = i-1
                            # getting nearest node number on new rank
                        if new_rank is not None:
                            nearest_node = node_nb_global-mg.nodeOffsets_subdomain_owned[new_rank]
                            rank_owning = new_rank
                else:
                    # find local element
                    local_element = getLocalElement(self.femSpace_velocity,
                                                    coords,
                                                    nearest_node)
                    if local_element is not None:
                        xi = self.femSpace_velocity.elementMaps.getInverseValue(local_element, coords)
            # ownership might have changed here
            comm.barrier()
            _, rank_owning = comm.allreduce((owning_rank, rank_owning),
                                            op=MPI.MAXLOC)
            _, nearest_node = comm.allreduce((owning_rank, nearest_node),
                                             op=MPI.MAXLOC)
            comm.barrier()
            # if ownership is the same after 1 loop and local_element not found
            # => coords must be outside domain
            if rank_owning == rank_owning_previous and local_element is None:
                coords_outside = True
                break
        comm.barrier()
        xi = comm.bcast(xi, rank_owning)
        eN = comm.bcast(local_element, rank_owning)
        rank = rank_owning
        node = comm.bcast(nearest_node, rank_owning)
        return xi, node, eN, rank

    def getFluidVelocityLocalCoords(self, xi, element, rank):
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
        #log Profiling.logEvent("FINDING VELOCITY AT RANK "+str(rank)+", "+str(element) + ", " + str(xi))
        comm = Comm.get().comm.tompi4py()
        if comm.rank == rank:
            u = self.u[1].getValue(element, xi)
            v = self.u[2].getValue(element, xi)
            if self.nd > 2:
                w = self.u[3].getValue(element, xi)
            if self.nd <= 2:
                w = 0
            # broadcast to all processors
        else:
            u = v = w = None
        u = comm.bcast(u, rank)
        v = comm.bcast(v, rank)
        w = comm.bcast(w, rank)
        #log Profiling.logEvent("FOUND VELOCITY")
        return u, v, w

    def getFluidVelocityGradientLocalCoords(self, xi, element, rank):
        comm = Comm.get().comm.tompi4py()
        if comm.rank == rank:
            u_grad = self.u_grad[1].getGradientValue(element, xi)
            u_grad = comm.bcast(u_grad, rank)
            v_grad = self.u[2].getGradientValue(element, xi)
            v_grad = comm.bcast(v_grad, rank)
            if self.nd > 2:
                w_grad = self.u[3].getGradientValue(element, xi)
            if self.nd <= 2:
                w_grad = 0
            # broadcast to all processors
            w_grad = comm.bcast(w_grad, rank)
        return u_grad, v_grad, w_grad

    def setCollisionEnvelopeMargin(self, double envelope, double margin):
        self.thisptr.setCollisionEnvelopeMargin(envelope, margin)

    def log_bodies_text(self, d_time, l_logging_info):
        """
        Logs the chrono information into a text file at each timestep

        Parameters
        ----------
        self: object
            ProtChSystem being referenced
        d_time: float
            Simulation time
        l_logging_info: list
            Contains the information to be logged. Structure is [body number, type]
            with multiple entries being included as a 2d list. Valid types are
            'position', 'rotation', 'force', and 'torque'.

        Returns
        -------
        None. Data is logged to a text file
        """

        # Open the file
        s_filename = 'chrono_log_body.txt'
        o_file = open(s_filename, 'a+')

        # Loop and write the body information
        for i_entry_body in range(0, len(l_logging_info), 1):
            # Extract the coordinate positions
            o_body = self.subcomponents[l_logging_info[i_entry_body][0]].ChBody

            if l_logging_info[i_entry_body][1] == 'position':
                o_body_position = o_body.GetPos()

                o_file.write(str(d_time) + '\t' + 'position' + '\t' +
                             str(l_logging_info[i_entry_body][0]) + '\t' + str(o_body_position.x) + '\t' +
                             str(o_body_position.y) + '\t' + str(o_body_position.z) + '\n')

            elif l_logging_info[i_entry_body][1] == 'rotation':
                o_body_rotation = o_body.GetRot()

                o_file.write(str(d_time) + '\t' + 'rotation' + '\t' +
                             str(l_logging_info[i_entry_body][0]) + '\t' + str(o_body_rotation.e0) + '\t' +
                             str(o_body_rotation.e1) + '\t' + str(o_body_rotation.e2) + '\t' + str(o_body_rotation.e3) +
                             '\n')

            elif l_logging_info[i_entry_body][1] == 'force':
                o_body_force = o_body.Get_XForce()

                o_file.write(str(d_time) + '\t' + 'force' + '\t' +
                             str(l_logging_info[i_entry_body][0]) + '\t' + str(o_body_force.x) + '\t' +
                             str(o_body_force.y) + '\t' + str(o_body_force.z) + '\n')

            elif l_logging_info[i_entry_body][1] == 'torque':
                o_body_torque = o_body.Get_Xtorque()

                o_file.write(str(d_time) + '\t' + 'torque' + '\t' +
                             str(l_logging_info[i_entry_body][0]) + '\t' + str(o_body_torque.x) + '\t' +
                             str(o_body_torque.y) + '\t' + str(o_body_torque.z) + '\n')

            else:
                raise NotImplementedError('Chrono body log not understood.')

        # Close the log file
        o_file.close()


    def log_bodies_h5(self, l_logging_info):
        """
        Logs the chrono information into a h5 file at each timestep

        Parameters
        ----------
        self: object
            ProtChSystem being referenced
        l_logging_info: list
            Contains the information to be logged. Structure is [body number, type]
            with multiple entries being included as a 2d list. Valid types are
            'position', 'rotation', 'force', and 'torque'.

        Returns
        -------
        None. Data is saved to an h5 file.
        """

        # Open the log file
        o_file = h5py.File('chrono_log.h5')

        # Create the empty data holders
        l_position = []
        l_rotation = []
        l_force = []
        l_torque = []

        # Loop and write the body information
        for i_entry_body in range(0, len(l_logging_info), 1):
            # Extract the coordinate positions
            o_body = self.subcomponents[l_logging_info[i_entry_body][0]].ChBody

            if l_logging_info[i_entry_body][1] == 'position':
                o_body_position = o_body.GetPos()
                l_position.append([o_body_position.x, o_body_position.y, o_body_position.z])

            elif l_logging_info[i_entry_body][1] == 'rotation':
                o_body_rotation = o_body.GetRot()
                l_rotation.append([o_body_rotation.e0, o_body_rotation.e1, o_body_rotation.e2, o_body_rotation.e3])

            elif l_logging_info[i_entry_body][1] == 'force':
                o_body_force = o_body.Get_XForce()
                l_force.append([o_body_force.x, o_body_force.y, o_body_force.z])

            elif l_logging_info[i_entry_body][1] == 'torque':
                o_body_torque = o_body.Get_Xtorque()
                l_torque.append([o_body_torque.x, o_body_torque.y, o_body_torque.z])

            else:
                raise NotImplementedError('Chrono body log not understood.')

        # Log the position data
        if len(l_position) > 0:
            # Check if the dataset exists
            try:
                # Open the dataset
                dm_position = o_file['position']

                # Resize the dataset
                i_index = dm_position.shape[0]
                dm_position.resize(dm_position.shape[0] + 1, axis=0)

                # Store the values into the dataset
                dm_position[i_index, :, :] = np.array(l_position)

            except:
                # Create the initial dataset
                dm_position = o_file.create_dataset('position', (1, len(l_position), 3), compression="gzip",
                                                    maxshape=(None, len(l_position), 3), dtype=float)

                # Store the first entry into the dataset
                dm_position[0, :, :] = np.array(l_position)

        # Log the rotation data
        if len(l_rotation) > 0:
            # Check if the dataset exists
            try:
                # Open the dataset
                dm_rotation = o_file['rotation']

                # Resize the dataset
                i_index = dm_rotation.shape[0]
                dm_rotation.resize(dm_rotation.shape[0] + 1, axis=0)

                # Store the values into the dataset
                dm_rotation[i_index, :, :] = np.array(l_rotation)

            except:
                # Create the initial dataset
                dm_rotation = o_file.create_dataset('rotation', (1, len(l_rotation), 4), compression="gzip",
                                                    maxshape=(None, len(l_rotation), 4), dtype=float)

                # Store the first entry into the dataset
                dm_rotation[0, :, :] = np.array(l_rotation)

        # Log the force data
        if len(l_force) > 0:
            # Check if the dataset exists
            try:
                # Open the dataset
                dm_force = o_file['force']

                # Resize the dataset
                i_index = dm_force.shape[0]
                dm_force.resize(dm_force.shape[0] + 1, axis=0)

                # Store the values into the dataset
                dm_force[i_index, :, :] = np.array(l_force)

            except:
                # Create the initial dataset
                dm_force = o_file.create_dataset('force', (1, len(l_force), 3), compression="gzip",
                                                 maxshape=(None, len(l_force), 3), dtype=float)

                # Store the first entry into the dataset
                dm_force[0, :, :] = np.array(l_force)

        # Log the torque data
        if len(l_torque) > 0:
            # Check if the dataset exists
            try:
                # Open the dataset
                dm_torque = o_file['torque']

                # Resize the dataset
                i_index = dm_torque.shape[0]
                dm_torque.resize(dm_torque.shape[0] + 1, axis=0)

                # Store the values into the dataset
                dm_torque[i_index, :, :] = np.array(l_torque)

            except:
                # Create the initial dataset
                dm_torque = o_file.create_dataset('torque', (1, len(l_torque), 3), compression="gzip",
                                                  maxshape=(None, len(l_torque), 3), dtype=float)

                # Store the first entry into the dataset
                dm_torque[0, :, :] = np.array(l_torque)

        # Close the log file
        o_file.close()


    def log_springs_text(self, d_time, l_springs):
        """
        Logs the chrono information into a text file at each timestep
        Parameters
        ----------
        self: object
            ProtChSystem being referenced
        d_time: float
            Current simulation time
        l_springs: list
            Spring objects for which data is being stored
        Returns
        -------
        None. Data is saved to a text file.
        """

        # Open the file
        s_filename = 'chrono_log_spring.txt'
        o_file = open(s_filename, 'a+')

        # Loop and write the body information
        for i_entry_spring in range(0, len(l_springs), 1):
            # Get the current state of the spring
            d_internal_force = l_springs[i_entry_spring].GetSpringReact()
            d_spring_velocity = l_springs[i_entry_spring].GetSpringVelocity()
            d_spring_length = l_springs[i_entry_spring].GetSpringLength()

            # Write to the file
            o_file.write(str(d_time) + '\t' + str(i_entry_spring) + '\t' + str(d_internal_force) + '\t' +
                         str(d_spring_velocity) + '\t' + str(d_spring_length) + '\n')

        # Close the output file
        o_file.close()


    def log_springs_h5(self, l_springs):
        """
        Logs chrono spring information to an h5 file
        Parameters
        ----------
        self: object
            ProtChSystem object being referenced.
        l_springs: list
            Spring objects for which the data is being logged
        Returns
        -------
        None. Data is logged to an h5 file.
        """

        # Open the log file
        o_file = h5py.File('chrono_log.h5')

        # Create the empty data holders
        l_spring_data = []

        # Loop and write the body information
        for i_entry_spring in range(0, len(l_springs), 1):
            # Get the current state of the spring
            d_internal_force = l_springs[i_entry_spring].GetSpringReact()
            d_spring_velocity = l_springs[i_entry_spring].GetSpringVelocity()
            d_spring_length = l_springs[i_entry_spring].GetSpringLength()

            # Write to the file
            l_spring_data.append([d_internal_force, d_spring_velocity, d_spring_length])

        # Log the spring data
        try:
            # Open the dataset
            dm_springs = o_file['springs']

            # Resize the dataset
            i_index = dm_springs.shape[0]
            dm_springs.resize(dm_springs.shape[0] + 1, axis=0)

            # Store the values into the dataset
            dm_springs[i_index, :, :] = np.array(l_spring_data)

        except:
            # Create the initial dataset
            dm_springs = o_file.create_dataset('springs', (1, len(l_spring_data), 3), compression="gzip",
                                               maxshape=(None, len(l_spring_data), 3), dtype=float)

            # Store the first entry into the dataset
            dm_springs[0, :, :] = np.array(l_spring_data)

        # Close the output file
        o_file.close()

    def log_residuals_text(self, d_time, l_linklocks):
        """
        Logs the chrono information into a text file at each timestep
        Parameters
        ----------
        self: object
            ProtChSystem being referenced
        d_time: float
            Current simulation time
        l_linklocks: list
            Link lock objects for which the residual information is desired.
        Returns
        -------
        None. Data is saved to a text file.
        """

        # Loop and write the body information
        for i_entry_linkage in range(0, len(l_linklocks), 1):
            # Open the file
            s_filename = 'chrono_' + str(i_entry_linkage) + '.txt'
            o_file = open(s_filename, 'a+')

            # Get the residual matrix from the linkage
            o_linkage_residual = l_linklocks[i_entry_linkage].GetC()

            # The number of residual components will vary based on the the number of active dimensions in the domain.
            # Attempt to get each component and write zeros for the remainder for consistency of the code.
            s_residuals = str(d_time) + '\t'

            for i_entry_residual in range(0, 6, 1):
                try:
                    # Attempt to extract the value from the Chrono matrix
                    s_residuals += str(o_linkage_residual.GetElement(0, i_entry_residual)) + '\t'

                except:
                    # Residual doesn't exist. Keep the zero in the array.
                    s_residuals += str(0.000) + '\t'

            # Add the new line to the code, replacing the last tab
            s_residuals += '\n'

            # Write to the file
            o_file.write(s_residuals)

        # Close the output file
        o_file.close()

    def log_residuals_h5(self, l_linklocks):
        """
        Logs chrono spring information to an h5 file
        Parameters
        ----------
        self: object
            ProtChSystem object being referenced.
        l_linklocks: list
            Link lock objects for which the residual information is desired.
        Returns
        -------
        None. Data is logged to an h5 file.
        """

        # Open the log file
        o_file = h5py.File('chrono_log.h5')

        # The number of residual components will vary based on the the number of active dimensions in the domain.
        # Attempt to get each component and write zeros for the remainder for consistency of the code.
        dm_residuals = np.zeros((len(l_linklocks), 6))

        for i_entry_linkage in range(0, len(l_linklocks), 1):
                        # Get the residual matrix from the linkage
            o_linkage_residual = l_linklocks[i_entry_linkage].GetC()

             # Extract the data from the Chrono matrix
            for i_entry_residual in range(0, 6, 1):
                try:
                    # Attemp to extract the value from the Chrono matrix
                    dm_residuals[i_entry_linkage, i_entry_residual] = o_linkage_residual.GetElement(0, i_entry_residual)
                except:
                    # Residual doesn't exist. Keep the zero in the array.
                    dm_residuals[i_entry_linkage, i_entry_residual] = 0

        # Log the spring data
        try:
            # Open the dataset
            dm_residuals_h5 = o_file['residuals']

            # Resize the dataset in the h5 file to expand it by an entry
            i_index = dm_residuals_h5.shape[0]
            dm_residuals_h5.resize(dm_residuals_h5.shape[0] + 1, axis=0)

            # Store the values into the dataset
            dm_residuals_h5[i_index, :, :] = dm_residuals

        except:
            # Create the initial dataset
            dm_residuals_h5 = o_file.create_dataset('residuals', (1, len(l_linklocks), 6), compression="gzip",
                                                    maxshape=(None, len(l_linklocks), 6), dtype=float)

            # Store the first entry into the dataset
            dm_residuals_h5[0, :, :] = dm_residuals

        # Close the output file
        o_file.close()


    def log_times_h5(self, d_time):
        """
        Creates a log of the Proteus timestep within the h5 log file
        Parameters
        ----------
        self: object
            ProtChSystem being referenced
        d_time: float
            Time within the simulation
        Returns
        -------
        None. Data is logged to the disk.
        """

        # Open the log file
        o_file = h5py.File('chrono_log.h5')

        # Log the time data
        try:
            # Open the dataset
            dm_time = o_file['time']

            # Resize the dataset
            i_index = dm_time.shape[0]
            dm_time.resize(dm_time.shape[0] + 1, axis=0)

            # Store the values into the dataset
            dm_time[i_index] = np.array(d_time)

        except:
            # Create the initial dataset
            dm_time = o_file.create_dataset('time', (1,), compression="gzip",
                                            maxshape=(None,), dtype=float)

            # Store the first entry into the dataset
            dm_time[0] = d_time

        # Close the file
        o_file.close()

    # def findFluidVelocityAtCoords(self, coords):
    #     """Finds solution from NS for velocity of fluid at given coordinates

    #     Parameters
    #     ----------
    #     coords: array_like
    #         coordinates at which velocity solution is sought

    #     Returns
    #     -------
    #     u: float
    #         velocity in the x direction
    #     v: float
    #         velocity in the y direction
    #     w: float
    #         velocity in the z direction (0 if 2D)
    #     """
    #     comm = Comm.get().comm.tompi4py()
    #     # get nearest node on each processor
    #     nearest_node, nearest_node_distance = getLocalNearestNode(coords, self.nodes_kdtree)
    #     # look for element containing coords on each processor (if it exists)
    #     local_element = getLocalElement(self.femSpace_velocity, coords, nearest_node)
    #     # check which processor has element (if any)
    #     haveElement = int(local_element is not None)
    #     if haveElement:
    #         owning_proc = comm.rank
    #     if local_element:
    #         # NEXT LINE TO CHANGE
    #         nd = self.nd
    #         # get local coords
    #         xi = self.femSpace_velocity.elementMaps.getInverseValue(local_element, coords)
    #         # get solution
    #         u = self.u[1].getValue(local_element, xi)
    #         v = self.u[2].getValue(local_element, xi)
    #         if nd > 2:
    #             w = self.u[3].getValue(local_element, xi)
    #         # broadcast to all processors
    #         u = comm.bcast(u, owning_proc)
    #         v = comm.bcast(v, owning_proc)
    #         if nd > 2:
    #             w = comm.bcast(w, owning_proc)
    #         if nd <= 2:
    #             w = 0
    #     else:
    #         sys.exit('{coords} outside of domain'.format(coords=str(coords)))
    #     return u, v, w

# ctypedef np.ndarray vecarray(ChVector)

# ctypedef np.ndarray (*ChVector_to_npArray) (ChVector)

#def testx():
#    cdef ChSystem system = ChSystem()
#    cdef ChBody bod = ChBody()
#    cdef ChVector oo = ChVector[double](2.,3.,4.)
#    bod.SetPosDt(oo)
#    cdef ChVector& gg = bod.GetPosDt()
#    print(gg.x, gg.y, gg.z)


cdef class ProtChMesh:

    def __cinit__(self, ProtChSystem system):
        self.ChMeshh = chrono_fea.ChMesh()
        cdef SwigPyObject *swig_obj = <SwigPyObject*> self.ChMeshh.this
        cdef shared_ptr[ch.ChMesh]* pt_to_shp = <shared_ptr[ch.ChMesh]*> swig_obj.ptr;
        self.mesh = pt_to_shp[0]
        system.addProtChMesh(self)

    def getChronoObject(self):
        return self.ChMeshh



# def linkMoorings(System system, Moorings mooringA, int nodeA_ind, Moorings mooringB, int nodeB_ind):
#     """
#     Parameters
#     ----------
#     system: System
#         class instance of the System where to add link
#     mooringA: Moorings
#         class instance of first mooring cable (A)
#     nodeA_ind: int
#         index of node to be used as link on mooring cable A
#     mooringB: Moorings
#         class instance of second mooring cable (B)
#     nodeB_ind: int
#         index of node to be used as link on mooring cable B
#     """
#     cdef shared_ptr[ch.ChLinkPointPoint] link = make_shared[ch.ChLinkPointPoint]()
#     deref(link).Initialize(<shared_ptr[ch.ChNodeFEAxyz]> mooringA.thisptr.nodes[nodeA_ind], <shared_ptr[ch.ChNodeFEAxyz]> mooringB.thisptr.nodes[nodeB_ind])
#     system.thisptr.system.Add(<shared_ptr[ch.ChPhysicsItem]> link)


cdef class ProtChMoorings:
    """Class for building mooring cables

    Parameters
    ----------
    system: System
        Class instance of the system.
    mesh: Mesh
        Class instance of the mesh.
    length: np.ndarray
        Length of cable segments. Must be an array, if the cable only
        has one type of segment (e.g. only one type of chain), an
        array of length 1 can be passed.
    nb_elems: np.ndarray
        Number of elements per segments.
    d: np.ndarray
        Diameter of segments.
    rho: np.ndarray
        Density of segments.
    E: np.ndarray
        Young's modulus of segments.
    beam_type: str
        Type of elements (default: "CableANCF").
    """

    def __cinit__(self,
                  ProtChSystem system,
                  ProtChMesh mesh,
                  double[:] length,
                  np.ndarray nb_elems,
                  double[:] d,
                  double[:] rho,
                  double[:] E,
                  string beam_type=b"CableANCF"):
        check_arrays = [len(length), len(nb_elems), len(d), len(rho), len(E)]
        assert all(v == len(length) for v in check_arrays), 'arrays are not of same length'
        self.ProtChSystem = system
        self.ProtChSystem.addSubcomponent(self)
        self.nd = self.ProtChSystem.nd
        self.Mesh = mesh
        self.beam_type = beam_type
        self.nb_elems = nb_elems
        cdef vector[double] vec_length
        cdef vector[int] vec_nb_elems
        cdef vector[double] vec_d
        cdef vector[double] vec_rho
        cdef vector[double] vec_E
        for i in range(len(length)):
            vec_length.push_back(length[i])
            vec_nb_elems.push_back(nb_elems[i])
            vec_d.push_back(d[i])
            vec_rho.push_back(rho[i])
            vec_E.push_back(E[i])
        self.thisptr = newMoorings(system.thisptr.system,
                                   mesh.mesh,
                                   vec_length,
                                   vec_nb_elems,
                                   vec_d,
                                   vec_rho,
                                   vec_E,
                                   beam_type
                                   )
        self.nodes_function = lambda s: (s, s, s)
        self.nodes_built = False
        name=b'mooring'
        self.setName(name)
        self.external_forces_from_ns = True
        self.external_forces_manual = True
        self._record_etas = np.array([0.])
        self._record_names = ['ux', 'uy', 'uz',
                              'ax', 'ay', 'az',
                              'fluid_ux', 'fluid_uy', 'fluid_uz',
                              'fluid_ax', 'fluid_ay', 'fluid_az',
                              'dragx', 'dragy', 'dragz',
                              'amx', 'amy', 'amz']
        self._record_etas_names = []
        for eta in self._record_etas:
            self._record_etas_names += ['sx'+str(eta), 'sy'+str(eta), 'sz'+str(eta)]
        self.tCount = 0

    def setName(self, string name):
        """Sets name of cable, used for csv file

        Parameters
        ----------
        name: str
            Name of cable.
        """
        self.name = name

    def recordStrainEta(self, double[:] etas):
        self._record_etas = etas
        self._record_etas_names = []
        for i in range(len(self._record_etas)):
            eta = self._record_etas[i]
            self._record_etas_names += ['sx'+str(eta),
                                        'sy'+str(eta),
                                        'sz'+str(eta)]

    def _recordH5(self):
        Profiling.logEvent('recording h5 file of '+str(self.name))
        tCount = self.tCount
        t = self.ProtChSystem.ChSystem.GetChTime()
        self.hdfFileName = self.name
        hdfFileName = os.path.join(bytes(Profiling.logDir,'utf-8'), self.hdfFileName)+b'.h5'
        if tCount == 0:
            f = h5py.File(hdfFileName, 'w', libver='latest')
        else:
            f = h5py.File(hdfFileName, 'a', libver='latest')
        pos = self.getNodesPosition()
        element_connection = np.array([[i, i+1] for i in range(len(pos)-1)])
        dset = f.create_dataset('nodesSpatial_Domain'+str(tCount), pos.shape)
        dset[...] = pos
        dset = f.create_dataset('elementsSpatial_Domain'+str(tCount), element_connection.shape, dtype='i8')
        dset[...] = element_connection
        # time
        datav = t
        dset = f.create_dataset('t_t'+str(tCount), (1,))
        dset[0] = t
        # strain
        for i in range(len(self._record_etas)):
            eta = self._record_etas[i]
            datav = np.append(self.getNodesTension(eta=eta), np.array([[0.,0.,0.]]), axis=0)
            dset = f.create_dataset('tensions'+str(eta)+'_t'+str(tCount), datav.shape)
            dset[...] = datav
            dset = f.create_dataset('sx'+str(eta)+'_t'+str(tCount), (datav.shape[0],))
            dset[...] = datav[:,0]
            dset = f.create_dataset('sy'+str(eta)+'_t'+str(tCount), (datav.shape[0],))
            dset[...] = datav[:,1]
            dset = f.create_dataset('sz'+str(eta)+'_t'+str(tCount), (datav.shape[0],))
            dset[...] = datav[:,2]
        # velocity
        datav = self.getNodesVelocity()
        dset = f.create_dataset('velocity_t'+str(tCount), datav.shape)
        dset[...] = datav
        dset = f.create_dataset('ux_t'+str(tCount), (datav.shape[0],))
        dset[...] = datav[:,0]
        dset = f.create_dataset('uy_t'+str(tCount), (datav.shape[0],))
        dset[...] = datav[:,1]
        dset = f.create_dataset('uz_t'+str(tCount), (datav.shape[0],))
        dset[...] = datav[:,2]
        # acceleration
        datav = self.getNodesAcceleration()
        dset = f.create_dataset('acceleration_t'+str(tCount), datav.shape)
        dset[...] = datav
        dset = f.create_dataset('ax_t'+str(tCount), (datav.shape[0],))
        dset[...] = datav[:,0]
        dset = f.create_dataset('ay_t'+str(tCount), (datav.shape[0],))
        dset[...] = datav[:,1]
        dset = f.create_dataset('az_t'+str(tCount), (datav.shape[0],))
        dset[...] = datav[:,2]
        # fluid velocity
        datav = self.fluid_velocity_array
        dset = f.create_dataset('fluid_velocity_t'+str(tCount), datav.shape)
        dset[...] = datav
        dset = f.create_dataset('fluid_ux_t'+str(tCount), (datav.shape[0],))
        dset[...] = datav[:,0]
        dset = f.create_dataset('fluid_uy_t'+str(tCount), (datav.shape[0],))
        dset[...] = datav[:,1]
        dset = f.create_dataset('fluid_uz_t'+str(tCount), (datav.shape[0],))
        dset[...] = datav[:,2]
        # fluid acceleration
        datav = self.fluid_acceleration_array
        dset = f.create_dataset('fluid_acceleration_t'+str(tCount), datav.shape)
        dset[...] = datav
        dset = f.create_dataset('fluid_ax_t'+str(tCount), (datav.shape[0],))
        dset[...] = datav[:,0]
        dset = f.create_dataset('fluid_ay_t'+str(tCount), (datav.shape[0],))
        dset[...] = datav[:,1]
        dset = f.create_dataset('fluid_az_t'+str(tCount), (datav.shape[0],))
        dset[...] = datav[:,2]
        # drag
        datav = self.getDragForces()
        dset = f.create_dataset('drag_t'+str(tCount),datav.shape)
        dset[...] = datav
        dset = f.create_dataset('dragx_t'+str(tCount), (datav.shape[0],))
        dset[...] = datav[:,0]
        dset = f.create_dataset('dragy_t'+str(tCount), (datav.shape[0],))
        dset[...] = datav[:,1]
        dset = f.create_dataset('dragz_t'+str(tCount), (datav.shape[0],))
        dset[...] = datav[:,2]
        # addedmass
        datav = self.getAddedMassForces()
        dset = f.create_dataset('am_t'+str(tCount), datav.shape)
        dset[...] = datav
        dset = f.create_dataset('amx_t'+str(tCount), (datav.shape[0],))
        dset[...] = datav[:,0]
        dset = f.create_dataset('amy_t'+str(tCount), (datav.shape[0],))
        dset[...] = datav[:,1]
        dset = f.create_dataset('amz_t'+str(tCount), (datav.shape[0],))
        dset[...] = datav[:,2]
        # close file
        f.close()
        Profiling.logEvent('finished recording h5 file of '+str(self.name))

    def _recordXML(self):
        Profiling.logEvent('recording xml file of '+str(self.name))
        tCount = self.tCount
        t = self.ProtChSystem.ChSystem.GetChTime()
        xmlFile = str(os.path.join(bytes(Profiling.logDir,'utf-8'), self.name)+b'.xmf','utf-8')
        # if tCount == 0:
        root = ET.Element("Xdmf",
                          {"Version": "2.0",
                           "xmlns:xi": "http://www.w3.org/2001/XInclude"})
        domain = ET.SubElement(root, "Domain")
        arGridCollection = ET.SubElement(domain,
                                         "Grid",
                                         {"Name": "Mesh"+" Spatial_Domain",
                                          "GridType": "Collection",
                                          "CollectionType": "Temporal"})
        # else:
        #     tree = ET.parse(xmlFile)
        #     root = tree.getroot()
        #     domain = root[0]
        #     arGridCollection = domain[0]
        Xdmf_ElementTopology = "Polyline"
        pos = self.getNodesPosition()
        Xdmf_NumberOfElements= len(pos)-1
        Xdmf_NodesPerElement = 2
        dataItemFormat = "HDF"

        arGrid = ET.SubElement(arGridCollection,
                               "Grid",
                               {"GridType": "Uniform"})
        arTime = ET.SubElement(arGrid,
                               "Time",
                               {"Value": "{0:e}".format(t),
                                "Name": "{0:d}".format(tCount)})
        topology = ET.SubElement(arGrid,
                                "Topology",
                                {"Type": Xdmf_ElementTopology,
                                 "NumberOfElements": "{0:d}".format(Xdmf_NumberOfElements),
                                 "NodesPerElement": "{0:d}".format(Xdmf_NodesPerElement),
                                })

        elements = ET.SubElement(topology,
                                 "DataItem",
                                 {"Format": dataItemFormat,
                                  "DataType": "Int",
                                  "Dimensions": "{0:d} {1:d}".format(Xdmf_NumberOfElements,
                                                         Xdmf_NodesPerElement)})
        elements.text = "{0}.h5:/elementsSpatial_Domain{1:d}".format(str(self.hdfFileName,'utf-8'),tCount)
        geometry = ET.SubElement(arGrid,"Geometry",{"Type":"XYZ"})
        nodes = ET.SubElement(geometry,
                              "DataItem",
                              {"Format": dataItemFormat,
                               "DataType": "Float",
                               "Precision": "8",
                               "Dimensions": "{0:d} {1:d}".format(pos.shape[0], pos.shape[1])})
        nodes.text = "{0}.h5:/nodesSpatial_Domain{1:d}".format(str(self.hdfFileName,'utf-8'),tCount)
        all_names = self._record_names+self._record_etas_names
        for name in all_names:
            attr = ET.SubElement(arGrid,
                                 "Attribute",
                                 {"Name": name,
                                  "AttributeType": "Scalar",
                                  "Center": "Node"})
            data = ET.SubElement(attr,
                                 "DataItem",
                                 {"Format": dataItemFormat,
                                  "DataType": "Float",
                                  "Precision": "8",
                                  "Dimensions": "{0:d}".format(pos.shape[0])})
            data.text = "{0}.h5:/{1}_t{2:d}".format(str(self.hdfFileName,'utf-8'), name, tCount)

        tree = ET.ElementTree(root)

        with open(xmlFile, "wb") as f:
            xmlHeader = "<?xml version=\"1.0\" ?>\n<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n"
            f.write(bytes(xmlHeader,"utf-8"))
            indentXML(tree.getroot())
            tree.write(f)

        # dump xml str in h5 file
        hdfFileName = os.path.join(bytes(Profiling.logDir,'utf-8'), self.hdfFileName)+b'.h5'
        f = h5py.File(hdfFileName, 'a', libver='latest')
        datav = ET.tostring(arGrid)
        dset = f.create_dataset('Mesh_Spatial_Domain_{0:d}'.format(tCount),
                                (1,),
                                dtype=h5py.special_dtype(vlen=str))
        dset[...] = datav
        # close file
        f.close()
        Profiling.logEvent('finished recording xmf file of '+str(self.name))

    def _recordValues(self):
        """Records values in csv files
        """
        Profiling.logEvent('recording values file of '+str(self.name))
        self.record_file = str(os.path.join(bytes(Profiling.logDir,'utf-8'), self.name),'utf-8')
        def record(record_file, row, mode='a'):
            with open(record_file, mode) as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow(row)
        t_chrono = self.ProtChSystem.ChSystem.GetChTime()
        if self.ProtChSystem.model is not None:
            t_last = self.ProtChSystem.model.stepController.t_model_last
            try:
                dt_last = self.ProtChSystem.model.levelModelList[-1].dt_last
            except:
                dt_last = 0
            t = t_last
        else:
            t = t_chrono
        t_sim = Profiling.time()-Profiling.startTime
        if t == 0:
            header_x = []
            for i in range(self.thisptr.nodes.size()):
                header_x += ['x'+str(i), 'y'+str(i), 'z'+str(i)]
        # time
        file_name = '_t.csv'
        if t == 0:
            row = ['t', 't_ch', 't_sim']
            record(self.record_file+file_name, row, 'w')
        row = [t, t_chrono, t_sim]
        record(self.record_file+file_name, row)
        # # Positions
        # file_name = '_pos.csv'
        # if t == 0:
        #     record(self.record_file+file_name, header_x, 'w')
        # positions = self.getNodesPosition()
        # row = (positions.flatten('C')).tolist()
        # record(self.record_file+file_name, row)
        # # Velocity
        # file_name = '_posdt.csv'
        # if t == 0:
        #     record(self.record_file+file_name, header_x, 'w')
        # velocities = self.getNodesVelocity()
        # row = (velocities.flatten('C')).tolist()
        # record(self.record_file+file_name, row)
        # # Acceleration
        # file_name = '_posdtdt.csv'
        # if t == 0:
        #     record(self.record_file+file_name, header_x, 'w')
        # accelerations = self.getNodesAcceleration()
        # row = (accelerations.flatten('C')).tolist()
        # record(self.record_file+file_name, row)
        # Fairlead / anchor tensions
        file_name = '_tens.csv'
        if t == 0:
            row = ['Tb0', 'Tb1', 'Tb2', 'Tf0', 'Tf1', 'Tf2']
            record(self.record_file+file_name, row, 'w')
        Tb = self.getTensionBack()
        Tf = self.getTensionFront()
        row = [Tb[0], Tb[1], Tb[2], Tf[0], Tf[1], Tf[2]]
        record(self.record_file+file_name, row)
        # Tensions
        # for i in range(len(self._record_etas)):
        #     eta = self._record_etas[i]
        #     file_name = '_strain'+str(eta)+'.csv'
        #     if t == 0:
        #         record(self.record_file+file_name, header_x[:-3], 'w')
        #     tensions = self.getNodesTension(eta=eta)
        #     row = (tensions.flatten('C')).tolist()
        #     record(self.record_file+file_name, row)
        # # Drag
        # file_name = '_drag.csv'
        # if t == 0:
        #     record(self.record_file+file_name, header_x, 'w')
        # forces = self.getDragForces()
        # row = (forces.flatten('C')).tolist()
        # record(self.record_file+file_name, row)
        # # Added mass
        # file_name = '_AM.csv'
        # if t == 0:
        #     record(self.record_file+file_name, header_x, 'w')
        # forces = self.getAddedMassForces()
        # row = (forces.flatten('C')).tolist()
        # record(self.record_file+file_name, row)
        # # Fluid Velocity
        # file_name = '_u.csv'
        # if t == 0:
        #     record(self.record_file+file_name, header_x, 'w')
        # velocities = self.fluid_velocity_array
        # row = (velocities.flatten('C')).tolist()
        # record(self.record_file+file_name, row)
        # # Fluid Acceleration
        # file_name = '_udt.csv'
        # if t == 0:
        #     record(self.record_file+file_name, header_x, 'w')
        # accelerations = self.fluid_acceleration_array
        # row = (accelerations.flatten('C')).tolist()
        # record(self.record_file+file_name, row)
        Profiling.logEvent('finished recording values file of '+str(self.name))

    def getTensionBack(self):
        """
        Get Tension at the back of the cable
        """
        cdef ch.ChVector3d T
        if self.thisptr.constraint_back:
            T = deref(self.thisptr.constraint_back).GetReaction2().force
            return np.array([T.x(), T.y(), T.z()])
        else:
            return np.zeros(3)

    def getTensionFront(self):
        """
        Get Tension at the front of the cable
        """
        cdef ch.ChVector3d T
        if self.thisptr.constraint_front:
            T = deref(self.thisptr.constraint_front).GetReaction1().force
            return np.array([T.x(), T.y(), T.z()])
        else:
            return np.zeros(3)

    def calculate_init(self):
        # build position vector of nodes (for each segment)
        # self.setNodesPosition()
        # build cable (nodes, elements, etc)
        self.thisptr.buildCable()
        if self.beam_type == b"BeamEuler":
            nb_nodes = self.thisptr.nodesRot.size()
        else:
            nb_nodes = self.thisptr.nodes.size()
        if self.fluid_velocity_array is None:
            self.fluid_velocity_array = np.zeros((nb_nodes, 3))
            self.fluid_velocity_array_previous = np.zeros((nb_nodes, 3))
        if self.fluid_acceleration_array is None:
            self.fluid_acceleration_array = np.zeros((nb_nodes, 3))
        if self.fluid_density_array is None:
            self.fluid_density_array = np.zeros(nb_nodes)
        self.nearest_node_array = np.zeros(nb_nodes, dtype=np.int32)
        self.containing_element_array = np.zeros(nb_nodes, dtype=np.int32)
        self.owning_rank = np.zeros(nb_nodes, dtype=np.int32)

    def prestep(self):
        """Sets external forces on the cable (if any)
        """
        if self.ProtChSystem.model is not None and self.external_forces_from_ns is True:
            self.setExternalForces()
        elif self.external_forces_manual is True:
            self.setExternalForces()

    def poststep(self):
        """Records values
        """
        if self.initialized is False:
            self.initialized = True
        comm = Comm.get().comm.tompi4py()
        if comm.rank == 0 and self.ProtChSystem.record_values is True:
            self._recordValues()
            self._recordH5()
            self._recordXML()
            self.tCount += 1

    def setApplyDrag(self, bool boolval):
        for i in range(self.thisptr.cables.size()):
            deref(self.thisptr.cables[i]).applyDrag = boolval

    def setApplyAddedMass(self, bool boolval):
        for i in range(self.thisptr.cables.size()):
            deref(self.thisptr.cables[i]).applyAddedMass = boolval

    def setApplyBuoyancy(self, bool boolval):
        for i in range(self.thisptr.cables.size()):
            deref(self.thisptr.cables[i]).applyBuoyancy = boolval

    def setNodesPositionFunction(self, function_position, function_tangent=None):
        """Function to build nodes

        Parameters
        ----------
        function_position:
            Must be a function taking one argument (e.g. distance
            along cable) and returning 3 arguments (x, y, z) coords.
        function_position: Optional
            Must be a function taking one argument (e.g. distance
            along cable) and returning 3 arguments (x, y, z) tangents at coords.
        """
        self.nodes_function = function_position
        self.nodes_function_tangent = function_tangent

    def setFluidVelocityFunction(self, function):
        """Function to build nodes

        Parameters
        ----------
        function:
            Must be a function taking two arguments (3D coordinates
            and time), and returning velocity (x, y, z).
        """
        self.fluid_velocity_function = function

    def fixFrontNode(self, bool fixed):
        """Fix front node of cable

        Parameters
        ----------
        fixed: bool
            Fixes node if True
        """
        assert self.nodes_built is True, 'call buildNodes() before calling this function'
        if self.beam_type == b"BeamEuler":
            deref(self.thisptr.nodesRot.front()).SetFixed(fixed)
        else:
            deref(self.thisptr.nodes.front()).SetFixed(fixed)

    def fixBackNode(self, bool fixed):
        """Fix back node of cable

        Parameters
        ----------
        fixed: bool
            Fixes node if True
        """
        assert self.nodes_built is True, 'call buildNodes() before calling this function'
        if self.beam_type == b"BeamEuler":
            deref(self.thisptr.nodesRot.back()).SetFixed(fixed)
        else:
            deref(self.thisptr.nodes.back()).SetFixed(fixed)

    def attachBackNodeToBody(self, ProtChBody body):
        """Attaches back node to a body with ChLinkLockLock

        Parameters
        ----------
        body: ProtChBody
            body to which the node will be attached
        """
        assert self.nodes_built is True, 'call buildNodes() before calling this function'
        self.thisptr.attachBackNodeToBody(body.thisptr.body)

    def attachFrontNodeToBody(self, ProtChBody body):
        """Attaches front node to a body with ChLinkLockLock

        Parameters
        ----------
        body: ProtChBody
            body to which the node will be attached
        """
        assert self.nodes_built is True, 'call buildNodes() before calling this function'
        self.thisptr.attachFrontNodeToBody(body.thisptr.body)

    def getTensionElement(self, int i=0, eta=0.):
        cdef ch.ChVector3d F
        F = self.thisptr.getTensionElement(i, eta)
        return np.array([F.x(), F.y(), F.z()])

    def getNodesTension(self, eta=0.):
        cdef ch.ChVector3d vec
        if self.beam_type == b'BeamEuler':
            T = np.zeros((self.thisptr.nodesRot.size()-1,3 ))
        else:
            T = np.zeros(( self.thisptr.nodes.size()-1,3 ))
        for i in range(np.sum(self.nb_elems)):
            vec = self.thisptr.getTensionElement(i, eta)
            T[i] = [vec.x(), vec.y(), vec.z()]
        return T

    def setDragCoefficients(self, double tangential, double normal, int segment_nb):
        """Sets drag coefficients of cable

        Parameters
        ----------
        tangential: double
            Tangential drag coefficient.
        normal: double
            Normal drag coefficient.
        segment_nb: int
            Segment number to which these coefficients apply.
        """
        deref(self.thisptr.cables[segment_nb]).setDragCoefficients(tangential, normal)

    def setAddedMassCoefficients(self, double tangential, double normal, int segment_nb):
        """Sets added mass coefficients of cable

        Parameters
        ----------
        tangential: double
            Tangential added mass coefficient.
        normal: double
            Normal added mass coefficient.
        segment_nb: int
            Segment number to which these coefficients apply.
        """
        deref(self.thisptr.cables[segment_nb]).setAddedMassCoefficients(tangential, normal)

    def setRestLengthPerElement(self, double[:] length_array, int segment_nb):
        """Sets rest length per element of cable

        Parameters
        ----------
        length_array: array_like[double]
            Rest length of each element of cable.
        segment_nb: int
            Segment number to which these rest lengths apply.
        """
        assert len(length_array) == deref(self.thisptr.cables[segment_nb]).nb_elems, 'array of length of elements not matching number of elements'
        cdef vector[double] vec
        for length in length_array:
            vec.push_back(length)
        deref(self.thisptr.cables[segment_nb]).setRestLengthPerElement(vec)

    def setNodesPosition(self, double[:,:,:] positions=None, tangents=None):
        """Builds the nodes of the cable.

        (!) Must be called after setNodesPositionFunction()
        """
        # self.nodes_positions0 = positions
        # self.nodes_tangents0 = tangents
        cdef ch.ChVector3d vec
        if positions is None:
            for i in range(self.thisptr.cables.size()):
                deref(self.thisptr.cables[i]).mvecs.clear()
                L0 = deref(self.thisptr.cables[i]).L0
                L = deref(self.thisptr.cables[i]).length
                nb_elems = deref(self.thisptr.cables[i]).nb_elems
                if self.beam_type == b"CableANCF" or self.beam_type == b"BeamEuler":
                    nb_nodes = nb_elems+1
                else:
                    print("set element type")
                    sys.exit()
                ds = L/(nb_nodes-1)
                for j in range(nb_nodes):
                    x, y, z = self.nodes_function(L0+ds*j)
                    vec = ch.ChVector3d(x, y, z)
                    deref(self.thisptr.cables[i]).mvecs.push_back(vec)
        else:
            for i in range(self.thisptr.cables.size()):
                deref(self.thisptr.cables[i]).mvecs.clear()
                nb_nodes = len(positions[i])
                for j in range(len(positions[i])):
                    x, y, z = positions[i][j]
                    vec = ch.ChVector3d(x, y, z)
                    deref(self.thisptr.cables[i]).mvecs.push_back(vec)
        if tangents is None:
            for i in range(self.thisptr.cables.size()):
                deref(self.thisptr.cables[i]).mvecs_tangents.clear()
                L0 = deref(self.thisptr.cables[i]).L0
                L = deref(self.thisptr.cables[i]).length
                nb_elems = deref(self.thisptr.cables[i]).nb_elems
                if self.beam_type == b"CableANCF" or self.beam_type == b"BeamEuler":
                    nb_nodes = nb_elems+1
                else:
                    print("set element type")
                    sys.exit()
                ds = L/(nb_nodes-1)
                for j in range(nb_nodes):
                    x, y, z = self.nodes_function_tangent(L0+ds*j)
                    vec = ch.ChVector3d(x, y, z)
                    deref(self.thisptr.cables[i]).mvecs_tangents.push_back(vec)
        else:
            for i in range(self.thisptr.cables.size()):
                deref(self.thisptr.cables[i]).mvecs_tangents.clear()
                nb_nodes = len(tangents[i])
                for j in range(len(tangents[i])):
                    x, y, z = tangents[i][j]
                    vec = ch.ChVector3d(x, y, z)
                    deref(self.thisptr.cables[i]).mvecs_tangents.push_back(vec)
        # self.buildNodes()

    def buildNodes(self):
        # build nodes
        self.thisptr.buildNodes()
        # get pointers to access nodes in python
        cdef SwigPyObject *swig_obj
        cdef int nodeN
        self.nodes = []
        for nodeN in range(self.thisptr.nb_nodes_tot):
            if self.beam_type == b"BeamEuler":
                node = chrono_fea.ChNodeFEAxyzrot()
            elif self.beam_type == b"CableANCF":
                node = chrono_fea.ChNodeFEAxyzD()
            node.this.disown()
            swig_obj = <SwigPyObject*> node.this
            if self.beam_type == b"BeamEuler":
                swig_obj.ptr = <shared_ptr[ch.ChNodeFEAxyzrot]*> &self.thisptr.nodesRot.at(nodeN)
            elif self.beam_type == b"CableANCF":
                swig_obj.ptr = <shared_ptr[ch.ChNodeFEAxyzD]*> &self.thisptr.nodes.at(nodeN)
            self.nodes += [node]
        self.nodes_built = True
        # build elements
        self.thisptr.buildElements()
        # get pointers to access elements in python
        cdef int elemN
        self.elements = []
        for elemN in range(self.thisptr.nb_elems_tot):
            if self.beam_type == b"BeamEuler":
                elem = chrono_fea.ChElementBeamEuler()
            elif self.beam_type == b"CableANCF":
                elem = chrono_fea.ChElementCableANCF()
            elem.this.disown()
            swig_obj = <SwigPyObject*> elem.this
            if self.beam_type == b"BeamEuler":
                swig_obj.ptr = <shared_ptr[ch.ChElementBeamEuler]*> &self.thisptr.elemsBeamEuler.at(elemN)
            elif self.beam_type == b"CableANCF":
                swig_obj.ptr = <shared_ptr[ch.ChElementCableANCF]*> &self.thisptr.elemsCableANCF.at(elemN)
            self.elements += [elem]
        if self.beam_type == b"BeamEuler":
            self.nodes_nb = self.thisptr.nodesRot.size()
        elif self.beam_type == b"CableANCF":
            self.nodes_nb = self.thisptr.nodes.size()
        else:
            print("set element type")
            sys.exit()

    def getNodesPosition(self):
        """Gives array of nodes position

        Returns
        -------
        pos: np.ndarray
            Array of nodes position.
        """
        if self.beam_type == b'BeamEuler':
            pos = np.zeros(( self.thisptr.nodesRot.size(),3 ))
            for i in range(self.thisptr.nodesRot.size()):
                vec = deref(self.thisptr.nodesRot[i]).GetPos()
                pos[i] = [vec.x(), vec.y(), vec.z()]
            return pos
        else:
            pos = np.zeros(( self.thisptr.nodes.size(),3 ))
            for i in range(self.thisptr.nodes.size()):
                vec = deref(self.thisptr.nodes[i]).GetPos()
                pos[i] = [vec.x(), vec.y(), vec.z()]
            return pos

    def getNodesVelocity(self):
        """Gives array of nodes velocity

        Returns
        -------
        pos: np.ndarray
            Array of nodes velocity.
        """
        if self.beam_type == b'BeamEuler':
            pos = np.zeros(( self.thisptr.nodesRot.size(),3 ))
            for i in range(self.thisptr.nodesRot.size()):
                vec = deref(self.thisptr.nodesRot[i]).GetPosDt()
                pos[i] = [vec.x(), vec.y(), vec.z()]
            return pos
        else:
            pos = np.zeros(( self.thisptr.nodes.size(),3 ))
            for i in range(self.thisptr.nodes.size()):
                vec = deref(self.thisptr.nodes[i]).GetPosDt()
                pos[i] = [vec.x(), vec.y(), vec.z()]
            return pos

    def getNodesAcceleration(self):
        """Gives array of nodes acceleration

        Returns
        -------
        pos: np.ndarray
            Array of nodes acceleration.
        """
        if self.beam_type == b'BeamEuler':
            pos = np.zeros((self.nodes_nb,3 ))
            for i in range(self.thisptr.nodesRot.size()):
                vec = deref(self.thisptr.nodesRot[i]).GetPosDt2()
                pos[i] = [vec.x(), vec.y(), vec.z()]
            return pos
        else:
            pos = np.zeros(( self.thisptr.nodes.size(),3 ))
            for i in range(self.thisptr.nodes.size()):
                vec = deref(self.thisptr.nodes[i]).GetPosDt2()
                pos[i] = [vec.x(), vec.y(), vec.z()]
            return pos

    def getDragForces(self):
        cdef ch.ChVector3d Fd
        drag = np.zeros((self.nodes_nb,3 ))
        for i in range(self.thisptr.forces_drag.size()):
            Fd = deref(self.thisptr.forces_drag[i])
            drag[i] = [Fd.x(), Fd.y(), Fd.z()]
        return drag

    def getAddedMassForces(self):
        cdef ch.ChVector3d Fd
        drag = np.zeros((self.nodes_nb,3 ))
        for i in range(self.thisptr.forces_addedmass.size()):
            Fd = deref(self.thisptr.forces_addedmass[i])
            drag[i] = [Fd.x(), Fd.y(), Fd.z()]
        return drag

    def setIyy(self, double Iyy, int cable_nb):
        deref(self.thisptr.cables[cable_nb]).setIyy(Iyy)


    def getNodesD(self):
        """Gives direction of nodes

        Returns
        -------
        dire: np.ndarray
            Array of nodes direction.
        """
        dire = np.zeros(( self.thisptr.nodes.size(),3 ))
        for i in range(self.thisptr.nodes.size()):
            vec = deref(self.thisptr.nodes[i]).GetSlope1()
            dire[i] = [vec.x(), vec.y(), vec.z()]
        return dire

    def setContactMaterial(self, mat):
        """Sets contact material of the cable

        Parameters
        ----------
        mat: ChContactMaterialSMC
            Material of cable.
        """
        cdef SwigPyObject *swig_obj = <SwigPyObject*> mat.this
        cdef shared_ptr[ch.ChContactMaterialSMC]* pt_to_shp = <shared_ptr[ch.ChContactMaterialSMC]*> swig_obj.ptr;
        cdef shared_ptr[ch.ChContactMaterialSMC] matp = pt_to_shp[0]
        self.thisptr.setContactMaterial(matp)

    def setExternalForces(self, fluid_velocity_array=None, fluid_density_array=None,
                          fluid_acceleration_array=None):
        """
        Sets external forces acting on cables
        Pass fluid velocity_array as argument only for debugging (must be an array as long as the number of nodes)
        """
        # get velocity at nodes
        # cdef np.ndarray fluid_velocity = np.zeros((len(self.thisptr.nodes.size()), 3))
        self.fluid_velocity_array_previous[:] = self.fluid_velocity_array
        if fluid_velocity_array is not None:
            self.fluid_velocity_array = fluid_velocity_array
        if fluid_density_array is not None:
            self.fluid_density_array = fluid_density_array
        if fluid_acceleration_array is not None:
            self.fluid_acceleration_array = fluid_acceleration_array
        cdef vector[ch.ChVector3d] fluid_velocity
        cdef vector[ch.ChVector3d] fluid_acceleration
        cdef ch.ChVector3d vel
        cdef ch.ChVector3d acc
        cdef vector[double] fluid_density
        cdef double dens
        comm = Comm.get().comm.tompi4py()
        if self.beam_type == b"BeamEuler":
            nb_nodes = self.thisptr.nodesRot.size()
        else:
            nb_nodes = self.thisptr.nodes.size()
        cdef bool mesh_search = False
        cdef bool dist_search = False
        if self.ProtChSystem.model is not None and self.external_forces_from_ns is True:
            mesh_search = True
            if self.ProtChSystem.dist_search is True:
                dist_search = True
                if self.ProtChSystem.first_step is True:
                    if self.ProtChSystem.build_kdtree is True:
                        dist_search = False
            if dist_search is True:
                Profiling.logEvent("Starting distance search for cable nodes")
            else:
                Profiling.logEvent("Starting k-d tree search for cable nodes")
        for i in range(nb_nodes):
            if self.beam_type == b"BeamEuler":
                vec = deref(self.thisptr.nodesRot[i]).GetPos()
            else:
                vec = deref(self.thisptr.nodes[i]).GetPos()
            x = vec.x()
            y = vec.y()
            z = vec.z()
            coords = np.array([x, y, z])
            vel_arr = np.zeros(3)
            if mesh_search is True:
                vel_grad_arr = np.zeros(3)
                if dist_search is True:
                    xi, nearest_node, el, rank = self.ProtChSystem.findElementContainingCoordsDist(coords=coords[:self.nd],
                                                                                                   node_guess=self.nearest_node_array[i],
                                                                                                   eN_guess=self.containing_element_array[i],
                                                                                                   rank_guess=self.owning_rank[i])
                else:
                    xi, nearest_node, el, rank = self.ProtChSystem.findElementContainingCoordsKD(coords[:self.nd])
                if el is None:
                    el = -1
                self.nearest_node_array[i] = nearest_node
                self.containing_element_array[i] = el
                self.owning_rank[i] = rank
                comm.barrier()
                if rank is not None and xi is not None:
                    vel_arr[:] = self.ProtChSystem.getFluidVelocityLocalCoords(xi, el, rank)
                else:  # means node is outside domain
                    if self.fluid_velocity_function is not None:
                        vel_arr[:] = self.fluid_velocity_function(coords, self.ProtChSystem.t)
                    else:
                        vel_arr[:] = 0.
                comm.barrier()
            else:
                if self.fluid_velocity_function is not None:
                    vel_arr[:] = self.fluid_velocity_function(coords, self.ProtChSystem.t)
                else:
                    vel_arr[:] = 0
            self.fluid_velocity_array[i] = vel_arr
            vel = ch.ChVector3d(vel_arr[0], vel_arr[1], vel_arr[2])
            if self.fluid_velocity_function is not None and fluid_velocity_array is None:
                vel_arr = self.fluid_velocity_function(coords, self.ProtChSystem.t)
                vel = ch.ChVector3d(vel_arr[0], vel_arr[1], vel_arr[2])
            else:
                vel = ch.ChVector3d(self.fluid_velocity_array[i][0], self.fluid_velocity_array[i][1], self.fluid_velocity_array[i][2])
            fluid_velocity.push_back(vel)
            self.fluid_acceleration_array[i] = (self.fluid_velocity_array[i]-self.fluid_velocity_array_previous[i])/self.ProtChSystem.proteus_dt
            # acc = du/dt+u.grad(u)
            #vel_grad_arr[:] = self.ProtChSystem.getFluidVelocityGradientLocalCoords(xi, el, rank)
            #acc_arr = (vel_arr-fluid_velocity_array_previous[i])/dt+vel_arr*vel_grad_arr
            #arr[:self.nd] = self.ProtChSystem.findFluidVelocityAtCoords(coords[:self.nd])
            acc = ch.ChVector3d(self.fluid_acceleration_array[i][0], self.fluid_acceleration_array[i][1], self.fluid_acceleration_array[i][2])
            fluid_acceleration.push_back(acc)
            dens = self.fluid_density_array[i]
            fluid_density.push_back(dens)
        Profiling.logEvent("Finished search for cable nodes")
        self.thisptr.setFluidAccelerationAtNodes(fluid_acceleration)
        self.thisptr.setFluidVelocityAtNodes(fluid_velocity)
        self.thisptr.setFluidDensityAtNodes(fluid_density)
        self.updateForces()

    def updateForces(self):
        # update drag forces
        self.thisptr.updateDragForces()
        # update added mass forces
        self.thisptr.updateAddedMassForces()
        # update buoyancy forces
        # self.thisptr.updateBuoyancyForces()
        # apply forces
        self.thisptr.applyForces()

    def setFluidDensityAtNodes(self, np.ndarray density_array):
        cdef vector[double] fluid_density
        self.fluid_density_array = density_array
        cdef double dens
        for d in density_array:
            fluid_density.push_back(d)
        self.thisptr.setFluidDensityAtNodes(fluid_density)

    def setFluidVelocityAtNodes(self, np.ndarray velocity_array):
        cdef vector[ch.ChVector3d] fluid_velocity
        cdef ch.ChVector3d vel
        self.fluid_velocity_array = velocity_array
        for v in velocity_array:
            vel = ch.ChVector3d(v[0], v[1], v[2])
            fluid_velocity.push_back(vel)
        self.thisptr.setFluidVelocityAtNodes(fluid_velocity)

    def setFluidAccelerationAtNodes(self, np.ndarray acceleration_array):
        cdef vector[ch.ChVector3d] fluid_acceleration
        cdef ch.ChVector3d acc
        self.fluid_acceleration_array = acceleration_array
        for a in acceleration_array:
            acc = ch.ChVector3d(a[0], a[1], a[2])
            fluid_acceleration.push_back(acc)
        self.thisptr.setFluidAccelerationAtNodes(fluid_acceleration)


def getLocalNearestNode(coords, kdtree):
    """Finds nearest node to coordinates (local)
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

cdef pyxGetLocalNearestNode(double[:] coords,
                            double[:,:] nodeArray,
                            int[:] nodeStarOffsets,
                            int[:] nodeStarArray,
                            int node,
                            int rank):
    """Finds nearest node to coordinates (local)
    Parameters
    ----------
    coords: array_like
        coordinates from which to find nearest node
    nodeArray: array_like
        array of fluid mesh node coordinates
    nodeStarOffsets: array_like
        array of offsets from nodes (range)
    nodeStarArray: array_like
        array of neighbouring nodes
    node: int
        first guess for nearest node
    rank: int
        rank on which the nearest node is searched

    Returns
    -------
    node: int
        nearest node index
    dist: float
        distance to nearest node
    """
    # determine local nearest node distance
    cdef int nearest_node = copy.deepcopy(node)
    cdef int nearest_node0 = copy.deepcopy(node)
    cdef int nOffsets
    cdef double dist
    cdef double min_dist
    cdef double[:] node_coords
    cdef bool found_node = False
    node_coords = nodeArray[node]
    min_dist = (node_coords[0]-coords[0])*(node_coords[0]-coords[0])+\
               (node_coords[1]-coords[1])*(node_coords[1]-coords[1])+\
               (node_coords[2]-coords[2])*(node_coords[2]-coords[2])
    while found_node is False:
        nearest_node0 = nearest_node
        for nOffset in range(nodeStarOffsets[nearest_node0],
                             nodeStarOffsets[nearest_node0+1]):
            node = nodeStarArray[nOffset]
            node_coords = nodeArray[node]
            dist = (node_coords[0]-coords[0])*(node_coords[0]-coords[0])+\
                   (node_coords[1]-coords[1])*(node_coords[1]-coords[1])+\
                   (node_coords[2]-coords[2])*(node_coords[2]-coords[2])
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
        if nearest_node0 == nearest_node:
            found_node = True
    return nearest_node, dist


def getLocalElement(femSpace, coords, node):
    """Given coordinates and its nearest node, determine if it is on a
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
    patchBoundaryNodes=set()
    checkedElements=[]
    # nodeElementOffsets give the indices to get the elements sharing the node
    #log Profiling.logEvent("Getting Local Element")
    if node+1 < len(femSpace.mesh.nodeElementOffsets):
        for eOffset in range(femSpace.mesh.nodeElementOffsets[node], femSpace.mesh.nodeElementOffsets[node + 1]):
            eN = femSpace.mesh.nodeElementsArray[eOffset]
            checkedElements.append(eN)
            # union of set
            patchBoundaryNodes|=set(femSpace.mesh.elementNodesArray[eN])
            # evaluate the inverse map for element eN (global to local)
            xi = femSpace.elementMaps.getInverseValue(eN, coords)
            #J = femSpace.elementMaps.getJacobianValues(eN, )
            # query whether xi lies within the reference element
            if femSpace.elementMaps.referenceElement.onElement(xi):
                return eN
    else:
        for eOffset in range(femSpace.mesh.nodeElementOffsets[node]):
            eN = femSpace.mesh.nodeElementsArray[eOffset]
            checkedElements.append(eN)
            # union of set
            patchBoundaryNodes|=set(femSpace.mesh.elementNodesArray[eN])
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


cdef class ProtChAddedMass:
    """
    Class (hack) to attach added mass model to ProtChSystem
    This auxiliary variable is ONLY used to attach the AddedMass
    model to a ProtChSystem
    """

    def __cinit__(self,
                  system):
        self.ProtChSystem = system

    def attachModel(self, model, ar):
        """Attaches Proteus model to auxiliary variable
        """
        self.model = model
        # attaching model to ProtChSystem to access Aij
        self.ProtChSystem.model_addedmass = model
        return self

    def attachAuxiliaryVariables(self,avDict):
        pass

    def calculate_init(self):
        pass

    def calculate(self):
        pass


cpdef void attachNodeToNode(ProtChMoorings cable1, int node1, ProtChMoorings cable2, int node2):
    if cable1.beam_type == b"CableANCF":
        cppAttachNodeToNodeFEAxyzD(cable1.thisptr, node1, cable2.thisptr, node2)
    elif cable1.beam_type == b"BeamEuler":
        cppAttachNodeToNodeFEAxyzrot(cable1.thisptr, node1, cable2.thisptr, node2)


# cpdef linkBodies(ProtChBody body1,
#                  ProtChBody body2,
#                  ProtChSystem system,
#                  object coordsys,
#                  double limit_X=0.,
#                  double limit_Y=0.,
#                  double limit_Z=0.,
#                  double limit_Rx=0.,
#                  double limit_Ry=0.,
#                  double limit_Rz=0.):
#     """Create a link between 2 bodies.
#     Master body is body2.

#     Parameters:
#     -----------
#     body1: ProtChBody
#         Instance of first body
#     body2: ProtChBody
#         Instance of second body
#     body2: ProtChSystem
#         Instance of system to add link
#     coordsys: proteus.mbd.pyChronoCore.ChCoordsys
#         Coordinate system of link
#     limit_X: double
#         Limit in x direction
#     limit_Y: double
#         Limit in y direction
#     limit_Z: double
#         Limit in z direction
#     limit_Rx: double
#         Limit rotation around x axis
#     limit_Ry: double
#         Limit rotation around y axis
#     limit_Rz: double
#         Limit rotation around z axis
#     """
#     ChLinkLockBodies(body1.ChBodyAddedMass.sharedptr_chbody,
#                      body2.ChBodyAddedMass.sharedptr_chbody,
#                      system.thisptr.system,
#                      coordsys.cppobj,
#                      limit_X,
#                      limit_Y,
#                      limit_Z,
#                      limit_Rx,
#                      limit_Ry,
#                      limit_Rz)

cdef class ChBodyAddedMass:
    """Cython class for ChBodyAddedMass
    (!) Uses shared_ptr
    """

    def __cinit__(self):
        # make shared_ptr object (C++ syntax for Chrono)
        self.sharedptr = make_shared[ch.ChBodyAddedMass]()
        self.sharedptr_chbody = <shared_ptr[ch.ChBody]> self.sharedptr
        # get a raw ptr for SWIG
        self.thisptr = self.sharedptr.get()
        self.bodyptr = self.sharedptr_chbody.get()
        # create SWIG ChBody
        self.ChBodySWIG = chrono.ChBody()
        self.ChBodySWIG.this.disown()
        # delete? object pointed to by SWIG
        # point to new object (base of ChBodyAddedMass: ChBody)
        cdef SwigPyObject *swig_obj = <SwigPyObject*>self.ChBodySWIG.this
        swig_obj.ptr = <shared_ptr[ch.ChBody]*> &self.sharedptr_chbody
        # cdef shared_ptr[ch.ChSystemSMC]* pt_to_shp = <shared_ptr[ch.ChSystemSMC]*> swig_obj.ptr;
        # self.thisptr.system = pt_to_shp[0]
        # cdef SwigPyObject *swig_obj = <SwigPyObject*>self.ChBodySWIG.this
        # swig_obj.ptr = <ch.ChBody*?> &self.bodyptr

    cdef void SetMfullmass(self, ch.ChMatrixDynamic Mfullmass_in):
        self.thisptr.SetMfullmass(Mfullmass_in)

    # cdef void SetInvMfullmass(self, ch.ChMatrixDynamic inv_Mfullmass_in):
    #     self.thisptr.SetInvMfullmass(inv_Mfullmass_in)





def vec2array(vec):
    return np.array([vec.x(), vec.y(), vec.z()])

def pyvec2array(vec):
    return np.array([vec.x, vec.y, vec.z])

def mat332array(mat):
    return np.array([[mat.GetAxisX().x(), mat.GetAxisX().y(), mat.GetAxisX().z()],
                     [mat.GetAxisY().x(), mat.GetAxisY().y(), mat.GetAxisY().z()],
                     [mat.GetAxisZ().x(), mat.GetAxisZ().y(), mat.GetAxisZ().z()]])

def pymat332array(mat):
    return np.array([[mat.GetAxisX().x, mat.GetAxisX().y, mat.GetAxisX().z],
                     [mat.GetAxisY().x, mat.GetAxisY().y, mat.GetAxisY().z],
                     [mat.GetAxisZ().x, mat.GetAxisZ().y, mat.GetAxisZ().z]])

def quat2array(quat):
    return np.array([quat.e0(), quat.e1(), quat.e2(), quat.e3()])

def pyquat2array(quat):
    return np.array([quat.e0, quat.e1, quat.e2, quat.e3])
