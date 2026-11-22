from math import cos, sin, sqrt, atan2, acos, asin, pi
import csv
import os
import numpy as np
from proteus import AuxiliaryVariables, Archiver, Comm, Profiling
from proteus.Profiling import logEvent as logEvent
from collections import OrderedDict


class RigidBody(AuxiliaryVariables.AV_base, object):
    """
    Auxiliary variable used to calculate attributes of an associated shape
    class instance acting as a rigid body. To set a shape as a rigid body, use
    shape.setRigidBody(). The class instance is created automatically when
    shape.setRigidBody() has been called and after calling assembleDomain().

    Parameters
    ----------
    shape: proteus.SpatialTools.Shape_RANS
        Class instance of the shape associated to the rigid body calculations.
    cfl_target: Optional[float]
        UNUSED (to implement), sets the maximum displacement of the body
        allowed per time step.
    dt_init: float
        first time step of the simulation.
    """

    def __init__(self, shape, cfl_target=0.9, dt_init=0.001, substeps=20):
        self.Shape = shape
        self.nd = nd = shape.Domain.nd
        # if isinstance(shape, (Rectangle, Cuboid)):
        #     shape._setInertiaTensor()
        self.substeps = substeps
        self.dt_init = dt_init
        self.cfl_target = 0.9
        self.rotation_matrix = np.eye(3)
        self.h = np.array([0., 0., 0.])
        self.barycenter = np.zeros(3)
        self.i_start = None  # will be retrieved from setValues() of Domain
        self.i_end = None  # will be retrieved from setValues() of Domain
        self.It = self.Shape.It
        self.record_dict = OrderedDict()

        # variables
        self.position = np.zeros(3)
        self.last_position = np.array([0., 0., 0.])
        self.velocity = np.zeros(3, 'd')
        self.last_velocity = np.zeros(3, 'd')
        self.acceleration = np.zeros(3, 'd')
        self.last_acceleration = np.zeros(3, 'd')

        self.rotation = np.eye(3)
        self.last_rotation = np.eye(3)
        self.ang_disp = np.zeros(3, 'd')
        self.last_ang_disp = np.zeros(3, 'd')
        self.ang_vel = np.zeros(3, 'd')
        self.last_ang_vel = np.zeros(3, 'd')
        self.ang_acc = np.zeros(3, 'd')
        self.last_ang_acc = np.zeros(3, 'd')

        self.F = np.zeros(3, 'd')
        self.M = np.zeros(3, 'd')
        self.last_F = np.zeros(3, 'd')
        self.last_M = np.zeros(3, 'd')
        self.ang = 0.
        self.barycenter = self.Shape.barycenter
        self.mass = 0.

        self.pivot = np.zeros(3)
        self.last_pivot = np.zeros(3)
        self.init_barycenter = self.Shape.barycenter.copy()
        self.InputMotion = False

        # variables for checking numerical method
        self.ux = 0.0
        self.uy = 0.0
        self.uz = 0.0
        self.last_ux = 0.0
        self.last_uy = 0.0
        self.last_uz = 0.0

        # defaults, so that a body that never calls setNumericalScheme,
        # setSprings or setConstraints still runs: forward Euler, no springs
        # or dampers, all translations and rotations free
        self.scheme = 'Forward_Euler'
        self.springs = False
        self.Kx = self.Ky = self.Kz = self.Krot = 0.0
        self.Cx = self.Cy = self.Cz = self.Crot = 0.0
        self.free_x = np.ones(3)
        self.free_r = np.ones(3)

        # gravity
        if 'RigidBody' not in shape.auxiliaryVariables:
            shape._attachAuxiliaryVariable('RigidBody', self)

    def attachModel(self, model, ar):
        """
        Attaches model to auxiliary variable
        """
        self.model = model
        self.ar = ar
        self.writer = Archiver.XdmfWriter()
        self.nd = model.levelModelList[-1].nSpace_global
        m = self.model.levelModelList[-1]
        flagMax = max(m.mesh.elementBoundaryUserFlags)
        # flagMin = min(m.mesh.elementBoundaryMaterialTypes)
        self.nForces = flagMax + 1
        return self

    def calculate_init(self):
        """
        Function called automatically at the very beginning of the simulation
        by proteus.
        """
        nd = self.nd
        self.position[:] = self.Shape.barycenter.copy()
        self.last_position[:] = self.position
        self.rotation[:nd, :nd] = self.Shape.coords_system
        self.last_rotation[:nd, :nd] = self.Shape.coords_system
        self.rotation_euler = getEulerAngles(self.rotation)
        self.last_rotation_euler = getEulerAngles(self.last_rotation)

    def calculate(self):
        """
        Function called automatically at each time step by proteus.
        """
        # store previous values
        self._store_last_values()
        # for first time step
        try:
            self.dt = self.model.levelModelList[-1].dt_last
        except:
            self.dt = self.dt_init
        # get forces and moments
        self.F[:] = self.getTotalForces() * self.free_x
        self.M[:] = self.getTotalMoments() * self.free_r
        # calculate new properties with substepping
        self.step(self.dt)
        # record variables in .csv file
        if self.record_dict:
            self._recordValues()
        # print in proteus log file

        self._logTrace()

    def _store_last_values(self):
        """
        Store values of previous time step for displacement calculation
        """
        self.last_position[:] = self.position
        self.last_velocity[:] = self.velocity
        self.last_acceleration[:] = self.acceleration

        self.last_rotation[:] = self.rotation
        self.last_rotation_euler[:] = self.rotation_euler
        self.last_ang_disp[:] = self.ang_disp
        self.last_ang_vel[:] = self.ang_vel
        self.last_ang_acc[:] = self.ang_acc

        self.last_F[:] = self.F
        self.last_M[:] = self.M

        self.last_pivot = self.pivot
        self.last_ux = self.ux
        self.last_uy = self.uy
        self.last_uz = self.uz

    def getPressureForces(self):
        """
        Gives the pressure forces applied on each segments/facets of the rigid
        body
        Returns
        -------
        F_p: array_like
            pressure forces (x, y, z) as provided by Proteus
        """
        i0, i1 = self.i_start, self.i_end
        F_p = self.model.levelModelList[-1].coefficients.netForces_p[i0:i1, :]
        return F_p

    def getShearForces(self):
        """
        Gives the shear forces applied on each segments/facets of the rigid
        body
        Returns
        -------
        F_v: array_like
            shear forces (x, y, z) as provided by Proteus
        """
        i0, i1 = self.i_start, self.i_end
        F_v = self.model.levelModelList[-1].coefficients.netForces_v[i0:i1, :]
        return F_v

    def getGravityForce(self):
        """
        Returns
        -------
        Fg: array_like
            gravity force
        """
        nd = self.nd
        if nd == 2:
            Fg = self.mass * np.array([0., -9.81, 0.])
        if nd == 3:
            Fg = self.mass * np.array([0., 0., -9.81])
        return Fg

    def getMoments(self):
        """
        Gives the moments applied on each segments/facets of the rigid body
        Returns
        -------
        M: array_like
            moments (x, y, z) as provided by Proteus
        """
        i0, i1 = self.i_start, self.i_end
        M = self.model.levelModelList[-1].coefficients.netMoments[i0:i1, :]
        return M

    def getTotalMoments(self):
        """
        Gives the total moments applied the rigid body
        Returns
        -------
        M_t: array_like
            total moments (x, y, z) as provided by Proteus
        """
        M = self.getMoments()
        M_t = np.sum(M, axis=0)
        return M_t

    def getTotalForces(self):
        """
        Gives the total forces applied the rigid body: shear, pressure and
        gravity forces
        Returns
        -------
        F_t: array_like
            total forces (x, y, z) as provided by Proteus
        """
        F_p = self.getPressureForces()
        F_v = self.getShearForces()
        F_g = self.getGravityForce()
        F_t = np.sum(F_p + F_v, axis=0) + F_g
        return F_t

    def getAcceleration(self):
        """
        Returns
        -------
        a: array_like
            acceleration of current time step
        """
        a = self.F/self.mass
        return a

    def getAngularAcceleration(self):
        if sum(self.M) != 0:
            self.inertia = self.getInertia(self.M, self.Shape.barycenter)
            assert self.inertia != 0, 'Zero inertia: inertia tensor (It)' \
                                      'was not set correctly!'
            self.ang_acc = self.M[:]/self.inertia
        else:
            self.inertia = None
            self.ang_acc = np.array([0., 0., 0.])
        return self.ang_acc

    def getDisplacement(self, dt):
        # acceleration from force
        self.acceleration = self.getAcceleration()
        # substeps for smoother motion between timesteps
        dt_sub = dt/float(self.substeps)

    # Forward_Euler
        if self.scheme == 'Forward_Euler':
            for i in range(self.substeps):
                self.h[:], self.velocity[:] = forward_euler(p0=self.h, v0=self.velocity,
                                                            a=self.acceleration, dt=dt_sub)
    # Runge_Kutta
        elif self.scheme == 'Runge_Kutta':
            # settings
            Kx, Ky, Kz = self.Kx, self.Ky, self.Kz
            Cx, Cy, Cz = self.Cx, self.Cy, self.Cz
            Fx, Fy, Fz = self.F
            mass = self.mass
            # initial condition
            ux0 = self.last_position[0] - self.init_barycenter[0]      # x-axis displacement
            uy0 = self.last_position[1] - self.init_barycenter[1]      # y-axis displacement
            uz0 = self.last_position[2] - self.init_barycenter[2]      # z-axis displacement
            vx0 = self.last_velocity[0]                                # x-axis velocity
            vy0 = self.last_velocity[1]                                # y-axis velocity
            vz0 = self.last_velocity[2]                                # z-axis velocity
            ax0 = (Fx-Cx*vx0-Kx*ux0)/mass                        # x-axis acceleration
            ay0 = (Fy-Cy*vy0-Ky*uy0)/mass                        # y-axis acceleration
            az0 = (Fz-Cz*vz0-Kz*uz0)/mass                        # z-axis acceleration
            # solving numerical scheme
            ux, vx, ax = runge_kutta(u0=ux0, v0=vx0, a0=ax0, dt=dt_sub, substeps=self.substeps, F=Fx, K=Kx, C=Cx, m=mass, velCheck=False)
            uy, vy, ay = runge_kutta(u0=uy0, v0=vy0, a0=ay0, dt=dt_sub, substeps=self.substeps, F=Fy, K=Ky, C=Cy, m=mass, velCheck=False)
            uz, vz, az = runge_kutta(u0=uz0, v0=vz0, a0=az0, dt=dt_sub, substeps=self.substeps, F=Fz, K=Kz, C=Cz, m=mass, velCheck=False)
            # used for storing values of displacements through timesteps
            self.ux = ux
            self.uy = uy
            self.uz = uz
            # final values
            self.h[:] = np.array([self.ux - ux0, self.uy - uy0, self.uz - uz0])
            self.velocity = np.array([vx, vy, vz])
            self.acceleration = np.array([ax, ay, az])

        return self.h

    def getAngularDisplacement(self, dt):
        # angular acceleration from moment
        self.ang_acc = self.getAngularAcceleration()
        dt_sub = dt/float(self.substeps)

    # Forward_Euler
        if self.scheme == 'Forward_Euler':
            for i in range(self.substeps):
                # rotation
                self.ang_disp, self.ang_vel[:] = forward_euler(p0=self.ang_disp, v0=self.ang_vel,
                                                               a=self.ang_acc, dt=dt_sub)

    # Runge_Kutta
        elif self.scheme == 'Runge_Kutta':
            # the scheme integrates the rotation about z only
            if self.nd == 3 and np.any(self.free_r[:2]):
                raise NotImplementedError(
                    "RigidBody: the Runge_Kutta scheme integrates the "
                    "rotation about the z axis only, but free_r=%s frees "
                    "a rotation about x or y in 3D (it would never move). "
                    "Use setNumericalScheme('Forward_Euler')."
                    % (self.free_r,))
            # settings
            Krot = self.Krot
            Crot = self.Crot
            Fx, Fy, Fz = self.F

            # check for differenece between barycenter and pivot
            self.rp = (self.pivot - self.Shape.barycenter)
            rpx, rpy, rpz = self.rp
            Mpivot = np.array([(rpy * Fz - rpz * Fy), -(rpx * Fz - rpz * Fx), (rpx * Fy - rpy * Fx)])  # moment transformation calculated in pivot
            Mp = self.M - Mpivot                                                    # moment transformation
            self.inertia = self.getInertia(Mp, self.pivot)
            inertia = self.inertia

            # initial condition
            rz0 = atan2(self.last_rotation[0, 1], self.last_rotation[0, 0])  # angular displacement
            vrz0 = self.last_ang_vel[2]                             # angular velocity
            arz0 = (Mp[2]-Crot*vrz0-Krot*rz0)/inertia         # angular acceleration

            # solving numerical scheme
            rz, vrz, arz = runge_kutta(u0=rz0, v0=vrz0, a0=arz0, dt=dt_sub, substeps=self.substeps, F=Mp[2], K=Krot, C=Crot, m=inertia, velCheck=False)

            # final values
            self.ang_disp[2] = rz - atan2(self.last_rotation[0, 1], self.last_rotation[0, 0])
            self.ang_vel[2] = vrz
            self.ang_acc[2] = arz

        return self.ang_disp

    def setSprings(self, springs, Kx, Ky, Krot, Cx, Cy, Crot, Kz=0.0, Cz=0.0):
        """
        Sets a system of uniform springs to model soil's reactions (for moving bodies)

        Parameters
        ----------
        spring: string
            If True, spring module is switched on.
        Kx: float
            horizontal stiffness
        Ky: float
            vertical stiffness
        Krot: float
            rotational stiffness
        Cx: float
            horizontal damping parameter
        Cy: float
            vertical damping parameter
        Crot: float
            rotational damping parameter
        """
        self.springs = springs
        self.Kx = Kx
        self.Ky = Ky
        self.Kz = Kz
        self.Krot = Krot
        self.Cx = Cx
        self.Cy = Cy
        self.Cz = Cz
        self.Crot = Crot

    def setPivot(self, pivot=None):
        """
        Sets pivot centre of rotation for the angular calculation

        Parameters
        ----------
        pivot: array
        """
        self.pivot = pivot

    def setNumericalScheme(self, scheme):
        """
        Sets the numerical scheme used to solve motion.

        Parameters
        ----------
        scheme: string
            If Runge_Kutta, runge kutta scheme is applied.
            If Forward_Euler, forward euler scheme is applied.
            """
        self.scheme = scheme

    def inputMotion(self, InputMotion=False, pivot=None,
                    At=[0., 0., 0], Tt=[0., 0., 0],
                    Ar=[0., 0., 0], Tr=[0., 0., 0]):
        """
        Sets motion as an input. It's imposed rather than calculated.

        Parameters
        ----------
        InputMotion: bool
            If True, motion as input is applied.
        pivot: list
            Centre of rotation. If only translation, write barycenter's coordinates
        At: list
            Amplitude of translational motion
        Tt: list
            Period of translational motion
        Ar: list
            Amplitude of rotational motion
        Tr: list
            Period of rotational motion
            """
        self.InputMotion = InputMotion
        if pivot is None:
            self.pivot = self.Shape.barycenter
        else:
            self.pivot = np.array(pivot)
        self.At = np.array(At)
        self.Tt = np.array(Tt)
        self.Ar = np.array(Ar)
        self.Tr = np.array(Tr)

    def imposeSinusoidalMotion(self):
        """
        Motion is imposed rather than calculated.
        """
        t = self.model.stepController.t_model_last
        Tra = np.array([0., 0., 0.])
        Rot = np.array([0., 0., 0.])
        for ii in [0, 1, 2]:
            At, Tt = self.At[ii], self.Tt[ii]
            Ar, Tr = self.Ar[ii], self.Tr[ii]
            if Tt == 0.0:
                Wt = 0.0
            else:
                Wt = 2. * 3.14 / Tt

            if Tr == 0.0:
                Wr = 0.0
            else:
                Wr = 2. * 3.14 / Tr
            Dt = At * sin(Wt * t)
            Dr = Ar * sin(Wr * t)
        # motion update
            Tra[ii] = Dt - (self.last_position[ii] - self.init_barycenter[ii])
            Rot[ii] = Dr - (self.last_rotation_euler[ii])

        return Tra, Rot

    def step(self, dt):
        """
        Step for rigid body calculations in Python

        Parameters
        ----------
        dt: float
            time step
        """
        nd = self.nd
        # reinitialise displacement values
        self.h[:] = np.zeros(3)
        self.ang_disp[:] = np.zeros(3)

        # Calculate or impose motion of the rigid body
        if self.InputMotion == True:
            # sinusoidal motion imposed
            self.h[:], self.ang_disp[:] = self.imposeSinusoidalMotion()
        else:
            # Translational motion calculation
            self.h[:] = self.getDisplacement(dt)
            # Rotational motion calculation
            self.ang_disp[:] = self.getAngularDisplacement(dt)

        # translate
        self.Shape.translate(self.h[:nd])
        # rotate
        self.ang = np.linalg.norm(self.ang_disp[:])
        if nd == 2 and self.ang_vel[2] < 0:
            self.ang = -self.ang
        if self.ang != 0.:
            self.Shape.rotate(self.ang, self.ang_vel, self.Shape.barycenter)
            self.rotation[:nd, :nd] = self.Shape.coords_system
            self.rotation_matrix[:] = np.dot(np.linalg.inv(self.last_rotation),
                                             self.rotation)
            self.rotation_euler[:] = getEulerAngles(self.rotation)
        else:
            self.rotation_matrix[:] = np.eye(3)
        self.barycenter[:] = self.Shape.barycenter
        self.position[:] = self.Shape.barycenter

    def setConstraints(self, free_x, free_r):
        """
        Sets constraints on the Shape (for moving bodies)

        Parameters
        ----------
        free_x: array_like
            Translational constraints.
        free_r: array_like
            Rotational constraints.
        """
        self.free_x = np.array(free_x)
        self.free_r = np.array(free_r)

    def setMass(self, mass):
        """
        Set mass of the shape.

        Parameters
        ----------
        mass: float
            mass of the body
        """
        self.mass = float(mass)

    def setInertiaTensor(self, It):
        """
        Set the inertia tensor of the shape

        Parameters
        ----------
        It: array_like, float
            Inertia tensor of the body (3x3 array in 3D, float in 2D)
        Notes
        -----
        The inertia tensor is per unit mass: getInertia multiplies it by the
        mass of the body (set with setMass), so it must not be already scaled
        with the mass.
        """
        It = np.array(It)
        if self.nd == 2:
            assert isinstance(It, float), 'the inertia tensor of a 2D shape ' \
                'must be a float'
        if self.nd == 3:
            assert It.shape == (3, 3), 'the inertia tensor of a 3D shape ' \
                'must have a (3, 3) shape'
        self.It = It

    def getInertia(self, vec=(0., 0., 1.), pivot=None):
        """
        Gives the inertia of the shape from an axis and a pivot

        Parameters
        ----------
        vec: array_like
            Vector around which the body rotates.
        pivot: Optional[array_like]
            Pivotal point around which the body rotates. If not set, it will
            be the barycenter coordinates
        Returns
        -------
        I: float
            inertia of the mass
        Notes
        -----
        The inertia is calculated relative to the coordinate system of the
        shape (self.Shape.coords_system). If the shape was not initialised with a
        position corresponding to its inertia tensor (e.g. shape was already
        rotated when initialised), set the coordinate system accordingly
        before calling this function
        """
        assert self.It is not None, 'No inertia tensor! (' + self.Shape.name + ')'
        if pivot is None:
            pivot = self.barycenter
        # Pivot coords relative to shape centre of mass
        pivot = pivot - np.array(self.barycenter)
        # making unity vector/axis of rotation
        vec = vx, vy, vz = np.array(vec)
        length_vec = np.sqrt(vx**2 + vy**2 + vz**2)
        vec = vec/length_vec
        if self.nd == 2:
            I = self.It * self.mass
        elif self.nd == 3:
            # vector relative to original position of shape:
            vec = np.dot(vec, np.linalg.inv(self.Shape.coords_system))
            cx, cy, cz = vec
            # getting the tensor for calculaing moment of inertia
            # from arbitrary axis
            vt = np.array([[cx**2, cx * cy, cx * cz],
                           [cx * cy, cy**2, cy * cz],
                           [cx * cz, cy * cz, cz**2]])
            # total moment of inertia
            I = np.einsum('ij,ij->', self.mass * self.It, vt)
        return I

    def setRecordValues(self, filename=None, all_values=False, pos=False,
                        rot=False, ang_disp=False, F=False, M=False,
                        inertia=False, vel=False, acc=False, ang_vel=False, ang_acc=False):
        """
        Sets the rigid body attributes that are to be recorded in a csv file
        during the simulation.

        Parameters
        ----------
        filename: Optional[string]
            Name of file, if not set, the file will be named as follows:
            'record_[shape.name].csv'
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
            pos = rot = F = M = acc = vel = ang_acc = ang_vel = True
        if pos is True:
            self.record_dict['x'] = ['last_position', 0]
            self.record_dict['y'] = ['last_position', 1]
            self.record_dict['z'] = ['last_position', 2]
        if rot is True:
            self.record_dict['rx'] = ['last_rotation_euler', 0]
            self.record_dict['ry'] = ['last_rotation_euler', 1]
            self.record_dict['rz'] = ['last_rotation_euler', 2]
        if F is True:
            self.record_dict['Fx'] = ['F', 0]
            self.record_dict['Fy'] = ['F', 1]
            self.record_dict['Fz'] = ['F', 2]
            Fx = Fy = Fz = True
        if M is True:
            self.record_dict['Mx'] = ['M', 0]
            self.record_dict['My'] = ['M', 1]
            self.record_dict['Mz'] = ['M', 2]
        if acc is True:
            self.record_dict['ax'] = ['acceleration', 0]
            self.record_dict['ay'] = ['acceleration', 1]
            self.record_dict['az'] = ['acceleration', 2]
        if vel is True:
            self.record_dict['vx'] = ['velocity', 0]
            self.record_dict['vy'] = ['velocity', 1]
            self.record_dict['vz'] = ['velocity', 2]
        if ang_acc is True:
            self.record_dict['ang_ax'] = ['ang_acc', 0]
            self.record_dict['ang_ay'] = ['ang_acc', 1]
            self.record_dict['ang_az'] = ['ang_acc', 2]
        if ang_vel is True:
            self.record_dict['ang_vx'] = ['ang_vel', 0]
            self.record_dict['ang_vy'] = ['ang_vel', 1]
            self.record_dict['ang_vz'] = ['ang_vel', 2]
        if inertia is True:
            self.record_dict['inertia'] = ['inertia', None]

        if filename is None:
            self.record_filename = 'record_' + self.Shape.name + '.csv'
        else:
            self.record_filename = filename + '.csv'
        self.record_file = os.path.join(Profiling.logDir, self.record_filename)

    def _recordValues(self):
        """
        Records values of rigid body attributes at each time step in a csv file.
        """
        comm = Comm.get()
        if comm.isMaster():
            t_last = self.model.stepController.t_model_last
            dt_last = self.model.levelModelList[-1].dt_last
            t = t_last - dt_last
            values_towrite = [t]
            if t == 0:
                headers = ['t']
                for key in self.record_dict:
                    headers += [key]
                with open(self.record_file, 'w') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',')
                    writer.writerow(headers)
            for key, val in list(self.record_dict.items()):
                if val[1] is not None:
                    values_towrite += [getattr(self, val[0])[val[1]]]
                else:
                    values_towrite += [getattr(self, val[0])]
            with open(self.record_file, 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow(values_towrite)

    def _logTrace(self):
        # log values
        t_previous = self.model.stepController.t_model_last - self.dt
        t_current = self.model.stepController.t_model_last
        h = self.h
        last_pos, pos = self.last_position, self.position
        last_vel, vel = self.last_velocity, self.velocity
        rot = getEulerAngles(self.rotation)
        rot_x, rot_y, rot_z = rot[0], rot[1], rot[2]
        F = self.F
        M = self.M
        logEvent("================================================================")
        logEvent("=================== Rigid Body Calculation =====================")
        logEvent("================================================================")
        logEvent("Name: " + repr(self.Shape.name))
        logEvent("================================================================")
        logEvent("[proteus] t=%1.5fsec to t=%1.5fsec" %
                 (t_previous, t_current))
        logEvent("[proteus] dt=%1.5fsec" % (self.dt))
        logEvent("[body] ============== Pre-calculation attributes  ==============")
        logEvent("[proteus] t=%1.5fsec" % (t_previous))
        logEvent("[proteus] F=(% 12.7e, % 12.7e, % 12.7e)" % (F[0], F[1], F[2]))
        logEvent("[proteus] M=(% 12.7e, % 12.7e, % 12.7e)" % (M[0], M[1], M[2]))
        logEvent("[body]      pos=(% 12.7e, % 12.7e, % 12.7e)" %
                 (last_pos[0], last_pos[1], last_pos[2]))
        logEvent("[body]    vel=(% 12.7e, % 12.7e, % 12.7e)" %
                 (last_vel[0], last_vel[1], last_vel[2]))
        logEvent("[body] ===============Post-calculation attributes ==============")
        logEvent("[body]    t=%1.5fsec" % (t_current))
        logEvent("[body]    h=(% 12.7e, % 12.7e, % 12.7e)" % (h[0], h[1], h[2]))
        logEvent("[body]    pos=(% 12.7e, % 12.7e, % 12.7e)" %
                 (pos[0], pos[1], pos[2]))
        logEvent("[body]    vel=(% 12.7e, % 12.7e, % 12.7e)" %
                 (vel[0], vel[1], vel[2]))
        logEvent("[body]    rot=(% 12.7e, % 12.7e, % 12.7e)" %
                 (rot_x, rot_y, rot_z))
        logEvent("================================================================")


class CaissonBody(RigidBody):
    """
    Sub-class to create a caisson rigid body.
    """

    def __init__(self, shape, substeps):
        super(CaissonBody, self).__init__(shape, substeps)
        # friciton module parameter used for switching to dynamic motion cases
        self.sliding = False
        self.sliding_last = False
        self.friction = False
        self.overturning = False
        # friction and overturning parameters to be initialised
        self.pivot_friction = np.zeros(3)
        self.last_pivot_friction = np.zeros(3)
        self.Ftan = 0.0
        self.last_Ftan = 0.0
        self.Mp = np.zeros(3, 'd')
        self.last_Mp = np.zeros(3, 'd')
        self.rp = np.zeros(3, 'd')
        self.last_rp = np.zeros(3, 'd')
        self.init_barycenter = self.Shape.barycenter.copy()
        # variables for checking numerical method
        self.ux = 0.0
        self.uy = 0.0
        self.last_ux = 0.0
        self.last_uy = 0.0
        self.uxEl = 0.0
        self.last_uxEl = 0.0
        self.uxPl = 0.0
        self.last_uxPl = 0.0
        self.EL, self.last_EL, self.PL, self.last_PL = 0.0, 0.0, 0.0, 0.0

    def calculate_init(self):
        """
        Function called at the very beginning of the simulation by proteus.
        """
        nd = self.nd
        self.position[:] = self.Shape.barycenter.copy()
        self.last_position[:] = self.position
        self.rotation[:nd, :nd] = self.Shape.coords_system
        self.last_rotation[:nd, :nd] = self.Shape.coords_system
        self.rotation_euler = getEulerAngles(self.rotation)
        self.last_rotation_euler = getEulerAngles(self.last_rotation)
        # Initial position of the 2D caisson vertices
        self.cV_init = np.array([(self.Shape.vertices[0][0], self.Shape.vertices[0][1]),
                                 (self.Shape.vertices[1][0], self.Shape.vertices[1][1]),
                                 (self.Shape.vertices[2][0], self.Shape.vertices[2][1]),
                                 (self.Shape.vertices[3][0], self.Shape.vertices[3][1])])
        # Position of the 2D caisson vertices
        self.cV = np.array([(self.Shape.vertices[0][0], self.Shape.vertices[0][1]),
                            (self.Shape.vertices[1][0], self.Shape.vertices[1][1]),
                            (self.Shape.vertices[2][0], self.Shape.vertices[2][1]),
                            (self.Shape.vertices[3][0], self.Shape.vertices[3][1])])
        # Last position of the 2D caisson vertices
        self.cV_last = np.array([(self.Shape.vertices[0][0], self.Shape.vertices[0][1]),
                                 (self.Shape.vertices[1][0], self.Shape.vertices[1][1]),
                                 (self.Shape.vertices[2][0], self.Shape.vertices[2][1]),
                                 (self.Shape.vertices[3][0], self.Shape.vertices[3][1])])

    def _store_last_values(self):
        # store previous values
        self.last_position[:] = self.position
        self.last_velocity[:] = self.velocity
        self.last_acceleration[:] = self.acceleration
        self.last_rotation[:] = self.rotation
        self.last_rotation_euler[:] = self.rotation_euler
        self.last_ang_acc[:] = self.ang_acc
        self.last_ang_vel[:] = self.ang_vel
        self.last_ang_disp[:] = self.ang_disp
        self.last_F[:] = self.F
        self.last_M[:] = self.M
        # friciton and overturning
        self.cV_last[:] = self.cV
        self.sliding_last = self.sliding
        self.last_pivot_friction = self.pivot_friction
        self.last_Mp[:] = self.Mp
        self.last_rp = self.rp
        self.last_ux = self.ux
        self.last_uy = self.uy
        self.last_uxEl = self.uxEl
        self.last_uxPl = self.uxPl
        self.last_EL = self.EL
        self.last_PL = self.PL

    def step(self, dt, substeps=20):
        """
        Step for rigid body calculations in Python

        Parameters
        ----------
        dt: float
            time step
        """
        nd = self.Shape.Domain.nd
        # reinitialise displacement values
        self.ang_disp[:] = np.zeros(3)
        self.h[:] = np.zeros(3)

        # Translational motion calculation
        if self.friction != True:
            self.getDisplacement(dt)
        else:
            self.friction_module(dt)

        # Rotational motion calculation
        if self.overturning != True:
            self.getAngularDisplacement(dt)
        else:
            self.overturning_module(dt)

        # translate
        self.Shape.translate(self.h[:nd])
        # rotate
        if nd == 2:
            self.ang = self.ang_disp[2]
        else:
            self.ang = np.linalg.norm(self.ang_disp)
        if self.ang != 0.:
            if self.overturning != True:
                self.Shape.rotate(self.ang, self.ang_vel, self.Shape.barycenter)
            else:
                self.Shape.rotate(self.ang, self.ang_vel, self.pivot_friction)
            self.rotation[:nd, :nd] = self.Shape.coords_system
            self.rotation_matrix[:] = np.dot(np.linalg.inv(self.last_rotation),
                                             self.rotation)
            self.rotation_euler[:] = getEulerAngles(self.rotation)
        else:
            self.rotation_matrix[:] = np.eye(3)
        self.barycenter[:] = self.Shape.barycenter
        self.position[:] = self.Shape.barycenter

        # update vertices for friction and overturning modules
        self.cV[0] = self.Shape.vertices[0]
        self.cV[1] = self.Shape.vertices[1]
        self.cV[2] = self.Shape.vertices[2]
        self.cV[3] = self.Shape.vertices[3]

    def getInertia(self, vec=(0., 0., 1.), pivot=None):
        """
        Gives the inertia of the shape from an axis and a pivot

        Parameters
        ----------
        vec: array_like
            Vector around which the body rotates.
        pivot: Optional[array_like]
            Pivotal point around which the body rotates. If not set, it will
            be the barycenter coordinates
        Returns
        -------
        I: float
            inertia of the mass
        Notes
        -----
        The inertia is calculated relative to the coordinate system of the
        shape (self.Shape.coords_system). If the shape was not initialised with a
        position corresponding to its inertia tensor (e.g. shape was already
        rotated when initialised), set the coordinate system accordingly
        before calling this function
        """
        assert self.It is not None, 'No inertia tensor! (' + self.Shape.name + ')'
        if pivot is None:
            pivot = self.barycenter
        # Pivot coords relative to shape centre of mass
        distance = pivot - np.array(self.barycenter)
        # making unity vector/axis of rotation
        vec = vx, vy, vz = np.array(vec)
        length_vec = np.sqrt(vx**2 + vy**2 + vz**2)
        vec = vec/length_vec
        if self.Shape.Domain.nd == 2:
            L, H = self.Shape.dim
            dx, dy, dz = distance                             # To calculate It when pivot != barycenter
            Ix = float((H**2)/12.) + float(dy**2)             # To calculate x-component of It when pivot != barycenter
            Iy = float((L**2)/12.) + float(dx**2)             # To calculate y-component of It when pivot != barycenter
            self.It = Ix + Iy
            I = self.It * self.mass
        elif self.Shape.Domain.nd == 3:
            # vector relative to original position of shape:
            vec = np.dot(vec, np.linalg.inv(self.Shape.coords_system))
            cx, cy, cz = vec
            # getting the tensor for calculaing moment of inertia
            # from arbitrary axis
            vt = np.array([[cx**2, cx * cy, cx * cz],
                           [cx * cy, cy**2, cy * cz],
                           [cx * cz, cy * cz, cz**2]])
            # total moment of inertia
            I = np.einsum('ij,ij->', self.mass * self.It, vt)
        return I

    def setFriction(self, friction, m_static, m_dynamic, tolerance, grainSize):
        """
        Sets material properties for sliding and overturning modules

        Parameters
        ----------
        friction: string
            If True, friction module is switched on.
        m_static: float
            Static friction parameter (see Coulomb equation).
        m_dynamic: float
            Dynamic friction parameter.
        tolerance: float
            It's used to check if the body is in rotated state or not.
            It imposes a tolerance limit for the difference between the vertical coordinates of the 2 bottom vertices of the rigid body.
        grainSize: float
            Typical grain size of the rubble mound (if exists!) under the caisson.
            It offers an extra check of the body position.
            If the rigid body lower point position is higher than this value, it is a floating body.
        """
        self.friction = friction
        self.m_static = m_static
        self.m_dynamic = m_dynamic
        self.tolerance = tolerance
        self.grainSize = grainSize

    def setOverturning(self, overturning):
        """
        Sets material properties for sliding and overturning modules

        Parameters
        ----------
        overturning: string
            If True, overturning module is switched on.
        """
        self.overturning = overturning

    def setSprings(self, springs, Kx, Ky, Krot, Cx, Cy, Crot, Kz=0.0, Cz=0.0):
        """
        Sets a system of uniform springs to model soil's reactions (for moving bodies)

        Parameters
        ----------
        spring: string
            If True, spring module is switched on.
        Kx: float
            horizontal stiffness
        Ky: float
            vertical stiffness
        Krot: float
            rotational stiffness
        Cx: float
            horizontal damping parameter
        Cy: float
            vertical damping parameter
        Crot: float
            rotational damping parameter
        """
        self.springs = springs
        self.Kx = Kx
        self.Ky = Ky
        self.Kz = Kz
        self.Krot = Krot
        self.Cx = Cx
        self.Cy = Cy
        self.Cz = Cz
        self.Crot = Crot

    def setNumericalScheme(self, scheme):
        """
        Sets the numerical scheme used to solve motion.

        Parameters
        ----------
        scheme: string
            If Runge_Kutta, runge kutta scheme is applied.
            If Central_Difference, central difference scheme is applied.
            """
        self.scheme = scheme

    def friction_module(self, dt):
        """
        Calculate sliding motion modelling frictional force.

        Parameters
        ----------
        dt : Time step.
        """
        nd = self.Shape.Domain.nd
        substeps = 20
        dt_sub = dt/float(substeps)
        # movement_functions for friction test cases
        Fx, Fy, Fz = self.F
        eps = (10**-30)  # to avoid 0/0
        mass = self.mass
        sign_static = Fx/(abs(Fx)+eps)
        sign_dynamic = self.last_velocity[0]/(abs(self.last_velocity[0])+eps)
        if nd == 2:
            g = np.array([0., -9.81, 0.])
            Fv = Fy
            gv = g[1]
        if nd == 3:
            g = np.array([0., 0., -9.81])
            Fv = Fz
            gv = g[2]
        self.acceleration = np.zeros(3)

        #---------------------------------------------------------------
        def dynamic_case(self, sign, Fx, Fv, mass, m):
            """
            Set a dynamic friction.

            Parameters
            ----------
            sign : It's function of horizontal force.
                It's used to calculate frictional force.
            Fx : Total horizontal force from rigid body calculation (wave loading).
            Fy : Total vertical force from rigid body calculation (wave loading + weight of the body).
            mass : Mass of the rigid body.
            m : dynamic friction factor.
            """

            # Springs
            Kx = self.Kx
            Ky = self.Ky
            Cx = self.Cx
            Cy = self.Cy

            # initial condition on displacement, velocity and acceleration
            ux0 = self.last_uxEl                                                          # x-axis displacement
            uy0 = self.last_position[1] - self.init_barycenter[1]                         # y-axis displacement
            vx0 = self.last_velocity[0]                                                   # x-axis velocity
            vy0 = self.last_velocity[1]                                                   # y-axis velocity

            # calculation on the vertical direction for frictional force
            # solving numerical scheme
            ay0 = (Fv-Cy*vy0-Ky*uy0)/mass
            if self.scheme == 'Runge_Kutta':
                uy, vy, ay = runge_kutta(u0=uy0, v0=vy0, a0=ay0,
                                         dt=dt_sub, substeps=substeps,
                                         F=Fv, K=Ky, C=Cy, m=mass, velCheck=False)

            # Frictional force
            # self.PL=0.0
            # self.EL=0.0
            reactionx = -(Kx * ux0)
            reactiony = -(Ky * uy)
            Ftan = -sign * m * abs(reactiony)
            if Ftan == 0.0:
                Ftan = -sign * m * abs(Fv)

            # check on the status of the body
            if self.sliding == True:
                # caisson already experiences sliding and plastic displacements
                Kx = 0.0
                Cx = 0.0
                self.EL = 0.0
                self.PL = 1.0
                Fh = Fx + Ftan
                self.sliding = True
            elif abs(reactionx) > abs(Ftan) and (reactionx) * vx0 < 0.:
                # caisson starts to experience sliding and plastic displacements
                Kx = 0.0
                Cx = 0.0
                self.EL = 0.0
                self.PL = 1.0
                Fh = Fx + Ftan
                self.sliding = True
            else:
                # caisson experiences vibration motion and elastic displacements
                self.EL = 1.0
                self.PL = 0.0
                Fh = Fx
                self.sliding = False

            # initial condition acceleration
            # solving numerical scheme
            ax0 = (Fh-Cx*vx0-Kx*ux0)/mass
            if self.scheme == 'Runge_Kutta':
                ux, vx, ax = runge_kutta(u0=ux0, v0=vx0, a0=ax0,
                                         dt=dt_sub, substeps=substeps,
                                         F=Fh, K=Kx, C=Cx, m=mass, velCheck=True)

            # When horizontal velocity changes sign, 0-condition is passed
            # Loop must start from static case again

            if (vx0 * vx) < 0.0 and self.sliding == True:
                self.sliding = False

            # used for storing values of displacements through timesteps
            self.ux = ux
            self.uy = uy
            dx = self.ux - ux0
            dy = self.uy - uy0
            self.uxEl = dx * self.EL + self.last_uxEl   # updating elastic displacement
            self.uxPl = dx * self.PL + self.last_uxPl   # updating plastic displacement

            # final values
            self.h[0] = dx
            self.h[1] = dy
            self.velocity[0] = vx
            self.velocity[1] = vy
            self.acceleration[0] = ax
            self.acceleration[1] = ay

        #---------------------------------------------------------------

        if (Fv * gv) > 0:
            #--- Friction module, static case
            if self.sliding == False:
                sign = sign_static
                m = self.m_static
        #--- Friction module, dynamic case
            else:
                sign = sign_dynamic
                m = self.m_dynamic
            dynamic_case(self, sign=sign, Fx=Fx, Fv=Fv, mass=mass, m=m)

        if (Fv * gv) < 0:
            #--- Floating module, static case
            if self.last_velocity[0] == 0.0 or self.sliding == False:
                dynamic_case(self, sign_static, Fx, Fv, mass, m=0.0)
        #--- Floating module, dynamic case
            else:
                dynamic_case(self, sign_dynamic, Fx, Fv, mass, m=0.0)

    def overturning_module(self, dt):
        """
        Calculate overturning motion modelling soil foundation reactions.

        Parameters
        ----------
        dt : Time step.
        """
        nd = self.Shape.Domain.nd
        substeps = 20
        dt_sub = dt/float(substeps)

        #---------------------------------------------------------------
        def calculate_rotation(self, floating, h):
            """
            Calculate rotation.

            Parameters
            ----------
            floating : enable floating body calculation with NO restrictions offered by friction or springs.
            h : Caisson's vertex flag.
                The choice is between vertices at the bottom layer in contact with the soil foundation.
                It's used to check position of the caisson and spring deformation.
                h = 0 represents body in unrotated position.
                h = 1 represents body in rotated position, vertex_1 inside porous medium, positive rotation.
                h = 2 represents body in rotated position, vertex_2 inside porous medium, negative rotation.
            """

            if floating == True:
                self.barycenter = self.last_position
                self.pivot_friction = self.barycenter
            else:  # medium point in the bottom of the caisson between vertex1 and vertex2
                self.pivot_friction = np.array((0.5 * (self.cV[0][0] + self.cV[1][0]), 0.5 * (self.cV[0][1] + self.cV[1][1]), 0.0), dtype=float)

            # angular acceleration from moment
            self.rp = (self.pivot_friction - self.barycenter)
            rpx, rpy, rpz = self.rp
            Fx, Fy, Fz = self.F
            Mpivot = np.array([(rpy * Fz - rpz * Fy), -(rpx * Fz - rpz * Fx), (rpx * Fy - rpy * Fx)])  # moment transformation calculated in pivot
            Mp = self.M - Mpivot                                                    # moment transformation

            # moment equilibrium -- M = I*angAcc + C*angVel + K*angDispl
            # all these terms can be written in function of the rotation (angDispl)

            if floating == True or self.springs == False:
                Krot = 0.0
                Crot = 0.0
            else:
                Krot = self.Krot
                Crot = self.Crot

            self.inertia = self.getInertia(self.Mp, self.pivot_friction)
            assert self.inertia != 0, 'Zero inertia: inertia tensor (It)' \
                                      'was not set correctly!'
            inertia = self.inertia

            # initial condition
            rz0 = atan2(self.last_rotation[0, 1], self.last_rotation[0, 0])  # angular displacement
            vrz0 = self.last_ang_vel[2]                             # angular velocity
            arz0 = (Mp[2]-Crot*vrz0-Krot*rz0)/inertia         # angular acceleration

            # solving numerical scheme
            if self.scheme == 'Runge_Kutta':
                rz, vrz, arz = runge_kutta(u0=rz0, v0=vrz0, a0=arz0, dt=dt_sub, substeps=substeps, F=Mp[2], K=Krot, C=Crot, m=inertia, velCheck=False)

            # final values
            self.ang_disp[2] = rz - atan2(self.last_rotation[0, 1], self.last_rotation[0, 0])
            self.ang_vel[2] = vrz
            self.ang_acc[2] = arz

        #---------------------------------------------------------------

        # Check position and then call calculate rotation
        # h1 and h2 in the same horizontal line, flat
        if abs(self.cV_last[0][1] - self.cV_last[1][1]) < self.tolerance:

            # caisson inside rubble mound
            if self.cV_last[0][1] < self.cV_init[0][1]:
                transl_x = 0.0
                transl_y = self.cV_init[0][1] - self.cV_last[0][1]
                transl_z = 0.0
                transl = (transl_x, transl_y)
                self.Shape.translate(transl)
                calculate_rotation(self, floating=False, h=0)     # ----> caisson under the rubble mound: I move it again on the mound!!!

            # caisson on the rubble mound (still in contact)
            elif self.cV_last[0][1] == self.cV_init[0][1] or (self.cV_last[0][1] - self.cV_init[0][1]) < self.grainSize:
                calculate_rotation(self, floating=False, h=0)     # ----> caisson on the rubble mound: I calculate normally rotation

            # floating caisson (no contact)
            else:
                # ----> caisson up the rubble mound: FLOATING CASE I calculate normally rotation on barycenter!!!
                calculate_rotation(self, floating=True, h=0)

        # h1 and h2 NOT in a horizontal line, NOT flat
        else:
            if self.cV_last[1][1] < self.cV_last[0][1]:
                if self.cV_last[1][1] == self.cV_init[1][1] or (self.cV_last[1][1] - self.cV_init[1][1]) < self.grainSize:
                    calculate_rotation(self, floating=False, h=2)        # ----> caisson on the rubble mound: I calculate normally rotation
                else:
                    # ----> caisson up the rubble mound: FLOATING CASE I calculate normally rotation on barycenter!!!
                    calculate_rotation(self, floating=True, h=0)

            elif self.cV_last[0][1] < self.cV_last[1][1]:
                if self.cV_last[0][1] == self.cV_init[0][1] or (self.cV_last[0][1] - self.cV_init[0][1]) < self.grainSize:
                    calculate_rotation(self, floating=False, h=1)        # ----> caisson on the rubble mound: I calculate normally rotation
                else:
                    # ----> caisson up the rubble mound: FLOATING CASE I calculate normally rotation on barycenter!!!
                    calculate_rotation(self, floating=True, h=0)

    def setRecordValues(self, filename=None, all_values=False, pos=False,
                        rot=False, ang_disp=False, F=False, M=False,
                        inertia=False, vel=False, acc=False, ang_vel=False, ang_acc=False, elasticPlastic=False):
        """
        Sets the rigid body attributes that are to be recorded in a csv file
        during the simulation.

        Parameters
        ----------
        filename: Optional[string]
            Name of file, if not set, the file will be named as follows:
            'record_[shape.name].csv'
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
            pos = rot = F = M = acc = vel = ang_acc = ang_vel = elasticPlastic = True
        if pos is True:
            self.record_dict['x'] = ['last_position', 0]
            self.record_dict['y'] = ['last_position', 1]
            self.record_dict['z'] = ['last_position', 2]
        if rot is True:
            self.record_dict['rx'] = ['last_rotation_euler', 0]
            self.record_dict['ry'] = ['last_rotation_euler', 1]
            self.record_dict['rz'] = ['last_rotation_euler', 2]
        if F is True:
            self.record_dict['Fx'] = ['F', 0]
            self.record_dict['Fy'] = ['F', 1]
            self.record_dict['Fz'] = ['F', 2]
            Fx = Fy = Fz = True
        if M is True:
            self.record_dict['Mx'] = ['M', 0]
            self.record_dict['My'] = ['M', 1]
            self.record_dict['Mz'] = ['M', 2]
        if acc is True:
            self.record_dict['ax'] = ['acceleration', 0]
            self.record_dict['ay'] = ['acceleration', 1]
            self.record_dict['az'] = ['acceleration', 2]
        if vel is True:
            self.record_dict['vx'] = ['velocity', 0]
            self.record_dict['vy'] = ['velocity', 1]
            self.record_dict['vz'] = ['velocity', 2]
        if ang_acc is True:
            self.record_dict['ang_ax'] = ['ang_acc', 0]
            self.record_dict['ang_ay'] = ['ang_acc', 1]
            self.record_dict['ang_az'] = ['ang_acc', 2]
        if ang_vel is True:
            self.record_dict['ang_vx'] = ['ang_vel', 0]
            self.record_dict['ang_vy'] = ['ang_vel', 1]
            self.record_dict['ang_vz'] = ['ang_vel', 2]
        if inertia is True:
            self.record_dict['inertia'] = ['inertia', None]
        if elasticPlastic is True:
            self.record_dict['uxElastic'] = ['uxEl', None]
            self.record_dict['uxPlastic'] = ['uxPl', None]
            self.record_dict['uy'] = ['uy', None]

        if filename is None:
            self.record_filename = 'record_' + self.Shape.name + '.csv'
        else:
            self.record_filename = filename + '.csv'
        self.record_file = os.path.join(Profiling.logDir, self.record_filename)


class PaddleBody(RigidBody):
    """
    Sub-class to create a PADDLE rigid body.
    """

    def __init__(self, shape, substeps):
        super(PaddleBody, self).__init__(shape, substeps)
        self.pivot = np.zeros(3)
        self.last_pivot = np.zeros(3)
        self.Shape.barycenter = np.array([(self.Shape.vertices[0][0] + self.Shape.vertices[1][0]) * 0.5, 0.0, 0.0])
        self.init_barycenter = self.Shape.barycenter.copy()
        # variables for checking numerical method
        self.ux = 0.0
        self.uy = 0.0
        self.last_ux = 0.0
        self.last_uy = 0.0

    def calculate_init(self):
        """
        Function called at the very beginning of the simulation by proteus.
        """
        nd = self.nd
        self.position[:] = self.Shape.barycenter.copy()
        self.last_position[:] = self.position
        self.rotation[:nd, :nd] = self.Shape.coords_system
        self.last_rotation[:nd, :nd] = self.Shape.coords_system
        self.rotation_euler = getEulerAngles(self.rotation)
        self.last_rotation_euler = getEulerAngles(self.last_rotation)
        # Initial position of the 2D caisson vertices
        self.cV_init = np.array([(self.Shape.vertices[0][0], self.Shape.vertices[0][1]),
                                 (self.Shape.vertices[1][0], self.Shape.vertices[1][1]),
                                 (self.Shape.vertices[2][0], self.Shape.vertices[2][1]),
                                 (self.Shape.vertices[3][0], self.Shape.vertices[3][1])])
        # Position of the 2D caisson vertices
        self.cV = np.array([(self.Shape.vertices[0][0], self.Shape.vertices[0][1]),
                            (self.Shape.vertices[1][0], self.Shape.vertices[1][1]),
                            (self.Shape.vertices[2][0], self.Shape.vertices[2][1]),
                            (self.Shape.vertices[3][0], self.Shape.vertices[3][1])])
        # Last position of the 2D caisson vertices
        self.cV_last = np.array([(self.Shape.vertices[0][0], self.Shape.vertices[0][1]),
                                 (self.Shape.vertices[1][0], self.Shape.vertices[1][1]),
                                 (self.Shape.vertices[2][0], self.Shape.vertices[2][1]),
                                 (self.Shape.vertices[3][0], self.Shape.vertices[3][1])])

    def _store_last_values(self):
        # store previous values
        self.last_position[:] = self.position
        self.last_velocity[:] = self.velocity
        self.last_acceleration[:] = self.acceleration
        self.last_rotation[:] = self.rotation
        self.last_rotation_euler[:] = self.rotation_euler
        self.last_ang_acc[:] = self.ang_acc
        self.last_ang_vel[:] = self.ang_vel
        self.last_ang_disp[:] = self.ang_disp
        self.last_F[:] = self.F
        self.last_M[:] = self.M
        # friciton and overturning
        self.cV_last[:] = self.cV
        self.last_pivot = self.pivot
        self.last_ux = self.ux
        self.last_uy = self.uy

    def step(self, dt, substeps=20):
        """
        Step for rigid body calculations in Python

        Parameters
        ----------
        dt: float
            time step
        """
        nd = self.Shape.Domain.nd
        # reinitialise displacement values
        self.ang_disp[:] = np.zeros(3)
        self.h[:] = np.zeros(3)

        # Calculate or impose motion of the rigid body
        if self.InputMotion == True:
            # sinusoidal motion imposed
            self.h[:], self.ang_disp[:] = self.imposeSinusoidalMotion()
        else:
            # Translational motion calculation
            self.h[:] = self.getDisplacement(dt)
            # Rotational motion calculation
            self.ang_disp[:] = self.getAngularDisplacement(dt)

        # translate
        self.Shape.translate(self.h[:nd])
        # rotate
        if nd == 2:
            self.ang = self.ang_disp[2]
        else:
            self.ang = np.linalg.norm(self.ang_disp)
        if self.ang != 0.:
            self.Shape.rotate(self.ang, self.ang_vel, self.pivot)
            self.rotation[:nd, :nd] = self.Shape.coords_system
            self.rotation_matrix[:] = np.dot(np.linalg.inv(self.last_rotation),
                                             self.rotation)
            self.rotation_euler[:] = getEulerAngles(self.rotation)
        else:
            self.rotation_matrix[:] = np.eye(3)
        self.barycenter[:] = self.Shape.barycenter
        self.position[:] = self.Shape.barycenter

    def inputMotion(self, InputMotion=False, pivot=None,
                    At=[0., 0., 0], Tt=[0., 0., 0],
                    Ar=[0., 0., 0], Tr=[0., 0., 0],
                    rampStart=0, rampEnd =0, Tend = 1e6):
        """
        Sets motion as an input. It's imposed rather than calculated.

        Parameters
        ----------
        InputMotion: bool
            If True, motion as input is applied.
        pivot: list
            Centre of rotation. If only translation, write barycenter's coordinates
        At: list
            Amplitude of translational motion
        Tt: list
            Period of translational motion
        Ar: list
            Amplitude of rotational motion
        Tr: list
            Period of rotational motion
        rampStart: float
            Time for ramping waves at the beginning
        rampStart: float
            Time for ramping waves at end
        Tend: float 
            End time of paddle operation (needed for rampEnd)
            
            """
        self.InputMotion = InputMotion
        if pivot is None:
            self.pivot = self.Shape.barycenter
        else:
            self.pivot = np.array(pivot)
            self.Shape.barycenter = self.pivot
            self.init_barycenter = self.Shape.barycenter.copy()

        self.At = np.array(At)
        self.Tt = np.array(Tt)
        self.Ar = np.array(Ar)
        self.Tr = np.array(Tr)
        self.rampS = rampStart
        self.rampE = rampEnd
        self.Tend = Tend
    def imposeSinusoidalMotion(self, tt = None):
        """
        Motion is imposed rather than calculated.

        Parameters
        ----------
        tt: None or float
            tt should be None for other than testing purposes. If not None,
            time is externally set.
        """
        if tt is not None:
            t= tt
        else:
            t = self.model.stepController.t_model_last


        rS = 1.
        rE = 1.
        if self.rampS > 0:
            rS = min(t/self.rampS , 1.)
        if self.rampE > 0:
            rE = min((self.Tend - t)/(self.rampE) , 1.)
            rE = max(0.,rE)

        rr = rE*rS
        Tra = np.array([0., 0., 0.])
        Rot = np.array([0., 0., 0.])
        for ii in [0, 1, 2]:
            At, Tt = self.At[ii], self.Tt[ii]
            Ar, Tr = self.Ar[ii], self.Tr[ii]
            if Tt == 0.0:
                Wt = 0.0
            else:
                Wt = 2. * pi / Tt

            if Tr == 0.0:
                Wr = 0.0
            else:
                Wr = 2. * pi / Tr
            Dt = rr*At * sin(Wt * t)
            Dr = rr*Ar * sin(Wr * t)
        # motion update
            Tra[ii] = Dt - (self.last_position[ii] - self.init_barycenter[ii])
            Rot[ii] = Dr - (self.last_rotation_euler[ii])

        return Tra, Rot


def forward_euler(p0, v0, a, dt):
    v1 = v0 + a * dt
    p1 = p0 + v1 * dt
    return p1, v1


# Need to be tested.
# def leapfrog(p0, v0, a, dt):
#    p1 = p0+v0*dt+0.5*a*dt**2
#    v1 = v0+a*dt
#    return p1, v1


def getEulerAngles(coord_system):
    rot = coord_system
    rx = atan2(rot[1, 2], rot[2, 2])
    ry = -asin(rot[0, 2])
    rz = atan2(rot[0, 1], rot[0, 0])
    return [rx, ry, rz]


def runge_kutta(u0, v0, a0, dt, substeps, F, K, C, m, velCheck):
    """
    Function that applies Runge Kutta's scheme for motion calculation.

    Parameters
    ----------
    u0 : translational or rotational displacement.
    v0 : translational or rotational velocity.
    a0 : translational or rotational acceleration.
    dt : Time step.
    substeps : integer number of substeps.
    F : translational or rotational loading.
    K : translational or rotational stiffness.
    C : translational or rotational damping factor.
    m : mass (translational calculation) or inertia (rotational calculation).
    velCheck : check on translational velocity sign (friction module only!).
    """
    for ii in range(substeps):
        # 1 step
        u1 = u0
        v1 = v0
        a1 = a0
    # 2 step
        u2 = u1 + v1 * dt / 2.
        v2 = v1 + a1 * dt / 2.
        a2 = (F-C*v2-K*u2)/m
    # 3 step
        u3 = u1 + v2 * dt / 2.
        v3 = v1 + a2 * dt / 2.
        a3 = (F-C*v3-K*u3)/m
    # 4 step
        u4 = u1 + v3 * dt
        v4 = v1 + a3 * dt
        a4 = (F-C*v4-K*u4)/m
    # calculation
        u = u0 + (dt/6.) * (v1 + 2. * v2 + 2. * v3 + v4)
        v = v0 + (dt/6.) * (a1 + 2. * a2 + 2. * a3 + a4)
        a = (F-C*v-K*u)/m
    # velocity check
        if velCheck == True:
            # When velocity changes sign, it means that 0-condition is passed
            # Loop must start from static case again
            if (v0 * v) < 0.0:
                break
    # updating values for next substep
        u0 = u
        v0 = v
        a0 = a
    return u, v, a