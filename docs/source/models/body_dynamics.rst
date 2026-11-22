.. _body_dynamics:

Multi-Body Solid Dynamics
*************************

Implementation
==============

The module described in this page can be imported as follows:

.. code-block:: python

   from proteus.mbd import CouplingFSI as fsi

Proteus uses wrappers and modified/derived classes from the Chrono engine, an open-source multibody dynamics library available at https://github.com/projectchrono/chrono.

The bodies and cables classes described below can interact with proteus models
such as Navier-Stokes to retrieve forces and moments, moving (ALE) mesh for
moving the domain with the structure, and added-mass model.

.. todo::

   Maybe replace cython declaration for Chrono headers with the Chrono
   pyEngine, but only if there is still flexibility such as access to C++
   pointer from the python implementation of Chrono.


Classes
=======


ProtChSystem
------------

The `ProtChSystem` class has a pointer to a Chrono `ChSystem`, and holds the
general options for the Chrono simulation, such as time step size, gravity,
etc. All the physical bodies described must be associated to a `ProtChSystem`
instance.


.. code-block:: python

   import pychono
   from proteus.mbd import CouplingFSI as fsi

   my_system = fsi.ProtChSystem()
   g = np.array([0., 0., -9.81])
   my_system.setGravitationalAcceleration(g)
   my_system.setTimeStep(0.001)  # the time step for Chrono calculations
   my_chsystem = my_system.getChronoObject()  # access chrono object

.. important::

   The `ProtChSystem` instance itself must be added to the `auxiliaryVariables`
   list of the Navier-Stokes model in order to calculate and retrieve the fluid
   forces from the fluid pressure field provided by Proteus at the boundaries
   of the different bodies.


ProtChBody
----------

Class for creating a rigid body. It has a Chrono `ChBody` body variable
(`ProtChBody.ChBody`) accessible within python with some of the
functionalities/functions of Chrono `ChBody`. It must be associated to a
`ProtChSystem` instance in order to be included in the multibody dynamics
simulation. This can be done with the passing of the `system` argument as the
`ProtChBody` instance is created (see example below). Otherwise, the function
`ProtChSystem.addProtChBody(my_body)` can be called separately.

.. code-block:: python

   my_body = fsi.ProtChBody(system=my_system)
   my_body.attachShape(my_shape)  # sets everything automatically
   my_body.setRecordValues(all_values=True) # record everything
   my_chbody = my_body.getChronoObject()  # access chrono object


When set up properly and running with a Proteus Navier-Stokes simulation, the
fluid pressure will be applied on the boundaries of the rigid body. The ChBody
will be moved accordingly, as well as its boundaries (supposing that a moving
mesh or immersed boundaries are used).

.. attention::

   The `ProtChBody.ChBody`  variable accessible with
   `ProtChBody.getChronoObject()` is actually using a derived class from the
   base Chrono `ChBody` in order to add the possibility of using an added-mass
   matrix (see `ChBodyAddedMass` in proteus.mbd.ChRigidBody.h).

ProtChMesh
----------

This class creates a `ChMesh` that is needed to create moorings.

.. code-block:: python

   my_mesh = fsi.ProtChMesh(system=my_system)
   my_chmesh = my_mesh.getChronoObject()


ProtChMoorings
--------------

This class is for easily creating cables. The following properties must be
known in order to instantiate a `ProtChMoorings`: `ProtChSystem` instance,
`Mesh` instance, `length` for the length of the cable/segment, `nb_elems` for
the number of elements along the cable/segment, `d` for the diameter of the
cable/segment, `rho` for the density of the cable/segment, `E` for the Young
modulus of the cable/segment.

.. code-block:: python

   my_mooring = fsi.ProtChMoorings(system=my_system,
                                   mesh=my_mesh,
                                   length=np.array([10.]),
                                   nb_elems=np.array([10], dtype=np.int32),
                                   d=np.array([0.01]),
                                   rho=np.array([300.2]),
                                   E=np.array([1e9]))
   # set function to place the nodes along cable ('s' is the position along the 1D cable)
   fpos = lambda s: np.array([s, 1., 0.])  # position along cable
   ftan = lambda s: np.array([1., 0., 0.])  # tangent of cable along cable
   my_mooring.setNodesPositionFunction(fpos, ftan)
   # set the nodes position from the function
   my_mooring.setNodesPosition()
   # build nodes (automatic with fpos/ftan)
   # nodes are equally spaced according to the number of elements (nb_elems)
   my_mooring.buildNodes()
   # add a body as fairlead
   my_mooring.attachBackNodeToBody(my_body)
   # fix front node as anchor
   my_mooring.fixFrontNode(True)

Setting the position function is useful when a relatively complex layout of the
cable is desired, such as a catenary shape.


.. note::

   The reason for the array structure for the `length`, `nb_elems`, `d`, `rho`,
   and `E` parameters is that a cable can be multi-segmented (different
   sections of the same cable having different material properties).


ProtChAddedMass
---------------

A class to deal with the added mass model from proteus.kernels.AddedMass. This
class should not be instantiated manually and will be automatically
instantiating as a variable of `ProtChSystem` (accessible as
`my_system.ProtChAddedMass`). It is used to build the added mass matrix for the
rigid bodies.

.. important::

   This class instance must be passed to the `AddedMass` model
   `auxiliaryVariables` to have any effect
   (`auxiliaryVariables.append(my_system.ProtChAddedMass`)


Postprocessing Tools
====================

ProtChBody
----------

The data related to mooring cables is saved in an csv file, usually
``[my_body.name].csv``. Additionally, if the added mass model was used, the
values of the added mass matrix are available in ``[my_body.name]_Aij_.csv``

ProtChMoorings
--------------

The data related to mooring cables is saved in an hdf5 file, usually
``[my_mooring.name].h5``, which can be read directly with h5py. Another way to
read and visualise the data is to use the associated ``[my_mooring.name].xmf``.
That file only lists the most recent time step, so first run ``gatherTimes``
(installed with Proteus) to collect every time step stored in the ``.h5`` file:

.. code-block::

   gatherTimes [my_mooring.name]

This creates ``[my_mooring.name]_complete.xmf``, which can be opened in
Paraview to navigate the time steps that have been recorded. The argument may
be given with or without the ``.xmf`` extension, and ``-t N`` keeps only the
first ``N`` time steps. Running it on an ``.xmf`` that already lists every
stored time step leaves the time steps unchanged.
