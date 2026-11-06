.. Proteus documentation master file. This file generates the Proteus homepage (index.html) and serves as the root of the toctree

.. _intro-sec:

Introduction
============

Proteus is a Python package for rapidly developing computer models and
numerical methods. It is focused on models of continuum mechanical
processes described by partial differential equations and on
discretizations and solvers for computing approximate solutions to
these equations. Proteus consists of a collection of Python modules
and scripts. Proteus also uses several C, C++, and Fortran libraries,
which are either external open source packages or part of Proteus, and
several open source Python packages.

The design of Proteus is organized around two goals:

* Make it easy to solve new model equations with existing numerical methods
* Make it easy to solve existing model equations with new numerical methods

We want to improve the development process for models *and*
methods. Proteus is not intended to be an expert system for solving
partial differential equations. In fact, effective numerical methods
are often physics-based. Nevertheless many physical models are
mathematically represented by the same small set of differential
operators, and effective numerical methods can be developed with minor
adjustments to existing methods. The problem with much existing
software is that the physics and numerics are completely intertwined,
which makes it difficult to extend (and maintain). In Proteus the
description of the physical model and initial-boundary value problems
are nearly "method agnostic".  This approach has been used in the
developement of a variety of mathematical models and numerical
methods, both of which are described in more detail below
(:ref:`capabilities-sec`).

.. _obtaining-sec:

Obtaining and Installing Proteus
================================

For learning and experimenting there is a
`Docker image <https://cloud.docker.com/u/erdc/repository/docker/erdc/proteus>`_.
Proteus can be installed through conda with:::

  % conda install proteus -c conda-forge

Proteus is available as source from our public
`GitHub <https://github.com/proteustoolkit/proteus>`_
repository. For a development installation, the installation of
dependencies and the compilation of Proteus from source is done with:::

  % git clone https://github.com/proteustoolkit/proteus
  % cd proteus
  % conda env create -f environment-dev.yml
  % conda activate proteus-dev
  % pip install -v -e .

Alternatively, if you already have compilers (C,C++, and Fortran!)
installed on your system, you can install Proteus through hashdist
with the following commands:::

  % git clone https://github.com/proteustoolkit/proteus
  % cd proteus
  % make develop
  % make test

More information is available on our `Wiki <https://github.com/proteustoolkit/proteus/wiki>`_, and you can ask for help on
the `Developers' Mailing List <https://groups.google.com/forum/#!forum/proteus-dev>`_.

.. _running-sec:

Running
=======

If you have successfully compiled and tested Proteus then you should be able to do::

   % cd $PROTEUS/test/ci
   % $PROTEUS_PREFIX/bin/parun poisson_3d_p.py poisson_3d_c0p1_n.py

The solution will be saved in an HDF5 file ending in .h5, described by
a YMF document ending in .ymf. To open it in ParaView or VisIt, write an
XDMF view of it first::

   % ymf2xmf poisson_3d_c0p1pe1.ymf

and open the resulting .xmf.

Threads and MPI
---------------

Run under MPI with more than one rank, Proteus gives each rank one thread: it sets
``OMP_NUM_THREADS``, ``OPENBLAS_NUM_THREADS``, ``MKL_NUM_THREADS`` and ``VECLIB_MAXIMUM_THREADS``
to 1 (``proteus/_threads.py``), because one rank per core is how a node is filled and
OpenBLAS would otherwise start a thread per core in every rank. A value you set yourself
is kept, so ``OMP_NUM_THREADS=4 mpiexec -n 8 parun ...`` runs eight ranks of four threads.
A serial run is not touched and uses whatever threads the libraries choose. The variables
are read when numpy is imported: import ``proteus`` before numpy, or install
``threadpoolctl``, which lets Proteus limit libraries that are already loaded.

.. _capabilities-sec:

Capabilities
============

Test problems and some analytical solutions have been implemented for

* Poisson's equation
* The heat equation
* Linear advection-diffusion-reaction equations, including problems with interfaces that do not follow the mesh (CutFEM for embedded boundaries, immersed finite elements for immersed interfaces, and the hybrid CutFEM-IFEM method; see :ref:`adr`)
* Singly degenerate nonlinear advection-diffusion-reaction equations (including various forms of Burger's equation)
* Doubly degenerate nonlinear advection-diffusion-reaction equations
* The eikonal (signed distance) equation
* The diffusive wave equations for overland flow
* 1D and 2D Shallow Water Equations
* 2D Dispersive Shallow Water Equations
* Richards' equation (mass conservative head- and saturation-based)
* Two-phase flow in porous media with diffuse interface (fully coupled  and IMPES formulations)
* Two-phase flow in porous media with a sharp interface (level set formulation)
* Stokes equations
* Navier-Stokes equations
* Reynolds-Averged Navier-Stokes equations
* Two-phase Stokes/Navier-Stokes/RANS flow with a sharp interface (level set/VOF formulation)
* Linear elasticity

These problems are solved on unstructured simplicial meshes. Simple
meshes can be generated with tools included with Proteus, and more
complex meshes can by imported from other mesh generators. The finite
elements implemented are

Classical methods with various types of stabilization (entropy viscosity, variational multiscale, and algebraic methods)

* :math:`C_0 P_1`
* :math:`C_0 P_2`
* :math:`C_0 Q_1`
* :math:`C_0 Q_2`

Discontinuous Galerkin methods

* :math:`C_{-1} P_0`
* :math:`C_{-1} P_1`  (Lagrange Basis)
* :math:`C_{-1} P_2`  (Lagrange Basis)
* :math:`C_{-1} P_k`  (Monomial Basis)

Non-conforming and mixed methods

* :math:`P_1` non-conforming
* :math:`C_0 P_1 C_0 P_2` Taylor-Hood

The time integration methods are

* Backward Euler
* Forward Euler
* :math:`\Theta` Methods
* Strong Stability Preserving Runge-Kutta Methods
* Adaptive BDF Methods
* Pseudo-transient continuation

The linear solvers are

* Jacobi
* Gauss-Seidel
* Alternating Schwarz
* Full Multigrid
* Wrappers to LAPACK, SuperLU, and PETSc

The nonlinear solvers are

* Jacobi
* Gauss-Seidel
* Alternating Schwarz
* Newton's method
* Nonlinear Multigrid (Full Approximation Scheme)
* Fast Marching and Fast Sweeping

Additional tools are included for pre- and post-processings meshes and
solutions files generated by Proteus and other models, including methods for
obtaining locally-conservative velocity fields from :math:`C_0` finite
elements.

.. _release-sec:

Release Policy
==============

The releases are numbered major.minor.revision

* A revision increment indicates a bug fix or extension that shouldn't break any models working with the same major.minor number.
* A minor increment introduces significant new functionality but is mostly backward compatible
* A major increment may require changes to input files and significantly change the underlying Proteus implementation.

These are not hard and fast rules, and there is no time table for releases.

References
==========

* Hybrid CutFEM-IFEM Model for Embedded and Immersed Interfaces with
  Triply-Cut Cells for the 2D Poisson Problem (2026) D. Nathawani and
  C.E. Kees, submitted.
* `An Unstructured Finite Element Model for Incompressible Two-Phase
  Flow Based on a Monolithic Conservative Level Set Method
  <https://doi.org/10.1002/fld.4817>`_
  (2020) M. Quezada de Luna, J. H. Collins, and C.E. Kees,
  *International Journal for Numerical Methods in Fluids*.
* `Robust explicit relaxation technique for solving the Green-Naghdi equations
  <https://doi.org/10.1016/j.jcp.2019.108917>`_ (2019) J.-L. Guermond,
  B. Popov, E. Tovar, C.E. Kees, *Journal of Computational Physics*
* Preconditioners for Two-Phase Incompressible Navier-Stokes
  Flow (2019) N. Bootland, C.E. Kees, A. Wathen,
  A. Bentley *SIAM Journal on Scientific Computing*, In Press.
* `Modeling Sediment Transport in Three-Phase Surface Water Systems
  <https://doi.org/10.1080/00221686.2019.1581673>`_ (2019)
  C.T. Miller, W.G. Gray, C.E. Kees, I.V. Rybak, B.J. Shepherd,
  *Journal of Hydraulic Engineering*
* `Fast Random Wave Generation in Numerical Tanks
  <https://doi.org/10.1680/jencm.17.00016>`_ (2019) A. Dimakopoulos, T. de
  Lataillade, C.E. Kees, *Proceedings of the Institution of Civil
  Engineers - Engineering and Computational Mechanics*, 1-29.
* `A Partition of Unity Approach to Adaptivity and Limiting in
  Continuous Finite Element Methods
  <https://doi.org/10.1016/j.camwa.2019.03.021>`_ (2019) D. Kuzmin, M. Quezada
  de Luna, C.E. Kees, *Computers and Mathematics with Applications*.
* `Simulating Oscillatory and Sliding Displacements of Caisson
  Breakwaters Using a Coupled Approach
  <https://doi.org/10.1061/(ASCE)WW.1943-5460.0000504>`_ (2019) G. Cozzuto,
  A. Dimakopoulos, T. de Lataillade, P.O. Morillas, and C.E. Kees,
  *Journal of Waterway, Port, Coastal, and Ocean Engineering*.
* `A Monolithic Conservative Level Set Method with Built-In
  Redistancing
  <https://doi.org/10.1016/j.jcp.2018.11.044>`_ (2019) M. Quezada de
  Luna, D. Kuzmin, C.E. Kees, *Journal of Computational Physics*, 379,
  262-278.
* `Computational Model for Wave Attenuation by Flexible Vegetation
  <https://doi.org/10.1061/(ASCE)WW.1943-5460.0000487>`_ (2018)
  S.A. Mattis, C.E. Kees, M.V. Wei, A. Dimakopoulos, and C.N. Dawson,
  *Journal of Waterway, Port, Coastal, and Ocean Engineering* 145(1),
  p.04018033.
* `Well-Balanced Second-Order Finite Element Approximation of the
  Shallow Water Equations with Friction
  <https://doi.org/10.1137/17M1156162>`_ (2018) J.L. Guermond, M.Q. de
  Luna, B. Popov, C.E. Kees, and M.W. Farthing *SIAM Journal on
  Scientific Computing* 40(6), A3873-A3901.
* `Dual-Scale Galerkin Methods for Darcy Flow
  <https://doi.org/10.1016/j.jcp.2017.10.047>`_ (2018) G. Wang, G. Scovazzi, L. Nouveau,
  C.E. Kees, Simone Rossi, O. Colomes, and A. Main (2018) *Journal of
  Computational Physics* 354, 111-134.
* `Implementation details of the level set two-phase Navier-Stokes
  equations in Proteus
  <https://www.clemson.edu/science/departments/math-stat/documents/technical-reports/TR2017_10_ab.nb.aw.ck.pdf>`_ (2017) A. Bentley, N. Bootland, A. Wathen, C. Kees,
  *Technical-Report-TR2017-10-ab.nb.aw.ck*.
* `Evaluation of Galerkin and Petrov-Galerkin Model Reduction for
  Finite Element Approximations of the Shallow Water Equations
  <https://doi.org/10.1016/j.cma.2017.01.027>`_ (2017) A. Lozovsky, M. W. Farthing,
  and C.E. Kees, *Computational Methods in Applied Mechanics and
  Engineering* 318, 537-571.
* `POD-Based Model Reduction for Stabilized Finite Element
  Approximations of Shallow Water Flows
  <https://doi.org/10.1016/j.cam.2016.01.029>`_ (2016) A. Lozovskiy,
  M.W. Farthing, C.E. Kees, E. Gildin *Journal of Computational and
  Applied Mathematics*, 302, 50-70.
* `An Immersed Structure Approach for Fluid-Vegetation Interaction
  <https://doi.org/10.1016/j.advwatres.2015.02.014>`_ (2015)
  S.A. Mattis, C.N. Dawson, C.E. Kees, M.W. Farthing, *Advances in
  Water Resources*, 80,1-16.
* `Finite Element Methods for Variable Density Flow And Solute
  Transport <https://doi.org/10.1007/s10596-012-9330-2>`_ (2013)
  T.J. Povich, C.N. Dawson, M.W. Farthing, C.E. Kees *Computational
  Geosciences* 17(3), 529-549.
* `Numerical simulation of water resources problems: Models, methods,
  and trends
  <https://doi.org/10.1016/j.advwatres.2012.05.008>`_ (2013)
  Cass T. Miller, Clint N. Dawson, Matthew W. Farthing, Thomas Y. Hou,
  Jingfang Huang, Christopher E. Kees, C.T. Kelley, and Hans Petter
  Langtangen *Advances in Water Resources*, 51, 405-437,
* `Numerical modeling of drag for flow through vegetated domains and
  porous structures
  <http://dx.doi.org/10.1016/j.advwatres.2012.01.002>`_ (2012)
  S.A. Mattis, C. N. Dawson, C. E. Kees, and M. W. Farthing, *Advances
  in Water Resources*, 39, pp44-59
* `Parallel Computational Methods and Simulation for Coastal and
  Hydraulic Applications Using the Proteus Toolkit
  <http://www.dlr.de/sc/en/Portaldata/15/Resources/dokumente/pyhpc2011/submissions/pyhpc2011_submission_11.pdf>`_
  (2011) C. E. Kees and M. W. Farthing (2011) *Supercomputing11:
  Proceedings of the PyHPC11 Workshop*
* `A Conservative Level Set Method for Variable-Order Approximations
  and Unstructured Meshes
  <http://dx.doi.org/10.1016/j.jcp.2011.02.030>`_ (2011) C.E. Kees,
  I. Akkerman, Y. Bazilevs, and M. W. Farthing *Journal of
  Computational Physics* 230(12), pp4536–4558
* `Locally Conservative, Stabilized Finite Element Methods For
  Variably Saturated Flow
  <http://dx.doi.org/10.1016/j.cma.2008.06.005>`_ (2008) Kees, C.E.,
  M. W. Farthing, and C. N. Dawson, *Computer Methods in Applied
  Mechanics and Engineering*, 197, pp4610-4625
* `Locally Conservative, Stabilized Finite Element Methods for a Class
  of Variable Coefficient Navier-Stokes Equations
  <http://chl.erdc.usace.army.mil/ERDC-CHL-TR-09-12>`_ (2009)
  C. E. Kees, M. W. Farthing, and M. T. Fong, *ERDC/CHL TR-09-12*
* `Evaluating Finite Element Methods for the Level Set Equation
  <http://chl.erdc.usace.army.mil/ERDC-CHL-TR-09-11>`_ (2009)
  M. W. Farthing and C. E. Kees, *ERDC/CHL TR-09-11*
* `A Review of Methods for Moving Boundary Problems
  <http://chl.erdc.usace.army.mil/ERDC-CHL-TR-09-10>`_ (2009)
  C. E. Kees, M. W. Farthing, T. C. Lackey, and R. C. Berger,
  *ERDC/CHL TR-09-10*
* `Implementation of Discontinuous Galerkin Methods for the Level Set
  Equation on Unstructured Meshes
  <http://chl.erdc.usace.army.mil/library/publications/chetn/pdf/chetn-xiii-2.pdf>`_
  (2008) M. W. Farthing and C. E. Kees, *ERDC/CHL CHETN-XIII-2*

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Source Code Documentation
=========================

.. toctree::
   :maxdepth: 1
   :caption: General
   :name: sec-general

   API docs <apicapi>

.. toctree::
   :maxdepth: 1
   :caption: Tools
   :name: sec-tools

   tools/spatial_tools
   tools/wave_tools
   tools/gauges

.. toctree::
   :maxdepth: 1
   :caption: Models
   :name: sec-models

   models/adr
   models/navier_stokes
   models/SWFlow
   models/free_surface
   models/body_dynamics
