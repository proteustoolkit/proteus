.. _SWFlows:

Shallow Water Flow
*******************


Description
===========

There are currently 2 different models that describe shallow water flow in Proteus:

* Classical Saint-Venant equations (e.g. Shallow Water equations)
* Dispersive shallow water model based on the Green-Naghdi equations (e.g. mGN equations)

Shallow Water Equations
=======================
The Shallow Water Equations are a set of partial differential equations that form a
hyperbolic system. They can be used to describe a body of water evolving under
the action of gravity under the assumption that the deformations of the free surface
are small compared to the water height.

The implementation of the SWEs with source documentation is
available here: :py:mod:`proteus.kernels.SW2DCV`.



Modified Green-Naghdi equations
===============================
The modified Green-Naghdi equations are a set of partial differential equations
that form a hyperbolic system and are an O(h) approximation to the traditional
Green-Naghdi equations, where h is the mesh size. The Green-Naghdi equations
are used to describe dispersive water waves.

The implementation of the mGN equations with source documentation is
available here: :py:mod:`proteus.kernels.GN_SW2DCV`.


Running the tests
=================
All tests that concern shallow water flows can be found at :py:mod:`proteus.tests.SWFlow`.
