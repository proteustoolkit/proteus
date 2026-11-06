.. _navier_stokes:

Multiphase Navier-Stokes
************************


Description
===========

There are currently 3 implementations of Navier-Stokes equations in proteus:

* Two-phase flow (e.g. air/water)
* Three-phase flow (e.g. air/water/sediment)
* Two-phase flow with immersed boundaries (solid)

Two-Phase
=========

The two-phase implementation of Navier-Stokes, with source documentation
available here: :py:mod:`proteus.mprans.RANS2P`.


Three-Phase
===========

The three-phase implementation of Navier-Stokes, with source documentation
available here: :py:mod:`proteus.mprans.RANS3P`.


Dealing with Moving Domains
===========================

When dealing with moving domains, the option ``movingDomain`` must be set to
``True``. This is necessary to signal to the model that mesh nodes velocity is
to be expected from an external model.


Moving (ALE) Mesh
-----------------

In the current implementation, if a model for moving the mesh is used such as
:py:mod:`proteus.mprans.MoveMesh`, it should be the first model to be solved,
as the mesh velocity is calculated from the previous time step.


Immersed Boundaries
-------------------

The two-phase implementation of Navier-Stokes with immersed solid boundaries, with
source documentation available here: :py:mod:`proteus.mprans.RANS2P_IB`. This is a different
method from the immersed finite elements of the ADR model (:ref:`adr`).
