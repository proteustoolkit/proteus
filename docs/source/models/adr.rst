.. _adr:

Advection-Diffusion-Reaction with Unfitted Interfaces
*****************************************************


Description
===========

The ADR model (:py:mod:`proteus.ADR`) solves linear advection-diffusion-reaction equations

.. math::

   \nabla\cdot(\mathbf{v}u - a\nabla u) = f

on a background mesh. Besides the standard formulation, it can treat two kinds of interfaces
that the mesh does not follow. Both are described by signed distance functions, so moving an
interface needs no new mesh.

* **Embedded boundary** :math:`\Gamma_s` (CutFEM). The part of the domain where the signed
  distance is negative (a solid) is removed. A Dirichlet value is imposed weakly on
  :math:`\Gamma_s` by Nitsche's method, or a Robin condition is applied instead. A ghost
  penalty on the interior edges of the active mesh that belong to a cut cell keeps cells with
  tiny fluid parts stable.
* **Immersed interface** :math:`\Gamma_f` (immersed finite elements, IFEM). The diffusion
  coefficient jumps from :math:`\mu_a` to :math:`\mu_b` across :math:`\Gamma_f`. Cut cells use a
  two-branch IFEM basis, for :math:`P_1` and :math:`P_2`, that satisfies the interface
  conditions. The symmetric and consistent edge terms of SCIFEM restore consistency on edges
  cut by :math:`\Gamma_f`, and an interior penalty stabilizes them for :math:`P_2`. Prescribed
  jumps of the solution and of the flux are supported.

With both interfaces, as for a floating body at the water surface, the domain holds two fluid
phases and a solid, and a cell can be cut by both curves (a triply-cut cell). Such cells are
integrated with composite equivalent polynomials and receive a consistency correction on the
edges cut by :math:`\Gamma_f`. In the paper's tests this hybrid CutFEM-IFEM method converges at
the optimal orders for :math:`p=1,2` and coefficient contrasts up to 1000: :math:`p+1` in
:math:`L^2` and :math:`p` in the broken :math:`H^1` seminorm. The :math:`L^\infty` error also
decreases at order :math:`p+1` on average, with fluctuating rates at the highest contrast.

All options are arguments of :py:class:`proteus.ADR.Coefficients`; its documentation lists
them with the values used in the paper below.


Running the test problems
=========================

The drivers are in ``test/ci``; the first two reproduce every table of the paper:

* ``ladr_3p_2d_p.py`` with ``ladr_3p_2d_c0pk_n.py``: the unfitted method. The option
  ``paper=True`` sets the paper's penalty parameters and edge-term choices.
* ``ladr_fitted_2d_p.py`` with ``ladr_fitted_2d_n.py``: the same problems on a mesh fitted to
  both interfaces, for comparison.
* ``ladr_ss_2d_p.py`` with ``ladr_ss_2d_c0p1_n.py`` or ``ladr_ss_2d_c0p2_n.py``: two-phase
  problems with an immersed interface only.

For example, Test 1 of the paper with quadratic elements on a structured mesh::

   % cd test/ci
   % parun ladr_3p_2d_p.py ladr_3p_2d_c0pk_n.py -C "paper=True test=2.0 pOrder=2 nCells=40"

The test case is selected with ``test`` (``2.0``: Test 1; ``2.1`` with ``test21solutionJump``
and ``test21fluxJump``: Tests 2 and 3; ``3.0``: Test 4, Robin condition), the coefficient
contrast with ``mub``, the degree with ``pOrder``, and the mesh with ``nCells``,
``refinement`` and ``unstructured``. All options and their defaults are listed, with a short
description, at the top of ``ladr_3p_2d_p.py``. With an analytical solution the model logs the :math:`L^2` and
:math:`L^\infty` errors, and the broken :math:`H^1` seminorm when the exact gradient is
available.

In parallel, the ghost penalty and the SCIFEM edge terms need two layers of mesh overlap;
the drivers set this, and the model stops with an error if a mesh has fewer.

The tests in ``test/IFEM`` (``test_ifem.py`` for two phases, ``test_ifem_3p.py`` for three
phases, including two-rank runs) check these cases against stored values, and
``test/test_equivalent_polynomials.py`` checks the cut-cell integration.


Reference
=========

* D. Nathawani and C. E. Kees, *Hybrid CutFEM-IFEM model for embedded and immersed interfaces
  with triply-cut cells for the 2D Poisson problem* (2026), submitted.
