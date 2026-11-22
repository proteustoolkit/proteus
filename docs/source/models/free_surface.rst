Free Surface
************


There are two implementations for dealing with the free surface.


VOF - NCLS
==========

In this implementation, the following 4 models must be solved in order:

* Volume of Fluid (VOF): :py:mod:`proteus.kernels.VOF`
* Non-conservative level set (NCLS): :py:mod:`proteus.kernels.NCLS`
* Redistancing: :py:mod:`proteus.kernels.RDLS`
* Mass correction: :py:mod:`proteus.kernels.MCorr`


CLSVOF
======

In this implementation, a conservative level set is used and only the following
model must be solved: :py:mod:`proteus.kernels.CLSVOF`.
