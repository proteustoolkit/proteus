"""Stably stratified hydrostatic column (Barua & Kees, compatible coupling).

A closed, partly saturated column holds a salinity that decreases linearly upward, c = c_bottom (1 - z/H),
in hydrostatic equilibrium with the variable density rho_T(c). With no molecular diffusion
(the dispersion vanishes with the velocity), the exact solution is stationary, and so is the
compatible discrete one: the edge-based flow sees no potential jump and the transport no exchange.
"""
from proteus.default_so import *

try:
    from . import column_flow_p, column_flow_n, column_transport_p, column_transport_n
except ImportError:
    import column_flow_p, column_flow_n, column_transport_p, column_transport_n

pnList = [("column_flow_p", "column_flow_n"),
          ("column_transport_p", "column_transport_n")]

systemStepControllerType = Sequential_MinModelStep
systemStepExact = True
name = "stratified_column"
needEBQ_GLOBAL = False
needEBQ = False

T = 10.0
nDTout = 10
tnList = [0.0, 1.0e-8, 2.0e-8] + [i * T / nDTout for i in range(1, nDTout + 1)]
archiveFlag = ArchiveFlags.EVERY_USER_STEP
useOneMesh = True
useOneArchive = True
